# Copyright 2023 Mixtral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
We adapt P6 modeling from Mixtral on Huggingface.
The only difference between P6 and Mixtral modeling is
1. P6 uses LayerNorm while Mixtral uses RMSNorm
2. P6 has Key LayerNorm and Context Norm while Mixtral doesn't
"""

import inspect
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.distributed import nn as dist_nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from seed_models.integrations.xperf import XperfGenerationMixin
from seed_models.utils import _flash_attention_forward, is_fused_moe_available, logging
from seed_models.utils.modeling_outputs import MoeModelOutputWithPastAndAuxLosses
from seed_models.utils.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from seed_models.models.p6.configuration_p6 import P6Config


if is_fused_moe_available():
    from seed_models.utils import fused_moe_forward

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import (
        pad_input,  # noqa
    )

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
else:
    _flash_supports_window_size = None

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "P6Config"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->P6
# TODO @Arthur no longer copied from LLama after static cache
class P6RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[P6Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`P6RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# TODO @Arthur no longer copied from LLama after static cache
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, interleaved_kv_shared: bool = True) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    if interleaved_kv_shared:
        hidden_states = hidden_states[:, None, :, :, :].expand(batch, n_rep, num_key_value_heads, slen, head_dim)
    else:
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class P6Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: P6Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        if self.config.use_context_groupnorm:
            self.context_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)

        if self.config.use_key_layernorm:
            self.key_layernorm = nn.LayerNorm(
                self.head_dim,
                eps=self.config.layer_norm_eps,
            )
        self.interleaved_kv_shared = self.config.interleaved_kv_shared

        self.rotary_emb = P6RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        past_key_values_length = 0 if not use_cache else past_key_value.get_usable_length(q_len, self.layer_idx)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (bsz, q_len),
            hidden_states,
            past_key_values_length,
            sliding_window=None if self.config.sliding_window is None else self.config.sliding_window[self.layer_idx],
        )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config.use_key_layernorm:
            key_states = self.key_layernorm(key_states)
            # in fsdp training mode, the norm will be autocasted to float32
            if key_states.dtype != query_states.dtype:
                key_states = key_states.to(query_states.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups, self.interleaved_kv_shared)
        value_states = repeat_kv(value_states, self.num_key_value_groups, self.interleaved_kv_shared)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.config.use_context_groupnorm:
            attn_output = self.context_norm(attn_output)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->P6
class P6FlashAttention2(P6Attention):
    """
    P6 flash attention module. This module inherits from `P6Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        max_seqlen: int = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if cu_seqlens is None:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len = cu_seqlens.diff().max().item()

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window[self.layer_idx]
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory"
                " efficient implementation make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            assert cu_seqlens is None, "`cu_seqlens` is not incompatible with past_key_value"
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window[self.layer_idx]
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window[self.layer_idx]

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window[self.layer_idx] - 1:
                    raise ValueError(
                        "past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1,"
                        f" head_dim`), got {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config.use_key_layernorm:
            key_states = self.key_layernorm(key_states)
            # in fsdp training mode, the norm will be autocasted to float32
            if key_states.dtype != query_states.dtype:
                key_states = key_states.to(query_states.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups, self.interleaved_kv_shared)
        value_states = repeat_kv(value_states, self.num_key_value_groups, self.interleaved_kv_shared)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=q_len,
            position_ids=position_ids,
            is_causal=self.is_causal,
            dropout=dropout_rate,
            cu_seqlens=cu_seqlens,
            sliding_window=self.config.sliding_window[self.layer_idx] if use_sliding_windows else None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            training=self.training,
            layer_number=self.layer_idx,
            max_seqlen=max_seqlen,
        )

        if self.config.use_context_groupnorm:
            attn_output = self.context_norm(attn_output)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->P6
class P6SdpaAttention(P6Attention):
    """
    P6 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `P6Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from P6Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "P6Model is using P6SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups, self.interleaved_kv_shared)
        value_states = repeat_kv(value_states, self.num_key_value_groups, self.interleaved_kv_shared)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.config.use_context_groupnorm:
            attn_output = self.context_norm(attn_output)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None, past_key_value


P6_ATTENTION_CLASSES = {
    "eager": P6Attention,
    "flash_attention_2": P6FlashAttention2,
    "sdpa": P6SdpaAttention,
}


class P6Experts(nn.Module):
    def __init__(self, config: P6Config):
        super().__init__()
        self.num_experts = config.moe_num_expert
        self.hidden_dim = config.hidden_size
        self.ffn_dim = int(config.intermediate_size)
        self.fc1_1 = torch.nn.Parameter(
            torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim),
            requires_grad=True,
        )
        self.fc1_2 = torch.nn.Parameter(
            torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim),
            requires_grad=True,
        )
        self.fc2 = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, expert_idx, hidden_states):
        fc1_1_out = torch.matmul(hidden_states, self.fc1_1[expert_idx].transpose(0, 1))
        fc1_2_out = torch.matmul(hidden_states, self.fc1_2[expert_idx].transpose(0, 1))

        fc1_out = self.act_fn(fc1_1_out) * fc1_2_out
        out = torch.matmul(fc1_out, self.fc2[expert_idx].transpose(0, 1))
        return out


# Auxiliary loss for Moe
class AuxLossBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, aux_loss, scale):
        # Preserve the aux_loss by storing it in the context to avoid garbage collection.
        ctx.aux_loss = aux_loss
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Scale the auxiliary loss like the main loss.
        scaled_aux_loss_grad = torch.ones_like(ctx.aux_loss) * ctx.scale
        return grad_output, scaled_aux_loss_grad, None


class P6TopkCapGate(nn.Module):
    def __init__(
        self,
        config,
        capacity_factor=1.0,
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.moe_num_expert
        self.top_k = config.moe_topk
        self.moe_freeze_gating = config.moe_freeze_gating

        self.capacity_factor = capacity_factor
        self.wg = nn.Parameter(torch.zeros(self.hidden_dim, self.num_experts), requires_grad=True)
        self.expert_caps = [1.0] * self.num_experts

        self.alpha = 0.95
        self.ce_alpha = 0.0
        self.over_compute = 1.2
        self.register_buffer("cal_weights", torch.tensor(self.expert_caps), persistent=False)
        self.register_buffer("wg_ema", torch.zeros(self.hidden_dim, self.num_experts))
        self.register_buffer("ce_ema", torch.zeros(self.num_experts))

        if self.moe_freeze_gating:
            self.wg.requires_grad = False

    def calc_metric(self, router_logits, selected_experts):
        mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1)
        me = torch.nn.functional.softmax(router_logits, dim=1).mean(dim=0)
        ce = mask.float().mean(dim=0)
        ce = ce * self.cal_weights
        with torch.no_grad():
            self.ce_ema = self.ce_alpha * self.ce_ema.float() + (1 - self.ce_alpha) * ce.float()
        hot_num, hot_exp = self.ce_ema.max(), self.ce_ema.argmax()
        cold_num, cold_exp = self.ce_ema.min(), self.ce_ema.argmin()
        aux_loss = torch.nn.functional.relu(hot_num - cold_num * self.over_compute) * (me[hot_exp] - me[cold_exp])
        with torch.no_grad():
            cap = torch.sum(ce)
        return aux_loss, cap

    def update_gate_ema(self, moe_freeze_gating):
        if self.training and torch.is_grad_enabled() and not moe_freeze_gating:
            with torch.no_grad():
                self.wg_ema = self.alpha * self.wg_ema + (1 - self.alpha) * self.wg

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_aux_losses: Optional[bool] = None,
    ):
        wg_running = 0.5 * (self.wg_ema + self.wg.float())
        router_logits = torch.matmul(hidden_states.float(), wg_running)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.cal_weights.dtype != torch.float32:
            self.cal_weights = self.cal_weights.to(torch.float32)
        if self.ce_ema.dtype != torch.float32:
            self.ce_ema = self.ce_ema.to(torch.float32)

        # If we are in training mode, we need to compute the auxiliary loss
        if output_aux_losses:
            aux_loss, _ = self.calc_metric(router_logits, selected_experts)
        else:
            aux_loss = None

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        self.update_gate_ema(self.moe_freeze_gating)
        return routing_weights, router_logits, aux_loss, expert_mask, selected_experts


class P6MoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_expert
        self.top_k = config.moe_topk
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # gating
        self.gate = P6TopkCapGate(config)
        self.experts = P6Experts(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_aux_losses: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, router_logits, aux_loss, expert_mask, _ = self.gate(hidden_states, output_aux_losses)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.experts(expert_idx, current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # We need to apply Auxiliary Loss backward hook to scale the gradients of the auxiliary loss
        # final_hidden_states = AuxLossBackwardHook.apply(final_hidden_states, aux_loss, self.router_aux_loss_coef)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits, aux_loss


class P6FusedMoeBlock(P6MoeBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_aux_losses: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # MOE Step 1: compute each token's weight for all experts.
        # router_logits shape (batch_size * sequence_len, num_experts)
        routing_weights, router_logits, aux_loss, _, selected_experts = self.gate(hidden_states, output_aux_losses)

        # MOE Step 2: compute experts with group gemm.
        final_hidden_states = fused_moe_forward(
            self.num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            self.experts.fc1_1,
            self.experts.fc1_2,
            self.experts.fc2,
        )

        # reshape output to input shape
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits, aux_loss


P6_MOE_CLASSES = {
    "eager": P6MoeBlock,
    "fused": P6FusedMoeBlock,
}


class P6MLP(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.moe = P6_MOE_CLASSES[config._moe_implementation](config)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx
        self.config = config

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        output_aux_losses: Optional[bool] = None,
    ) -> torch.FloatTensor:
        hidden_states, router_logits, aux_loss = self.moe(
            hidden_states,
            output_aux_losses,
        )
        hidden_states = self.dropout(hidden_states)

        return hidden_states, router_logits, aux_loss


class P6DecoderLayer(nn.Module):
    def __init__(self, config: P6Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # We only support FlashAttn2 and Eager here because sdpa doesn't support sliding window
        self.attn = P6_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = P6MLP(config)

        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        output_aux_losses: Optional[bool] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use"
                " `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states, router_logits, aux_loss = self.mlp(hidden_states, output_aux_losses)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        if output_aux_losses:
            outputs += (aux_loss,)

        return outputs


P6_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`P6Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare P6 Model outputting raw hidden-states without any specific head on top.",
    P6_START_DOCSTRING,
)
class P6PreTrainedModel(PreTrainedModel):
    config_class = P6Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["P6DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def post_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(substring in name for substring in ["o_proj", "fc2"]):
                module.std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)

        super().post_init()

    def _init_weights(self, module):
        if hasattr(module, "std"):
            std = module.std
        else:
            std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


P6_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare P6 Model outputting raw hidden-states without any specific head on top.",
    P6_START_DOCSTRING,
)
class P6Model(P6PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`P6DecoderLayer`]

    Args:
        config: P6Config
    """

    def __init__(self, config: P6Config):
        super().__init__(config)

        # check sliding window
        if isinstance(config.sliding_window, int):
            config.sliding_window = [config.sliding_window] * config.num_hidden_layers

        if config.sliding_window is not None:
            assert (
                len(config.sliding_window) == config.num_hidden_layers
            ), "len(config.sliding_window) must equal to config.num_hidden_layers"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embd_dropout = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([P6DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = P6RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    @add_start_docstrings_to_model_forward(P6_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_aux_losses: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPastAndAuxLosses]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            if (cu_seqlens is not None or position_ids is not None) and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            if (cu_seqlens is not None or position_ids is not None) and inputs_embeds.dim() == 1:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        assert not (use_cache and cu_seqlens is not None), "`cu_seqlens` is incompatible with `use_cache`"

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if cu_seqlens is None:
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                host_seqlens = cu_seqlens.diff().cpu()
                position_ids = torch.cat([torch.arange(l, dtype=torch.long, device=device) for l in host_seqlens])
                position_ids = position_ids.unsqueeze(0)
            max_seqlen = None
        else:
            position_ids = position_ids.view(batch_size, -1).long()
            max_seqlen = position_ids.max().item() + 1

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        inputs_embeds = self.embd_dropout(inputs_embeds)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of P6. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        assert self._attn_implementation in [
            "flash_attention_2",
            "eager",
        ], "Only support flash_attention_2 and eager implementation for P6"
        assert (
            cu_seqlens is None or self._attn_implementation == "flash_attention_2"
        ), "`seqlens` is only supported in flash_attention_2"

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        # for _attn_implementation is "eager", no need to prepare attention mask here, but in attention step.

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_aux_losses = () if output_aux_losses else None
        next_decoder_cache = None

        for decoder_layer in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cu_seqlens,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    output_aux_losses,
                    use_cache,
                    position_embeddings,
                    max_seqlen,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    output_aux_losses=output_aux_losses,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    max_seqlen=max_seqlen,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-2 if output_aux_losses else -1],)

            if output_aux_losses:
                all_aux_losses += (layer_outputs[-1],)

        hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPastAndAuxLosses(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            aux_losses=all_aux_losses,
        )


class P6ForCausalLM(P6PreTrainedModel, XperfGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = P6Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.moe_num_expert
        self.moe_topk = config.moe_topk
        self.loss_fct = CrossEntropyLoss()
        # Initialize weights and apply final processing
        self.post_init()
        self._reset_xperf()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    @add_start_docstrings_to_model_forward(P6_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_aux_losses: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_aux_losses=output_aux_losses,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            if cu_seqlens is not None:
                # Mask the last token of each sequence to torch.CrossEntropyLoss ignore_index, default is -100
                shift_labels[cu_seqlens[1:-1] - 1] = -100
            elif position_ids is not None:
                position_ids_ = position_ids.flatten()
                indices_q = torch.arange(position_ids_.size(0), device=position_ids_.device, dtype=torch.int32)
                cu_seq_lens = torch.cat(
                    (
                        indices_q[position_ids_ == 0],
                        torch.tensor(position_ids_.size(), device=position_ids_.device, dtype=torch.int32),
                    )
                )
                shift_labels[cu_seq_lens[1:-1] - 1] = -100

            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_aux_losses:
            # aux_loss: (Union[`torch.Tensor`, Tuple[torch.Tensor]), should be a tuple of model.config.num_hidden_layers tensors of aux_loss
            aux_losses = outputs.aux_losses
            compute_device = aux_losses[0].device
            aux_loss = sum(layer_aux_loss.to(compute_device) for layer_aux_loss in aux_losses)

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_aux_losses:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def get_xperf_compatible_state_dict(self):
        if self.num_experts > 0:
            from .convert_p6_checkpoint import convert_p6_moe_to_xperf_compatible

            return convert_p6_moe_to_xperf_compatible(self.state_dict(), self.num_experts)
        else:
            raise NotImplementedError("Xperf compatible state_dict not implemented for non-moe model")

    def get_xperf_config_file(self):
        from .convert_p6_checkpoint import p6_get_temporary_xperf_config

        return p6_get_temporary_xperf_config(self.config)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        **kwargs,
    ):
        if self._xperf_generate is None and self._xperf_optimized_model is None:
            self._init_xperf(generation_config, **kwargs)

        if self._xperf_optimized_model is not None:
            logger.info("Using xperf_gpt to generate")
            input_ids = kwargs.pop("input_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            local_generation_config = self._process_xperf_generate_kwargs(**kwargs)
            self._xperf_optimized_model.set_generator_strategy(**local_generation_config)
            out = self._xperf_optimized_model(input_ids, attention_mask)
        else:
            out = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        return out


@add_start_docstrings(
    """
    The P6 Model transformer with a sequence classification head on top (linear layer).

    [`P6ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    P6_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->P6, LLAMA->P6
class P6ForSequenceClassification(P6PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = P6Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    @add_start_docstrings_to_model_forward(P6_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The P6 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    P6_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->P6, LLAMA->P6
class P6ForTokenClassification(P6PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = P6Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(P6_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_aux_losses: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_aux_losses=output_aux_losses,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class P6ForTextEncoder(P6PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = P6Model(config)
        if getattr(config, "embedding_head_size", 0) > 0:
            self.emb_head = torch.nn.Linear(self.config.hidden_size, self.embedding_head_size, bias=False)
        else:
            self.emb_head = None
        # Initialize weights and apply final processing
        self.post_init()
        self.tau = getattr(config, "tau", 1.0)
        self.matryoshka_dims = getattr(config, "matryoshka_dims", [])
        self.matryoshka_weight = getattr(config, "matryoshka_weight", [])
        if self.matryoshka_dims:
            if self.matryoshka_weight:
                assert len(self.matryoshka_weight) == len(self.matryoshka_dims)
            else:
                self.matryoshka_weight = [1.0] * len(self.matryoshka_dims)

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_aux_losses: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query_index: Optional[torch.LongTensor] = None,
        doc_index: Optional[torch.LongTensor] = None,
        return_embeddings: Optional[bool] = False,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_aux_losses=output_aux_losses,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        embeddings = hidden_states[cu_seqlens[1:] - 1]
        if self.emb_head is not None:
            embeddings = self.emb_head(embeddings)
        loss = None
        if labels is not None:
            assert query_index is not None
            assert doc_index is not None
            rank, world_size = dist.get_rank(), dist.get_world_size()

            query_embeddings = embeddings[query_index]
            doc_embeddings = embeddings[doc_index]

            shadow_query_embeddings = query_embeddings.detach().requires_grad_()
            shadow_doc_embeddings = doc_embeddings.detach().requires_grad_()

            query_num = torch.tensor(query_embeddings.shape[0], dtype=torch.int, device=torch.device("cuda"))
            query_num_list = torch.zeros(world_size, dtype=torch.int, device=torch.device("cuda"))
            dist.all_gather_into_tensor(query_num_list, query_num)
            query_num_list = query_num_list.tolist()
            losses = []
            for curr_src, curr_query_embeddings in self._broadcast_embedding_iter(
                shadow_query_embeddings, query_num_list
            ):
                targets = labels if rank == curr_src else None

                if self.matryoshka_dims:
                    loss = 0
                    for dim, weight in zip(self.matryoshka_dims, self.matryoshka_weight):
                        loss += self._partial_loss_and_backward(
                            curr_query_embeddings[..., :dim],
                            shadow_doc_embeddings[..., :dim],
                            targets,
                            weight,
                        )
                else:
                    loss = self._partial_loss_and_backward(curr_query_embeddings, shadow_doc_embeddings, targets)
                losses.append(loss)

            loss = sum(losses) / query_num
            query_embedding_grad = shadow_query_embeddings.grad / query_num
            doc_embeddings_grad = shadow_doc_embeddings.grad / query_num
            loss = FakeLoss.apply(loss, query_embeddings, query_embedding_grad, doc_embeddings, doc_embeddings_grad)

        outputs = {"loss": loss}
        if return_embeddings:
            outputs["embeddings"] = embeddings

        return outputs

    def _partial_loss_and_backward(
        self,
        query_embeddings: torch.FloatTensor,
        doc_embeddings: torch.FloatTensor,
        targets: torch.IntTensor = None,
        weight: float = 1.0,
    ):
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, dim=-1)
        logits = query_embeddings @ doc_embeddings.t()
        logits = logits.float() / self.tau

        logits_max = logits.detach().max(dim=-1, keepdim=True)[0]  # no grad
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX)
        logits.sub_(logits_max)

        exp = logits.exp()
        exp_sum = exp.sum(dim=-1, keepdim=True)
        exp_sum = dist_nn.all_reduce(exp_sum, op=dist.ReduceOp.SUM)

        if targets is not None:
            loss = exp_sum.log() - logits.gather(dim=1, index=targets.unsqueeze(1))
        else:
            loss = 0 * exp_sum

        loss = loss.sum() * weight
        if torch.is_grad_enabled():
            loss_scale = float(os.environ.get("MAAS_FINETUNE_EMBEDDING_LOSS_SCALE", "1.0"))
            bwd_loss = loss
            if loss_scale != 1.0:
                bwd_loss = bwd_loss * loss_scale
            bwd_loss.backward()

        loss = loss.detach()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        return loss

    def _broadcast_embedding_iter(self, src_embedding: torch.Tensor, embedding_sizes):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert len(embedding_sizes) == world_size

        def async_broadcast_embedding(src):
            if rank == src:
                embeddings = src_embedding
            else:
                embeddings = torch.empty(
                    (embedding_sizes[src], src_embedding.shape[-1]),
                    dtype=src_embedding.dtype,
                    device=src_embedding.device,
                    requires_grad=src_embedding.requires_grad,
                )
            return async_broadcast(embeddings, src)

        next_src, (next_embeddings, next_work) = 0, async_broadcast_embedding(0)
        for src in range(1, world_size):
            curr_src, embeddings, work = next_src, next_embeddings, next_work
            next_src, (next_embeddings, next_work) = src, async_broadcast_embedding(src)
            work.wait()
            yield curr_src, embeddings
        next_work.wait()
        yield next_src, next_embeddings


class _AsyncBroadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        work = dist.broadcast(tensor, src, group=group, async_op=True)
        return tensor, work

    @staticmethod
    def backward(ctx, grad_output, noused):
        gx = dist_nn.functional._Reduce.apply(ctx.src, dist.ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.rank:
            gx.zero_()
        return (None, None, gx, None)


def async_broadcast(tensor, src, group=dist.group.WORLD):
    return _AsyncBroadcast.apply(src, group, tensor)


class FakeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, x, x_grad, y, y_grad):
        ctx.x_grad = x_grad
        ctx.y_grad = y_grad

        return loss.detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, ctx.x_grad, None, ctx.y_grad, None
