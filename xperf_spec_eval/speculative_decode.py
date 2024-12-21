import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import itertools
import pdb

from common import Monitor, GenerationConfig
from typing import *
from n_grams_decoding import find_candidate_pred_tokens

# enable monitor, try to track perf data without affect e2e time
ENABLE_MONITOR = True
local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
torch.manual_seed(0)

def _is_int8_cache(quant_mode: nn.Module):
    return "C8" in quant_mode or "W8A8" in quant_mode


def _init_inputs_for_generation(prompts: List[str], tokenizer, max_input_len):
    token_out = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = token_out.input_ids.cuda()
    if input_ids.shape[-1] > max_input_len:
        input_ids = input_ids[:, -max_input_len:]
    return input_ids


def _sample(probs, do_sample):
    if do_sample:
        probs = probs / probs.sum()
        noise = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / noise, dim=-1, keepdim=False)
    else:
        return torch.argmax(probs, dim=-1, keepdim=False)


def _accept_score(
    draft_prob: torch.Tensor,
    target_prob: torch.Tensor,
    input_id: torch.Tensor,
    do_sample: bool,
):
    batch_size = draft_prob.shape[0]
    # do formal spec decode
    if do_sample:
        random_number = torch.rand(batch_size, device=draft_prob.device)
        draft_prob = draft_prob[:, input_id]
        target_prob = target_prob[:, input_id]
        accept_matrix = random_number < (target_prob / draft_prob).squeeze(1)
        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"accept_matrix: {accept_matrix.tolist()} \ndraft_prob: {draft_prob.tolist()} \ntarget_prob: {target_prob.tolist()} \nrandom_number: {random_number.tolist()}"
            )
    # any token doesn't match target model greedy results will be rejected
    else:
        target_tokens = torch.argmax(target_prob, dim=-1)
        accept_matrix = input_id.unsqueeze(1) == target_tokens
        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"accept_matrix: {accept_matrix.tolist()}\ntarget_tokens: {target_tokens.tolist()} \ncandidates: {input_id.tolist()} \n"
            )

    return accept_matrix[0]


def _logits_processor(token_scores: torch.Tensor, generation_config: GenerationConfig):
    token_scores = token_scores.contiguous()
    if not generation_config.do_sample:
        return token_scores
    if generation_config.temperature > 0 and generation_config.temperature != 1.0:
        token_scores = token_scores / generation_config.temperature
    if generation_config.top_p > 0.0 and generation_config.top_p < 1.0:
        need_reshape = token_scores.dim() == 3
        batch_size = seq_len = vocab_size = 0
        if need_reshape:
            batch_size, seq_len, vocab_size = token_scores.shape
            token_scores = token_scores.reshape(batch_size * seq_len, vocab_size)
        sorted_scores, sorted_indices = torch.sort(token_scores, dim=-1, descending=True)
        token_scores = torch.classes.XGPT.TopkTopp().topp_sample(
            token_scores,
            sorted_scores,
            sorted_indices,
            generation_config.top_p,
            -float("Inf"),
            1,
        )
        if need_reshape:
            token_scores = token_scores.reshape(batch_size, seq_len, vocab_size)
    return token_scores


def _reject_sampling(
    actual_draft_step,
    draft_context_shift,
    draft_token_probs,
    target_token_probs,
    target_hidden_states,
    candidates_input_ids,
    output_ids,
    do_sample,
    tokenizer,
    context_len,
    max_new_tokens,
):
    finished = False
    accepted_len = 0
    total_draft_step = candidates_input_ids.shape[-1]
    n_grams_step = candidates_input_ids.shape[-1] - actual_draft_step
    for step_id in range(total_draft_step):
        draft_prob = draft_token_probs[step_id]
        target_prob = target_token_probs[step_id : step_id + 1, :]
        draft_token = candidates_input_ids[:, step_id]
        accepted = _accept_score(draft_prob, target_prob, draft_token, do_sample)
        if accepted:
            if logging.root.level <= logging.DEBUG and local_rank == 0:
                logging.debug(f"token {draft_token.item()} get accepted")
            accepted_len += 1
            if draft_token == tokenizer.eos_token_id:
                finished = True
                break
            # all draft tokens get accepted, get extra token from target_prob
            if step_id == total_draft_step - 1:
                # store last token
                if (
                    draft_context_shift - context_len + 1 + n_grams_step
                    >= max_new_tokens
                ):
                    finished = True
                    break
                extra_target_prob = target_token_probs[total_draft_step, :]
                next_tokens = _sample(
                    extra_target_prob.unsqueeze(0), do_sample
                ).unsqueeze(0)
                if next_tokens[0].item() == tokenizer.eos_token_id:
                    finished = True
                output_ids[
                    :,
                    draft_context_shift - context_len + 1 + n_grams_step,
                ] = next_tokens[0]
                if logging.root.level <= logging.DEBUG and local_rank == 0:
                    logging.debug(
                        f"all accepted, generate extra token: {next_tokens[0].item()}"
                    )
        else:
            if do_sample:
                next_tokens = _sample(
                    torch.clamp(target_prob[0] - draft_prob[0], min=0).unsqueeze(0),
                    do_sample,
                )
            else:
                next_tokens = _sample(target_prob, do_sample)

            if next_tokens[0].item() == tokenizer.eos_token_id:
                finished = True
            output_ids[
                :,
                draft_context_shift
                - context_len
                - actual_draft_step
                + accepted_len
                + 1,
            ] = next_tokens[0]
            if logging.root.level <= logging.DEBUG and local_rank == 0:
                logging.debug(
                    f"token {draft_token.item()} get rejected by token {next_tokens[0].item()}"
                )
            break
    # gather tokens for next round draft_inputs
    if accepted_len == total_draft_step:
        draft_input_ids = output_ids[
            :,
            draft_context_shift
            - context_len : draft_context_shift
            - context_len
            + 2
            + n_grams_step,
        ]
        if target_hidden_states is not None:
            target_hidden_states = target_hidden_states[-(2 + n_grams_step) :, :]
    else:
        if accepted_len < actual_draft_step:
            draft_context_shift = (
                draft_context_shift - actual_draft_step + accepted_len + 1
            )
            draft_input_ids = output_ids[
                :,
                draft_context_shift
                - context_len : draft_context_shift
                - context_len
                + 1,
            ]
            if target_hidden_states is not None:
                target_hidden_states = target_hidden_states[
                    accepted_len : accepted_len + 1, :
                ]
        else:
            draft_input_ids = output_ids[
                :,
                draft_context_shift
                - context_len : draft_context_shift
                - context_len
                + (accepted_len - actual_draft_step)
                + 2,
            ]
            if target_hidden_states is not None:
                target_hidden_states = target_hidden_states[
                    actual_draft_step
                    - 1 : actual_draft_step
                    + (accepted_len - actual_draft_step)
                    + 1,
                    :,
                ]

    # add the one generated by target itself
    accepted_len += 1

    return (
        finished,
        draft_context_shift,
        draft_input_ids,
        target_hidden_states,
        accepted_len,
    )

def speculative_perf(
    prompts: List[str],
    draft_model,
    target_model,
    generation_config: GenerationConfig,
    tokenizer,
):
    input_ids = _init_inputs_for_generation(
        prompts, tokenizer, generation_config.max_input_len
    )
    context_len = input_ids.shape[1]
    ite = 1
    g_start = torch.cuda.Event(enable_timing=True)
    g_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    # perf
    torch.manual_seed(0)
    g_start.record()
    for _ in range(ite):
        output, monitor = _speculative_decode(
            input_ids,
            None,
            draft_model,
            target_model,
            generation_config,
            tokenizer,
        )
    g_end.record()
    torch.cuda.synchronize()
    e2e_time = g_start.elapsed_time(g_end) / ite

    target_tokens = output[:, :]
    monitor.output_len = output.shape[1] - context_len
    # calculate accept_rate only for bs=1 case
    if generation_config.max_input_len<=4096 and output.shape[0] == 1:
        monitor.accept_rates = get_spec_accept_rate(
            target_tokens, context_len, draft_model, target_model, generation_config
        )

    per_step_time = e2e_time / (output.shape[1] - context_len)
    if local_rank == 0:
        logging.info(f"{monitor.description()}")
        logging.info(
            f"input_shape: {input_ids.shape} output_shape: {output.shape} e2e_time: {e2e_time}ms per_step: {per_step_time}ms"
        )

    results = tokenizer.batch_decode(output[:, context_len:], skip_special_tokens=False)
    return results, input_ids.shape, output.shape, e2e_time, per_step_time, monitor


def _draft_run(
    draft_model: nn.Module,
    draft_input_ids: torch.Tensor,
    target_hidden_states: torch.Tensor,
    kv_cache_index: torch.Tensor,
    output_ids: torch.Tensor,
    is_eagle_llm: bool,
    context_len: int,
    draft_context_shift: int,
    generation_config: GenerationConfig,
):
    draft_token_probs = []
    actual_draft_step = 0
    draft_hidden_states = None
    for step in range(generation_config.step_size):
        # stop generation when hit the max_len limitation
        if (
            draft_context_shift - context_len
            >= generation_config.max_new_tokens - draft_input_ids.shape[-1]
        ):
            break

        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"draft step: {step}\ndraft_inputs:\ninput_ids:{draft_input_ids.tolist()}\ndraft_context_shift:{draft_context_shift}"
            )

        eagle_hidden_states = None
        if is_eagle_llm:
            if step == 0:
                eagle_hidden_states = target_hidden_states
            else:
                eagle_hidden_states = draft_hidden_states[-1:, :]

        next_token_scores, draft_hidden_states = draft_model.forward_orca(
            context_input_ids=draft_input_ids,
            eagle_hidden_states=eagle_hidden_states,
            context_input_embeds=None,
            decode_input_ids=None,
            total_length=torch.tensor(
                [draft_input_ids.shape[-1]],
                dtype=torch.int,
                device=draft_input_ids.device,
            ),
            context_shifts=torch.tensor(
                [draft_context_shift], dtype=torch.int, device=draft_input_ids.device
            ),
            kv_cache_index=kv_cache_index,
            return_full_hidden_states=True,
            return_full_hidden_states_after_layernorm=generation_config.return_full_hidden_states_after_layernorm,
            orca_updated=True,
        )
        next_token_scores = _logits_processor(next_token_scores, generation_config)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        draft_token_probs.append(probs)
        next_tokens = _sample(probs, generation_config.do_sample)
        # probs = torch.zeros_like(next_token_scores)
        # next_tokens = torch.argmax(next_token_scores, dim=-1, keepdim=False)
        # probs[:, next_tokens] = 1.0
        # draft_token_probs.append(probs)
        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(f"draft next_tokens: {next_tokens.tolist()}")
        # update context_shifts
        draft_context_shift += draft_input_ids.shape[-1]
        output_ids[:, draft_context_shift - context_len] = next_tokens

        draft_input_ids = next_tokens.unsqueeze(0)
        actual_draft_step += 1
    return draft_token_probs, actual_draft_step, draft_context_shift


def _target_run(
    target_model: nn.Module,
    target_input_ids: torch.Tensor,
    kv_cache_index: torch.Tensor,
    desired_logits_indices: torch.Tensor,
    target_context_shift: int,
    is_eagle_llm: bool,
    generation_config: GenerationConfig,
):
    if logging.root.level <= logging.DEBUG and local_rank == 0:
        logging.debug(
            f"target_input_ids: {target_input_ids.tolist()} target_context_shift: {target_context_shift}"
        )
    results = target_model.forward_orca(
        context_input_ids=target_input_ids,
        decode_input_ids=None,
        total_length=torch.tensor(
            [target_input_ids.shape[-1]],
            dtype=torch.int,
            device=target_input_ids.device,
        ),
        context_shifts=torch.tensor(
            [target_context_shift], dtype=torch.int, device=target_input_ids.device
        ),
        kv_cache_index=kv_cache_index,
        return_full_hidden_states=is_eagle_llm,
        last_token_only=not is_eagle_llm,
        desired_logits_indices=desired_logits_indices,
        orca_updated=True,
    )

    if isinstance(results, tuple):
        target_token_scores, target_hidden_states = results
    else:
        target_token_scores = results
        target_hidden_states = None

    target_token_scores = _logits_processor(target_token_scores, generation_config)
    target_token_probs = nn.functional.softmax(target_token_scores, dim=-1)

    return target_input_ids, target_token_probs, target_hidden_states


def _speculative_decode(
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    draft_model: nn.Module,
    target_model: nn.Module,
    generation_config: GenerationConfig,
    tokenizer,
):
    monitor = Monitor()
    monitor.enable_track = ENABLE_MONITOR
    is_eagle_llm = generation_config.is_eagle_llm

    # prepare output_ids
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "please use LLMServer to test bs > 1"
    context_len = input_ids.shape[-1]
    max_new_tokens = min(
        generation_config.max_new_tokens, draft_model.config.max_length - context_len
    )
    if max_new_tokens <= 0:
        logging.error(
            f"prompts({input_ids.shape}) too long, exceed the model max_length({draft_model.config.max_length}"
        )
        return None, None
    output_ids = torch.zeros(
        (batch_size, max_new_tokens), dtype=torch.long, device=input_ids.device
    )

    # first step, target_model generate first token
    kv_cache_index = torch.tensor([[0]], dtype=torch.int, device=input_ids.device)
    _, target_token_probs, target_hidden_states = _target_run(
        target_model=target_model,
        target_input_ids=input_ids,
        kv_cache_index=kv_cache_index,
        desired_logits_indices=None,
        target_context_shift=0,
        is_eagle_llm=is_eagle_llm,
        generation_config=generation_config,
    )
    if target_hidden_states is not None:
        target_hidden_states = torch.cat(
            [torch.zeros_like(target_hidden_states[:1, :]), target_hidden_states], dim=0
        )
    next_tokens = _sample(target_token_probs, generation_config.do_sample)
    output_ids[:, 0] = next_tokens

    # prepare inputs for draft
    draft_input_ids = torch.cat((input_ids, next_tokens.unsqueeze(1)), dim=1)
    draft_context_shift = 0
    if generation_config.share_kv:
        draft_input_ids = next_tokens.unsqueeze(1)
        target_hidden_states = target_hidden_states[context_len:, :]
        draft_context_shift = context_len
        target_kv_cache = target_model.module._get_kv_cache(
            target_model.module.num_layers - 1,
            is_cache_quantization=_is_int8_cache(target_model.module.config.quant_mode),
        )
        assert not _is_int8_cache(draft_model.module.config.quant_mode), "Draft only support bf16 cache"
        if _is_int8_cache(target_model.module.config.quant_mode):
            kv_cache_qscale = target_model.module._get_kv_cache_qscale(
                target_model.module.num_layers - 1
            )
            dequanted_kv_cache = (
                torch.classes.XGPT.OrcaAttention().dequant_kv_cache_reset_swizzle(
                    torch.tensor(
                        [0, context_len], dtype=torch.int, device=draft_input_ids.device
                    ),
                    None,
                    target_kv_cache,
                    kv_cache_index,
                    kv_cache_qscale,
                    batch_size,
                    True,
                    draft_model.config.max_length,
                    False,
                )
            )
            draft_model.module._set_kv_cache(0, dequanted_kv_cache)
        else:
            draft_model.module._set_kv_cache(0, target_kv_cache)

    spec_step = 0
    finished = False
    while True:
        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(f"++++++" * 10)
            logging.debug(f"spec step: {spec_step}")

        # draft provide candidates
        if not generation_config.share_kv and spec_step == 0:
            start, end = monitor.start_track(tag="draft_context")
        else:
            start, end = monitor.start_track(tag="draft_forward")
        draft_token_probs, actual_draft_step, draft_context_shift = _draft_run(
            draft_model=draft_model,
            draft_input_ids=draft_input_ids,
            target_hidden_states=target_hidden_states,
            kv_cache_index=kv_cache_index,
            output_ids=output_ids,
            is_eagle_llm=is_eagle_llm,
            context_len=context_len,
            draft_context_shift=draft_context_shift,
            generation_config=generation_config,
        )
        if not generation_config.share_kv and spec_step == 0:
            monitor.track_draft_context(start, end)
        else:
            monitor.track_draft(start, end, actual_draft_step)

        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"output_ids after draft generate: {output_ids[:, :draft_context_shift - context_len + 1].tolist()} \nactual_draft_step: {actual_draft_step} \ndraft_context_shift: {draft_context_shift}"
            )
        if actual_draft_step == 0:
            finished = True
            break

        monitor.total_spec_step += 1
        monitor.total_draft_step += actual_draft_step

        # target verify candidates
        target_context_shift = draft_context_shift - actual_draft_step
        target_input_ids = output_ids[
            :,
            target_context_shift
            - context_len : target_context_shift
            - context_len
            + actual_draft_step
            + 1,
        ]

        # n-grams to provide more candidates
        candidates_n_grams = None
        if generation_config.post_spec_ngrams:
            start, end = monitor.start_track(tag="ngrams_draft")
            candidates_n_grams = find_candidate_pred_tokens(
                input_ids=torch.cat(
                    [input_ids, output_ids[:, : draft_context_shift - context_len + 1]],
                    dim=-1,
                ),
                num_pred_tokens=generation_config.num_pred_tokens,
            )
            if candidates_n_grams is not None:
                candidates_len = candidates_n_grams.shape[-1]
                remaining_len = max_new_tokens - (draft_context_shift - context_len + 1)
                candidates_n_grams = candidates_n_grams[
                    :, : min(remaining_len, candidates_len)
                ]
                output_ids[
                    :,
                    draft_context_shift
                    - context_len
                    + 1 : draft_context_shift
                    - context_len
                    + 1
                    + candidates_n_grams.shape[-1],
                ] = candidates_n_grams
                target_input_ids = torch.cat(
                    [target_input_ids, candidates_n_grams], dim=-1
                )
                for idx in range(candidates_n_grams.shape[-1]):
                    probs = torch.zeros_like(draft_token_probs[0])
                    probs[:, candidates_n_grams[:, idx]] = 1.0
                    draft_token_probs.append(probs)
            monitor.track_ngrams(start, end)

        start, end = monitor.start_track(tag="target_forward")
        target_input_ids, target_token_probs, target_hidden_states = _target_run(
            target_model=target_model,
            target_input_ids=target_input_ids,
            kv_cache_index=kv_cache_index,
            desired_logits_indices=torch.arange(
                target_input_ids.shape[-1],
                dtype=torch.long,
                device=target_input_ids.device,
            ),
            target_context_shift=target_context_shift,
            is_eagle_llm=is_eagle_llm,
            generation_config=generation_config,
        )
        monitor.track_target(start, end)

        # log stats
        monitor.total_target_step += 1

        candidates_input_ids = target_input_ids[:, 1:].clone()

        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"candidates_input_ids to be verified: {candidates_input_ids.tolist()}"
            )

        # Reject sampling
        start, end = monitor.start_track(tag="reject_sampling")
        (
            finished,
            draft_context_shift,
            draft_input_ids,
            target_hidden_states,
            accepted_len,
        ) = _reject_sampling(
            actual_draft_step=actual_draft_step,
            draft_context_shift=draft_context_shift,
            draft_token_probs=draft_token_probs,
            target_token_probs=target_token_probs,
            target_hidden_states=target_hidden_states,
            candidates_input_ids=candidates_input_ids,
            output_ids=output_ids,
            do_sample=generation_config.do_sample,
            tokenizer=tokenizer,
            context_len=context_len,
            max_new_tokens=max_new_tokens,
        )
        monitor.track_reject_sampling(start, end)
        if logging.root.level <= logging.DEBUG and local_rank == 0:
            logging.debug(
                f"output_ids after reject sampling: \n{output_ids[:, :draft_context_shift - context_len + draft_input_ids.shape[-1]]}"
            )
        monitor.accepted_len_stats.append(accepted_len)

        if finished:
            break

        spec_step += 1

    return (
        torch.cat(
            (
                input_ids,
                output_ids[:, : draft_context_shift - context_len + 1],
            ),
            dim=1,
        ),
        monitor,
    )


def get_spec_accept_rate(
    target_tokens,
    context_len,
    draft_model,
    target_model,
    generation_config: GenerationConfig,
):
    is_eagle_llm = generation_config.is_eagle_llm
    share_kv = generation_config.share_kv
    target_tokens = target_tokens.cuda()
    total_length = torch.tensor(
        [target_tokens.shape[-1]], dtype=torch.int, device=target_tokens.device
    )
    kv_cache_index = torch.tensor([[0]], dtype=torch.int, device=target_tokens.device)
    results = target_model.forward_orca(
        context_input_ids=target_tokens,
        decode_input_ids=None,
        total_length=total_length,
        kv_cache_index=kv_cache_index,
        orca_updated=True,
        last_token_only=not is_eagle_llm,
        return_full_hidden_states=is_eagle_llm,
        desired_logits_indices=torch.arange(
            target_tokens.shape[-1], dtype=torch.long, device=target_tokens.device
        ),
    )
    if isinstance(results, tuple):
        target_token_scores, target_hidden_states = results
    else:
        target_token_scores = results
        target_hidden_states = None
    eagle_hidden_states = None
    if is_eagle_llm:
        eagle_hidden_states = torch.cat(
            [
                torch.zeros_like(target_hidden_states[:1, :]),
                target_hidden_states[:-1, :],
            ],
            dim=0,
        )
        if share_kv:
            target_kv_cache = target_model.module._get_kv_cache(
                target_model.module.num_layers - 1,
                is_cache_quantization=_is_int8_cache(target_model.module.config.quant_mode),
            )
            assert not _is_int8_cache(draft_model.module.config.quant_mode), "Draft only support bf16 cache"
            if _is_int8_cache(target_model.module.config.quant_mode):
                kv_cache_qscale = target_model.module._get_kv_cache_qscale(
                    target_model.module.num_layers - 1
                )
                dequanted_kv_cache = (
                    torch.classes.XGPT.OrcaAttention().dequant_kv_cache_reset_swizzle(
                        torch.tensor(
                            [0, context_len], dtype=torch.int, device=target_tokens.device
                        ),
                        None,
                        target_kv_cache,
                        kv_cache_index,
                        kv_cache_qscale,
                        1,
                        True,
                        draft_model.config.max_length,
                        False,
                    )
                )
                draft_model.module._set_kv_cache(0, dequanted_kv_cache)
            else:
                draft_model.module._set_kv_cache(0, target_kv_cache)

    draft_tokens = target_tokens if not share_kv else target_tokens[:, context_len:]
    eagle_hidden_states = (
        eagle_hidden_states if not share_kv else eagle_hidden_states[context_len:, :]
    )
    total_length = torch.tensor(
        [draft_tokens.shape[-1]], dtype=torch.int, device=draft_tokens.device
    )
    context_shifts = (
        None
        if not share_kv
        else torch.tensor([context_len], dtype=torch.int, device=target_tokens.device)
    )

    draft_token_scores = draft_model.forward_orca(
        context_input_ids=draft_tokens,
        eagle_hidden_states=eagle_hidden_states,
        context_input_embeds=None,
        decode_input_ids=None,
        total_length=total_length,
        context_shifts=context_shifts,
        kv_cache_index=kv_cache_index,
        orca_updated=True,
        desired_logits_indices=torch.arange(
            draft_tokens.shape[-1], dtype=torch.long, device=draft_tokens.device
        ),
    )
    target_token_scores = target_token_scores / generation_config.temperature
    draft_token_scores = draft_token_scores / generation_config.temperature
    if generation_config.top_p > 0.0 and generation_config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(
            target_token_scores, dim=-1, descending=True
        )
        target_token_scores = torch.classes.XGPT.TopkTopp().topp_sample(
            target_token_scores,
            sorted_scores,
            sorted_indices,
            generation_config.top_p,
            -float("Inf"),
            1,
        )
        sorted_scores, sorted_indices = torch.sort(
            draft_token_scores, dim=-1, descending=True
        )
        draft_token_scores = torch.classes.XGPT.TopkTopp().topp_sample(
            draft_token_scores,
            sorted_scores,
            sorted_indices,
            generation_config.top_p,
            -float("Inf"),
            1,
        )
    target_token_probs = nn.functional.softmax(target_token_scores, dim=-1)
    draft_token_probs = nn.functional.softmax(draft_token_scores, dim=-1)

    if local_rank == 0:
        logging.info(
            f"context_len: {context_len} target_token_probs: {target_token_probs.shape} draft_token_probs: {draft_token_probs.shape}"
        )
    seq_len = target_token_probs.shape[0]
    accept_rates = []
    for seq_id in range(context_len, seq_len - 1):
        draft_seq_id = seq_id if not share_kv else seq_id - context_len
        accept_probs = draft_token_probs[draft_seq_id] * torch.clamp(
            target_token_probs[seq_id] / draft_token_probs[draft_seq_id], max=1
        )
        accept_probs = accept_probs[~torch.isnan(accept_probs)]
        accept_rates.append(torch.sum(accept_probs).item())
    return accept_rates
