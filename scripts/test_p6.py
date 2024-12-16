import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import seed_models

model_path = "/opt/tiger/EAGLE/hub/P61_D73_8B_official"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
)

prompts = ["小炒肉怎么做？", "请用Python写一下快速排序"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
inputs.pop("token_type_ids")

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_tokens)