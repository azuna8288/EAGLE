import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import seed_models

model_path = "/opt/tiger/mariana/EAGLE/241114_3b3_sft30_12b-kd-bo128_hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

prompts = ["小炒肉怎么做？", "请用Python写一下快速排序"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
inputs.pop("token_type_ids")

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_tokens)