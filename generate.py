import sys
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from speculative_sampling import speculative_sampling
from eagle.model.ea_model import EaModel
import json

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--data_file', default='rd_test.jsonl', help='Input prompt')
parser.add_argument('--max_new_tokens', type=int, default=4096, help='No. of max new tokens')
parser.add_argument('--target_model', default="hub/3b3_moe_p6", help='Target model (HF Causal LM model)')
parser.add_argument('--draft_model', default='output/latest', help='Draft model (HF Causal LM model)')
parser.add_argument('--step_size', default=2, type=int, help='Step size')
parser.add_argument('--temperature', default=1.0, type=float, help='Temperature')
parser.add_argument('--top_p', default=0.7, type=float, help='Temperature')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


if args.draft_model is None:
    print("Draft model should be specified for Speculative Sampling")
    sys.exit(1)

print("Using target model:", args.target_model)
print("Using draft model:", args.draft_model)

# target_model = AutoModelForCausalLM.from_pretrained(args.target_model).to(device)
# draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model).to(device)

ea_model = EaModel.from_pretrained(base_model_path=args.target_model,
                        ea_model_path=args.draft_model,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        # load_in_8bit=True,
                        device_map="auto")

target_model = ea_model.base_model
draft_model = ea_model.ea_layer

tokenizer = AutoTokenizer.from_pretrained(args.target_model)

prompts = []
with open(args.data_file, "r") as f:
    for line in f.readlines():
        line = json.loads(line)
        prompts.append(line["prompt"][0])

accept_len_list = []

for prompt in tqdm(prompts):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_text, curr_accept_len_list = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), tokenizer=tokenizer, temperature=args.temperature, top_p=args.top_p, debug=False)
    accept_len_list = [*accept_len_list, *curr_accept_len_list]
    print(generated_text)
    print(f'平均接受长度是 {len(accept_len_list)/len(accept_len_list):.3f} 接受次数为 {len(accept_len_list)}')