import sys
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from speculative_sampling import speculative_sampling
from eagle.model.ea_model import EaModel

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--prompt', default='Sampling Method (autogressive / speculative)', help='Input prompt')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='No. of max new tokens')
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


for prompt in tqdm(['<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>user\n在中国双方订婚后分手，男方有权要回彩礼吗？<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n']):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), tokenizer=tokenizer, temperature=args.temperature, top_p=args.top_p, debug=True)
    