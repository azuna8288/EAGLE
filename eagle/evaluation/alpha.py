import json
import numpy as np
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--anwer_file_path', type=str, required=True)
args = parser.parse_args()


json_files=[
    args.anwer_file_path,
]


for jsonl_file in json_files:
    data=[]
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    alphas=[0 for _ in range(5)]
    alphas_num=[0 for _ in range(5)]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        ids = sum(datapoint["choices"][0]['idxs'])
        alpha=datapoint["choices"][0]['alpha']
        alpha_num = datapoint["choices"][0]['alpha_num']
        for i in range(len(alpha)):
            alphas[i]+=alpha[i]
            alphas_num[i] += alpha_num[i]


    ar=np.array(alphas)/np.array(alphas_num)
    print(np.round(ar, 2))