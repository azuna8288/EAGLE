import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

import seed_models


def qs_to_input_ids(tokenizer, question):
    role_split_text = "\n"


    prompt_text = f"{tokenizer.bos_token}user" \
                f"{role_split_text}{question}{tokenizer.eos_token}"\
                f"{tokenizer.bos_token}assistant{role_split_text}"

    return prompt_text


model_path = f"{os.environ['HOME_DIR']}/hub/3b3_moe_p6"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
)

prompts = '''Develop a Python program that reads all the text files under a directory and returns top-5 words with the most number of occurrences.", "Can you parallelize it?"], "reference": ["Can be simple solutions like using Counter\n\nSample answer:\n```\nimport os\nimport re\nfrom collections import Counter\ndef get_files_in_directory(directory):\n    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]\ndef read_file(file_path):\n    with open(file_path, 'r', encoding='utf-8') as file:\n        return file.read()\ndef count_words(text):\n    words = re.findall(r'\\w+', text.lower())\n    return Counter(words)\ndef main():\n    directory = input(\"Enter the directory path: \")\n    files = get_files_in_directory(directory)\n    word_counts = Counter()\n    for file in files:\n        text = read_file(file)\n        word_counts += count_words(text)\n    top_5_words = word_counts.most_common(5)\n    print(\"Top 5 words with the most number of occurrences:\")\n    for word, count in top_5_words:\n        print(f\"{word}: {count}\")\nif __name__ == \"__main__\":\n    main()\n```", "You should carefully check whether the parallelization logic is correct and choose the faster implementation.\n\nSample answer:\n```\nimport os\nimport re\nfrom collections import Counter\nimport concurrent.futures\ndef get_files_in_directory(directory):\n    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]\ndef read_file(file_path):\n    with open(file_path, 'r', encoding='utf-8') as file:\n        return file.read()\ndef count_words(text):\n    words = re.findall(r'\\w+', text.lower())\n    return Counter(words)\ndef process_file(file):\n    text = read_file(file)\n    return count_words(text)\ndef main():\n    directory = input(\"Enter the directory path: \")\n    files = get_files_in_directory(directory)\n    word_counts = Counter()\n    with concurrent.futures.ThreadPoolExecutor() as executor:\n        future_word_counts = {executor.submit(process_file, file): file for file in files}\n        for future in concurrent.futures.as_completed(future_word_counts):\n            word_counts += future.result()\n    top_5_words = word_counts.most_common(5)\n    print(\"Top 5 words with the most number of occurrences:\")\n    for word, count in top_5_words:\n        print(f\"{word}: {count}\")\nif __name__ == \"__main__\":\n    main()\n```'''
prompts = qs_to_input_ids(tokenizer, prompts)
inputs = tokenizer([prompts], return_tensors="pt", padding=True).to("cuda")
inputs.pop("token_type_ids")

generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
generated_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_tokens)