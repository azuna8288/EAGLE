import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
model = EaModel.from_pretrained(
    base_model_path="/opt/tiger/mariana/EAGLE/hub/vicuna-13b-v1.3",
    ea_model_path="/opt/tiger/mariana/EAGLE/hub/EAGLE-Vicuna-13B-v1.3",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
    total_token=-1
)
model.eval()
your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).to("cuda:0")
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])