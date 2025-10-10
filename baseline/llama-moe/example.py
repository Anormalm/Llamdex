# python>=3.10
import os
from datetime import datetime

# change the root dir of transformer
os.environ["TRANSFORMERS_CACHE"] = "/home/zhaomin/project/SiT/cache/huggingface/hub/"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchviz import make_dot

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-4_16"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
)

# # Visualize the model
# example_input_text = "To grant sudo access for a user,"
# inputs_example = tokenizer(example_input_text, return_tensors="pt").to("cuda:7")
# model = model.to("cuda:7")
# pred_sample = model.generate(**inputs_example, max_length=50, temperature=0.0)
# make_dot(pred_sample, params=dict(model.named_parameters())).render("llama-moe-structure", format="png")

device = torch.device("cuda:1")
start_time = datetime.now()
model.eval()
model.to(device)
print("Loading time (s):", datetime.now() - start_time)

start_time = datetime.now()

input_text = "Wuhan is famous for three things: "
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to(device)

print("Tokenizing time (s):", datetime.now() - start_time)

print("Input text:", input_text)
start_time = datetime.now()
pred = model.generate(**inputs, max_length=100, temperature=0.0)
print("Inference time (s):", datetime.now() - start_time)

print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
