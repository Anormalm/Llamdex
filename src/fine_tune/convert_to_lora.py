# This code is used to convert the models saved in whole to LoRA part only.
# The models are saved in the folder structure.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

def process_folder(folder_path, base_model_name, mistral_models_path):
    model_final_path = os.path.join(folder_path, "model_final.pt")

    if not os.path.exists(model_final_path):
        print(f"Skipping {folder_path}: model_final.pt not found")
        return

    print(f"Processing {folder_path}")

    model = AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=mistral_models_path,
                                                 torch_dtype=torch.bfloat16)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, config)

    try:
        state_dict = torch.load(model_final_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading {model_final_path}: {str(e)}")
        print("Skipping this folder due to file loading error.")
        return

    model.save_pretrained(folder_path)

    os.remove(model_final_path)
    print(f"Processed {folder_path}: Saved LoRA weights and removed model_final.pt")


def main():
    base_dir = "model/llm"
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    mistral_models_path = "model/llm"

    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("exp-lora-syn_mistral"):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):
                process_folder(folder_path, base_model_name, mistral_models_path)


if __name__ == "__main__":
    main()
