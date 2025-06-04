import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
adapter_path = r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model"
config = PeftConfig.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,
)


tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    use_fast=False
)

# Lora
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
if __name__ == "__main__":
    print(generate_response("who is Current president of India"))
