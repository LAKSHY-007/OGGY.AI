
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model")
tokenizer = AutoTokenizer.from_pretrained(r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model")

model.save_pretrained(r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model", safe_serialization=True)
tokenizer.save_pretrained(r"E:\Pro\NLPCbot\tinyllama-finetuned-20250514_184430\final_model")