# train_model_tinyllama_final.py
import os
import torch
import logging
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model

# ===== Configuration =====
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "./processed_oasst1"
OUTPUT_DIR = "./tinyllama-finetuned"
LOGGING_DIR = "./logs"
MAX_LENGTH = 512

# ===== Setup Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def tokenize_function(examples, tokenizer):
    """Tokenize and prepare inputs"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False  # We'll pad during collation
    )

def main():
    set_seed(42)

    try:
        # ===== GPU Check =====
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"🚀 GPU Available: {use_cuda} | Using device: {device}")

        # ===== 1. Load Dataset =====
        logger.info("📂 Loading dataset...")
        dataset = load_from_disk(DATASET_PATH)
        logger.info("🔄 Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if use_cuda else "cpu",
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            low_cpu_mem_usage=True
        )

        # ===== 3. Tokenize Dataset =====
        logger.info("🔠 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            remove_columns=[col for col in dataset["train"].column_names if col != "message_tree_id"],
            num_proc=2
        )

        # ===== 4. LoRA Configuration =====
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # ===== 5. Training Arguments =====
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}-{timestamp}",
            per_device_train_batch_size=2 if use_cuda else 1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            num_train_epochs=3,
            logging_dir=LOGGING_DIR,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            warmup_steps=100,
            weight_decay=0.01,
            max_grad_norm=0.5,
            remove_unused_columns=True,
            no_cuda=not use_cuda,
            fp16=use_cuda,
            dataloader_num_workers=2 if use_cuda else 1
        )

        # ===== 6. Data Collator =====
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # ===== 7. Trainer Setup =====
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # ===== 8. Train =====
        logger.info("🚀 Starting training...")
        trainer.train()

        # ===== 9. Save Model =====
        logger.info("💾 Saving model...")
        trainer.save_model(f"{OUTPUT_DIR}-{timestamp}/final_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-{timestamp}/final_model")

        logger.info(f"✅ Training complete! Model saved to {OUTPUT_DIR}-{timestamp}")

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
