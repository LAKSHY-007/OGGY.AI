from datasets import load_dataset, DatasetDict
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_conversations(example):
    """
    Keep only high-quality conversations with rank >= 0.5.
    Safely handles None and missing values.
    """
    rank = example.get("rank")
    return isinstance(rank, (int, float)) and rank >= 0.5

def main():
    try:
        logger.info("📥 Loading OASST1 dataset...")
        dataset_dict = load_dataset("OpenAssistant/oasst1")

        logger.info("🔍 Filtering high-quality conversations...")
        filtered_dataset_dict = DatasetDict()

        for split in dataset_dict:
            logger.info(f"⏳ Filtering split: {split}")
            filtered_split = dataset_dict[split].filter(filter_conversations)
            logger.info(f"✅ {split} - {len(filtered_split)} examples kept out of {len(dataset_dict[split])}")
            filtered_dataset_dict[split] = filtered_split

        output_dir = "./processed_oasst1"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"💾 Saving filtered dataset to: {output_dir}")
        filtered_dataset_dict.save_to_disk(output_dir)
        logger.info("✅ Dataset saved successfully!")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
