from datasets import load_dataset

# Load full dataset
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="test")

print("Full dataset size:", len(dataset))

# Sample 10k
val_sample = dataset.shuffle(seed=42).select(range(20))
print("subset size:", len(val_sample))

# Or push to Hugging Face Hub (requires login: huggingface-cli login)
val_sample.push_to_hub("MinTR-KIEU/NuminaMath-val")
# sample_100k.push_to_hub("your-username/NuminaMath-CoT-100k")
