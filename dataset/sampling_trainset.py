from datasets import load_dataset

# Load full dataset
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")

print("Full dataset size:", len(dataset))

# Sample 10k
sample_10k = dataset.shuffle(seed=42).select(range(10_000))
print("10k subset size:", len(sample_10k))

# Sample 100k
sample_100k = dataset.shuffle(seed=42).select(range(100_000))
print("100k subset size:", len(sample_100k))

# Optionally save to disk
sample_10k.save_to_disk("NuminaMath-CoT-10k")
sample_100k.save_to_disk("NuminaMath-CoT-100k")

# Or push to Hugging Face Hub (requires login: huggingface-cli login)
# sample_10k.push_to_hub("your-username/NuminaMath-CoT-10k")
# sample_100k.push_to_hub("your-username/NuminaMath-CoT-100k")
