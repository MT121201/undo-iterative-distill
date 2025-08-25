from datasets import load_dataset

# Load dataset (example: NuminaMath-CoT)
dataset = load_dataset("MinTR-KIEU/NuminaMath-CoT-10k", split="train")

# Print dataset size
print("Dataset size:", len(dataset))

# Print the first sample
example = dataset[0]
print("One sample:\n", example)

print("===================================================")

# If you want to inspect keys
print("Keys in sample:", example.keys())

print("===================================================")
# Pretty print all key-value pairs in the sample
for key, value in example.items():
    print(f"{key}: {value}\n")
