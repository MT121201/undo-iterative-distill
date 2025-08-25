from datasets import load_from_disk

# Load the saved subsets from local folders
sample_10k = load_from_disk("NuminaMath-CoT-10k")
sample_100k = load_from_disk("NuminaMath-CoT-100k")

sample_10k.push_to_hub("MinTR-KIEU/NuminaMath-CoT-10k")
sample_100k.push_to_hub("MinTR-KIEU/NuminaMath-CoT-100k")
