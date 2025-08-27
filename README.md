# undo-iterative-distill

## Installation

## Dataset preparation
```bash
huggingface-cli login

from datasets import load_dataset
ds_10k = load_dataset("MinTR-KIEU/NuminaMath-CoT-10k")
ds_100k = load_dataset("MinTR-KIEU/NuminaMath-CoT-100k")

```



### Dev note
- Evaluation method pass all test cases for single box, including edge cases. But for multiple boxes it will only check last boxes