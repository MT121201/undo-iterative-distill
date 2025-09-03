# undo-iterative-distill

## Installation

See [doc/installation.md](doc/installation.md) for installation instructions.

## Dataset Preparation

To download the datasets, run:

```python
from datasets import load_dataset

ds_10k = load_dataset("MinTR-KIEU/NuminaMath-CoT-10k")
ds_100k = load_dataset("MinTR-KIEU/NuminaMath-CoT-100k")
```

## Running

Before running the script, set the following environment variable:

```bash
export HF_TOKEN="your_huggingface_token"
```

To perform teacher inference, use:

```bash
python src/teacher.py \
    --dataset 10k \
    --iter 0 \
    --output QWEN3_30B_10k_Iter0.jsonl \
    --hf_repo MinTR-KIEU/Teacher_CoT_NuminaMath_10k_I0 \
    --hf_private 0 \
    --push_every_min 30
```

## Dev Notes

- The evaluation method passes all test cases for a single box, including edge cases.
- For multiple boxes, it only checks the last box.