# undo-iterative-distill

## Installation

## Dataset preparation
```bash
huggingface-cli login

from datasets import load_dataset
ds_10k = load_dataset("MinTR-KIEU/NuminaMath-CoT-10k")
ds_100k = load_dataset("MinTR-KIEU/NuminaMath-CoT-100k")

```



### Next step
- Make val samples
- add teacher evaluation method
- edit teacher prompt