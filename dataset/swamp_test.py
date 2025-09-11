# pip install -q datasets huggingface_hub
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# ---------- CONFIG ----------
TARGET_REPO = "MinTR-KIEU/Test_SVAMP_boxed"
# login(token="hf_...")  # OR export HF_TOKEN before running
# ----------------------------

def build_answer(example):
    """
    Convert numeric/final answer into boxed LaTeX form.
    Example: 18 -> "$\\boxed{18}$"
    """
    ans = str(example["Answer"]).strip()
    return f"$\\boxed{{{ans}}}$"

def transform(example):
    return {
        "question": example["question_concat"].strip(),
        "answer": build_answer(example),
    }

def main():
    print("Loading SVAMP test split...")
    ds = load_dataset("ChilleD/SVAMP", split="test")

    print("Transforming format...")
    ds_new = ds.map(transform, remove_columns=ds.column_names)

    # Sanity check
    print("\n--- Example ---")
    print(ds_new[0]["question"])
    print(ds_new[0]["answer"])

    # Only test split
    dset = DatasetDict({"test": ds_new})

    print(f"\nPushing to {TARGET_REPO} ...")
    dset.push_to_hub(TARGET_REPO)
    print("Done!")

if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    main()
