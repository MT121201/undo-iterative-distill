# pip install -q datasets huggingface_hub
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import string

# ---------- CONFIG ----------
TARGET_REPO = "MinTR-KIEU/Test_MMLU_Pro_math_boxed"
# login(token="hf_...")  # OR export HF_TOKEN before running
# ----------------------------

def build_question(example):
    """
    Concatenate question + options into one string with dynamic letters.
    Works even if >5 options.
    """
    q = example["question"].strip()
    opts = example["options"]
    letters = list(string.ascii_uppercase)  # ["A", "B", ..., "Z"]
    options_str = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
    return f"{q}\n{options_str}"


def build_answer(example):
    """
    Convert correct answer letter to boxed form.
    Example: answer="C" -> "$\\boxed{C}$"
    """
    letter = example["answer"].strip()
    return f"$\\boxed{{{letter}}}$"

def transform(example):
    return {
        "question": build_question(example),
        "answer": build_answer(example),
    }

def main():
    print("Loading MMLU-Pro test split...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    print("Filtering for category == 'math'...")
    ds = ds.filter(lambda ex: ex["category"].lower() == "math")

    print("Transforming format...")
    ds_new = ds.map(transform, remove_columns=ds.column_names)

    # Sanity check
    print("\n--- Example ---")
    print(ds_new[0]["question"])
    print(ds_new[0]["answer"])

    # Wrap into DatasetDict with only test
    dset = DatasetDict({"test": ds_new})

    print(f"\nPushing to {TARGET_REPO} ...")
    dset.push_to_hub(TARGET_REPO)
    print("Done!")

if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    main()
