# pip install -q datasets huggingface_hub
import os, re
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# ---------- CONFIG ----------
TARGET_REPO = "MinTR-KIEU/Test_gsm8k_boxed"
# login(token="hf_...")  # or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN
# ----------------------------

def extract_final_number(answer: str) -> str:
    """
    Extract the final numeric token after '####'.
    Falls back to the last number anywhere if needed.
    """
    m = re.search(r"####\s*([^\n\r]*)", answer)
    if m:
        tail = m.group(1)
        n = re.search(r"[-+]?\d+(?:\.\d+)?", tail)
        if n:
            return n.group(0)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else "?"

def replace_hashes_with_boxed(answer: str) -> str:
    """
    Keep all previous lines intact.
    Replace the line that starts with '####' with '$\\boxed{n}$'.
    If no '####' line exists, append '$\\boxed{n}$' as a new final line.
    """
    num = extract_final_number(answer)
    boxed = f"$\\boxed{{{num}}}$"

    lines = answer.rstrip("\n").splitlines()
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("####"):
            idx = i
            break

    if idx is not None:
        lines[idx] = boxed
    else:
        if lines and lines[-1].strip() != boxed:
            lines.append(boxed)
        elif not lines:
            lines = [boxed]

    return "\n".join(lines)

def transform(example):
    return {"answer": replace_hashes_with_boxed(example["answer"])}

def main():
    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    print("Transforming final lines to $\\boxed{n}$ while preserving CoT...")
    ds_new = ds.map(transform, desc="Boxing final answers")

    # Sanity check on first sample
    print("\n--- BEFORE ---")
    print(ds[0]["answer"].splitlines()[-3:])
    print("\n--- AFTER ---")
    print(ds_new[0]["answer"].splitlines()[-3:])

    dset = DatasetDict({"test": ds_new})
    print(f"\nPushing to {TARGET_REPO} ...")
    dset.push_to_hub(TARGET_REPO)
    print("Done!")

if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    main()
