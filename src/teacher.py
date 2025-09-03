import argparse
import os
import json
import torch
import signal
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from string import Template
from time import sleep

# from prompt.teacher_prompt import TEACHER_PROMPT_ITER0, TEACHER_PROMPT_ITER1
from evaluate import evaluate_teacher_response
from prompt.teacher_prompt import build_teacher_prompt_iter0
from utils import HFPusher


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_mem = {i: "15GiB" for i in range(6)}
# --------------------------- Safe writer ---------------------------

def _append_jsonl_safely(path, records):
    """
    Append JSON lines safely:
    - open in append mode
    - write each line + newline
    - flush + fsync so data hits disk
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

# ------------------------------- Main ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="10k", help="Dataset size 10k/100k'")
    parser.add_argument("--iter", default="0", help="Iteration number")
    parser.add_argument("--output", default="teacher_iter0.jsonl", help="Path to save new dataset")
    parser.add_argument("--cont", type=int, default=None, help="Continue from this sample index")

    # HF pushing options
    parser.add_argument("--hf_repo", required=True,
                        help="HF dataset repo id, e.g., 'your-username/my-new-dataset'")
    parser.add_argument("--hf_path_in_repo", default=None,
                        help="Path inside repo (default: data/<output_basename>)")
    parser.add_argument("--push_every_min", type=int, default=30,
                        help="Autosave interval in minutes (>=1)")

    parser.add_argument("--hf_private", type=int, default=0, help="Create private repo (1) or public (0) if new")

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "10k":
        print("Loading 10k dataset")
        dataset_name = "MinTR-KIEU/NuminaMath-CoT-10k"
    elif args.dataset == "100k":
        print("Loading 100k dataset")
        dataset_name = "MinTR-KIEU/NuminaMath-CoT-100k"
    else:
        raise ValueError("Unknown dataset: choose '10k' or '100k'")

    dataset = load_dataset(dataset_name, split="train")

    # Continues sampling from index
    if args.cont is not None:
        dataset = dataset.select([i for i in range(args.cont, len(dataset))])

    # # Load prompts
    # if args.iter == "0":
    #     print("Using Iteration 0 Prompt")
    #     TEACHER_PROMPT = TEACHER_PROMPT_ITER0
    # else:
    #     print("Using Iteration 1+ Prompt")
    #     TEACHER_PROMPT = TEACHER_PROMPT_ITER1

    # load the tokenizer and the model
    # print("Using ", MODEL_NAME, "model")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_mem,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
        
    # Prepare output file (rotate if exists)
    out_path = args.output
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{base}_{timestamp}{ext}"
        print(f"Output file exists. Saving to new file: {out_path}")

    # Prepare HF pusher
    path_in_repo = args.hf_path_in_repo or f"data/{os.path.basename(out_path)}"
    pusher = HFPusher(
        repo_id=args.hf_repo,
        local_path=out_path,
        path_in_repo=path_in_repo,
        interval_sec=max(60, args.push_every_min * 60),
        private=bool(args.hf_private)
    )
    pusher.start()

    # Graceful stop on SIGINT/SIGTERM
    def _handle_signal(signum, frame):
        print(f"\n[Signal {signum}] Caught. Finalizing and pushing to HF...")
        pusher.stop_and_final_push()
        # Exit immediately after final push
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
            # Build prompt
            # import pdb; pdb.set_trace()
            problem_text = example.get("problem", "").strip()
            prompt = build_teacher_prompt_iter0(problem_text, tokenizer)
            print(prompt)

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # deterministic
                )
            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            print("=========RESPONSE=============")
            print(response)
            # Evaluate response
            gt = example.get("answer") or example.get("output") or example.get("gt") or example.get("solution")
            results = evaluate_teacher_response(response, gt)
            print("=========EVALUATION=============")
            print(results)
            if results.get("has_boxed") and results.get("is_correct"):
                _append_jsonl_safely(out_path, [{
                    "idx": idx,
                    "problem": example.get("problem"),
                    "teacher_solution": response,
                    "gt": gt,
                    
                }])


    finally:
        # Ensure a last push happens even on normal completion
        print("[Main] Finalizing...")
        pusher.stop_and_final_push()

if __name__ == "__main__":
    main()
