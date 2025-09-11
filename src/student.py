import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Use the SAME evaluator & pusher style as teacher
from evaluate import evaluate_model_response
from utils import HFPusher

# ================================================================
# Student model script (train / test) for math reasoning
# - Model: Qwen/Qwen2.5-Math-1.5B (chat template aware)
# - Modes:
#     --mode train: fine-tunes for 1 epoch on the given HF dataset and (optionally) pushes to HF Hub
#     --mode test : runs inference + accuracy evaluation on the given HF dataset
# - Datasets:
#   Train split (mode=train): idx, problem, teacher_solution (target), gt (ignored for loss)
#   Test  split (mode=test) : idx, problem, gt (used only for eval)
# - System prompt: ALWAYS place final answer as:  Final Answer: \\boxed{<answer>}
# ================================================================

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
SYS_PROMPT = (
    "You are a precise math problem solver. Work step by step where helpful, "
    "and ALWAYS present the final result on the last line as: Final Answer: \\boxed{<answer>}"
)

GT_KEYS = ["answer", "gt", "output", "solution"]

# --------------------------- Prompt Utils ---------------------------

def build_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": problem.strip()},
    ]


# --------------------------- Dataset Wrangling ---------------------------

def get_gt(example: Dict[str, Any]) -> Optional[str]:
    for k in GT_KEYS:
        if k in example and example[k] and str(example[k]).strip():
            return str(example[k])
    return None


@dataclass
class EncodedExample:
    input_ids: List[int]
    labels: List[int]


class ChatSFTBuilder:
    """Supervised fine-tuning where only the assistant completion is learned."""

    def __init__(self, tokenizer, max_len: int = 2048):
        self.tok = tokenizer
        self.max_len = max_len

    def build(self, problem: str, target_solution: str) -> EncodedExample:
        messages = build_messages(problem)
        # Prompt up to assistant start
        prompt_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        # Assistant tokens (= teacher/student completion)
        target_ids = self.tok(target_solution, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
        # Mask prompt tokens with -100
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        if len(input_ids) > len(prompt_ids):
            labels += input_ids[len(prompt_ids):]
        return EncodedExample(input_ids=input_ids, labels=labels)


# --------------------------- Inference + Eval ---------------------------

def run_inference_and_eval(model, tokenizer, dataset, max_new_tokens=256, save_jsonl: Optional[str] = None) -> Dict[str, Any]:
    total, correct = 0, 0
    records: List[Dict[str, Any]] = []
    for idx, ex in enumerate(dataset):
        problem = str(ex.get("problem", "")).strip()
        if not problem:
            continue
        gt = get_gt(ex)
        messages = build_messages(problem)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        eval_res = evaluate_model_response(pred, gt)
        is_ok = bool(eval_res.get("is_correct", False))
        total += 1
        correct += int(is_ok)

        rec = {
            "idx": int(ex.get("idx", idx)),
            "problem": problem,
            "student_solution": pred,
            "gt": gt,
            "evaluation": eval_res,
        }
        records.append(rec)

    metrics = {"n": total, "acc": (correct / total if total else 0.0), "correct": correct}
    if save_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(save_jsonl)), exist_ok=True)
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Evaluation:", json.dumps(metrics, indent=2))
    return metrics


# --------------------------- Training ---------------------------

def build_sft_dataset(train_ds, tokenizer):
    """SFT on teacher_solution only (ignore gt for loss)."""
    builder = ChatSFTBuilder(tokenizer)

    def _map_fn(ex):
        problem = str(ex.get("problem", "")).strip()
        target = str(ex.get("teacher_solution", "")).strip()
        if not target:
            target = "Final Answer: \\boxed{0}"  # fallback dummy
        enc = builder.build(problem, target)
        return {"input_ids": enc.input_ids, "labels": enc.labels}

    cols = train_ds.column_names
    tokenized = train_ds.map(_map_fn, remove_columns=cols)
    return tokenized


# --------------------------- HF Push Helpers ---------------------------

def start_pusher_if_needed(local_path: str, repo_id: Optional[str], path_in_repo: Optional[str], push_every_min: int, private: bool) -> Optional[HFPusher]:
    if not repo_id:
        return None
    interval = max(60, int(push_every_min) * 60)
    pusher = HFPusher(
        repo_id=repo_id,
        local_path=local_path,
        path_in_repo=path_in_repo or f"data/{os.path.basename(local_path)}",
        interval_sec=interval,
        private=bool(private),
    )
    pusher.start()
    return pusher


# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. MinTR_KIEU/datasetname")
    parser.add_argument("--output", default=None, help="Path to save predictions/logs (train: student jsonl; test: optional)")

    # HF dataset push (like teacher)
    parser.add_argument("--hf_repo", default=None, help="HF dataset repo id to push JSONL (e.g., your-username/my-new-dataset)")
    parser.add_argument("--hf_path_in_repo", default=None, help="Path inside repo (default: data/<basename>)")
    parser.add_argument("--push_every_min", type=int, default=30, help="Autosave interval in minutes (>=1)")
    parser.add_argument("--hf_private", type=int, default=0, help="1 private / 0 public for dataset push")

    # HF model push (optional, end-of-run)
    parser.add_argument("--hf_save", type=int, default=0, help="If >0, push fine-tuned model at end")
    parser.add_argument("--hf_model_repo", default=None, help="Repo to push the model, e.g. your-username/qwen2.5-math-student")
    parser.add_argument("--hf_model_private", type=int, default=0, help="1 private / 0 public for model repo")

    # Train knobs
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_dir", default="./student_ckpt")

    args = parser.parse_args()

    # Load dataset (no fixed split, always manual)
    split = "train" if args.mode == "train" else "test"
    ds = load_dataset(args.dataset, split=split)
    print(f"Loaded dataset {args.dataset} with {len(ds)} samples")

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if args.mode == "test":
        # Inference + evaluation (optionally save jsonl)
        metrics = run_inference_and_eval(model, tokenizer, ds, save_jsonl=args.output)
        print(json.dumps({"metrics": metrics}, indent=2))
        return

    # -------------------- TRAIN --------------------
    tokenized = build_sft_dataset(ds, tokenizer)
    os.makedirs(args.save_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.train_epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=0,
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized,
    )

    trainer.train()

    # Save locally
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # After training, run inference over the training set to create a new jsonl with student_solution
    out_path = args.output or os.path.join(args.save_dir, "student_train_preds.jsonl")
    print(f"Generating student solutions -> {out_path}")

    # Prepare HF pusher for dataset jsonl (same style as teacher)
    pusher = start_pusher_if_needed(
        local_path=out_path,
        repo_id=args.hf_repo,
        path_in_repo=args.hf_path_in_repo,
        push_every_min=args.push_every_min,
        private=bool(args.hf_private),
    )

    try:
        for i, ex in enumerate(ds):
            problem = str(ex.get("problem", "")).strip()
            if not problem:
                continue
            gt = get_gt(ex)
            teacher_solution = str(ex.get("teacher_solution", "")).strip()

            messages = build_messages(problem)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            student_solution = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            rec = {
                "idx": int(ex.get("idx", i)),
                "problem": problem,
                "teacher_solution": teacher_solution,
                "student_solution": student_solution,
                "gt": gt,
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        if pusher is not None:
            print("[Student] Finalizing dataset push…")
            pusher.stop_and_final_push()

    # Optional: push model weights at the very end
    if args.hf_save > 0:
        if not args.hf_model_repo:
            raise ValueError("--hf_model_repo is required when --hf_save > 0")
        print(f"Pushing model to HF: {args.hf_model_repo} (private={bool(args.hf_model_private)})")
        model.push_to_hub(args.hf_model_repo, private=bool(args.hf_model_private))
        tokenizer.push_to_hub(args.hf_model_repo, private=bool(args.hf_model_private))


if __name__ == "__main__":
    main()
