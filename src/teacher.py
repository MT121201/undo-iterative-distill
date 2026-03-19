"""
Teacher model inference — all iterations.

Iteration 0:   teacher solves each training question from scratch (UNDO baseline pass).
Iteration K≥1: teacher regenerates solutions using the full UNDO context, with our
               Progressive Hint Cascade (Contribution 1) applied to failed attempts.

Output JSONL schema per record:
    {
      "idx":              int,
      "problem":          str,
      "gt":               str,
      "gt_solution":      str,     — full chain-of-thought from dataset (for hint stage 3)
      "teacher_response": str,
      "iteration":        int,
      "hint_stage":       str,     — "none" | "answer" | "solution"  (null at iter 0)
      "hint_strategy":    str,     — "progressive" | "always_solution" | "none"
    }

Run (iter 0):
    python src/teacher.py --iter 0 --dataset 10k \\
        --output runs/teacher_iter0.jsonl \\
        --hf_repo MinTR-KIEU/intelligent-distill-outputs

Run (iter K≥1):
    python src/teacher.py --iter 1 --dataset 10k \\
        --prev_teacher_jsonl runs/teacher_iter0.jsonl \\
        --prev_student_jsonl runs/student_iter0.jsonl \\
        --val_signals_jsonl  runs/val_signals_iter0.jsonl \\
        --hint_strategy progressive \\
        --output runs/teacher_iter1.jsonl \\
        --hf_repo MinTR-KIEU/intelligent-distill-outputs
"""

import argparse
import json
import os
import signal
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import evaluate_model_response
from hint import HintResult, progressive_hint_cascade
from prompt.teacher_prompt import (
    build_teacher_prompt_iter0,
    build_teacher_prompt_answer_hint,
    build_teacher_prompt_iterK,
    build_teacher_prompt_solution_hint,
)
from utils import HFPusher

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_MEM  = {i: "15GiB" for i in range(6)}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _append_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_jsonl_by_idx(path: Optional[str]) -> Dict[int, Dict]:
    """Load a JSONL file into a dict keyed by the 'idx' field."""
    if not path or not os.path.exists(path):
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[int(rec["idx"])] = rec
    return records


def _load_val_signals(path: Optional[str]) -> List[Dict]:
    """Load validation error signals as a flat list."""
    if not path or not os.path.exists(path):
        return []
    signals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                signals.append(json.loads(line))
    return signals


# ── Model inference wrapper ────────────────────────────────────────────────────

def _make_infer_fn(model, tokenizer, max_new_tokens: int = 512) -> Callable[[str], str]:
    """
    Return a simple callable: prompt_str → response_str.
    Encapsulating inference here keeps hint.py model-agnostic and testable.
    """
    def infer(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
    return infer


# ── Iteration 0 ───────────────────────────────────────────────────────────────

def run_iter0(dataset, model, tokenizer, out_path: str) -> None:
    """
    UNDO baseline first pass: teacher solves each question independently.
    Only correct responses are saved to the distillation dataset.
    """
    infer = _make_infer_fn(model, tokenizer)

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Iter 0"):
        solution_text = example.get("solution", "").strip()
        if not evaluate_model_response(solution_text, r"\boxed{4}").get("has_boxed"):
            continue   # skip samples whose GT has no \\boxed{} — cannot evaluate

        problem     = example.get("problem", "").strip()
        gt          = example.get("answer") or example.get("solution") or ""
        gt_solution = example.get("solution") or gt

        prompt   = build_teacher_prompt_iter0(problem, tokenizer)
        response = infer(prompt)

        if idx == 0:
            print("=== PROMPT TEMPLATE (iter 0) ===\n", prompt)

        result = evaluate_model_response(response, gt)
        if result.get("is_correct"):
            _append_jsonl(out_path, [{
                "idx":              idx,
                "problem":          problem,
                "gt":               gt,
                "gt_solution":      gt_solution,
                "teacher_response": response,
                "iteration":        0,
                "hint_stage":       None,
                "hint_strategy":    None,
            }])


# ── Iteration K≥1 ─────────────────────────────────────────────────────────────

def run_iterK(
    dataset,
    model,
    tokenizer,
    out_path: str,
    iteration: int,
    hint_strategy: str,
    prev_teacher_records: Dict[int, Dict],
    prev_student_records: Dict[int, Dict],
    val_error_signals: List[Dict],
) -> None:
    """
    Iteration K≥1: regenerate teacher responses with UNDO context + hint cascade.

    hint_strategy options:
        "progressive"    — Contribution 1: try no-hint → answer-hint → solution-hint
        "always_solution" — ablation: always provide full GT solution
        "none"           — UNDO baseline: no hinting (drop on failure)
    """
    infer = _make_infer_fn(model, tokenizer)

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=f"Iter {iteration}"):
        solution_text = example.get("solution", "").strip()
        if not evaluate_model_response(solution_text, r"\boxed{4}").get("has_boxed"):
            continue

        idx_key     = int(example.get("idx", idx))
        problem     = example.get("problem", "").strip()
        gt          = example.get("answer") or example.get("solution") or ""
        gt_solution = example.get("solution") or gt

        prev_teacher = prev_teacher_records.get(idx_key, {}).get("teacher_response", "")
        prev_student = prev_student_records.get(idx_key, {}).get("student_solution", "")

        result: Optional[HintResult] = None

        if hint_strategy == "progressive":
            result = progressive_hint_cascade(
                problem=problem,
                gt_answer=gt,
                gt_solution=gt_solution,
                val_error_signals=val_error_signals,
                student_response_prev=prev_student or None,
                teacher_response_prev=prev_teacher or None,
                teacher_infer_fn=infer,
                tokenizer=tokenizer,
            )

        elif hint_strategy == "always_solution":
            prompt   = build_teacher_prompt_solution_hint(
                problem, gt_solution, val_error_signals,
                prev_student or None, prev_teacher or None, tokenizer,
            )
            response = infer(prompt)
            eval_res = evaluate_model_response(response, gt)
            if eval_res.get("is_correct"):
                result = HintResult(
                    teacher_response=response,
                    hint_stage="solution",
                    extracted_answer=eval_res.get("extracted_answer"),
                    comparison_mode=eval_res.get("comparison_mode"),
                )

        else:  # "none" — UNDO baseline: no hinting
            prompt   = build_teacher_prompt_iterK(
                problem, val_error_signals,
                prev_student or None, prev_teacher or None, tokenizer,
            )
            response = infer(prompt)
            eval_res = evaluate_model_response(response, gt)
            if eval_res.get("is_correct"):
                result = HintResult(
                    teacher_response=response,
                    hint_stage="none",
                    extracted_answer=eval_res.get("extracted_answer"),
                    comparison_mode=eval_res.get("comparison_mode"),
                )

        if result is not None:
            _append_jsonl(out_path, [{
                "idx":              idx_key,
                "problem":          problem,
                "gt":               gt,
                "gt_solution":      gt_solution,
                "teacher_response": result.teacher_response,
                "iteration":        iteration,
                "hint_stage":       result.hint_stage,
                "hint_strategy":    hint_strategy,
            }])


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Teacher model inference for Intelligent Distillation pipeline."
    )
    parser.add_argument("--dataset",     default="10k",  choices=["10k", "100k"])
    parser.add_argument("--iter",        type=int, default=0, help="Pipeline iteration (0-indexed)")
    parser.add_argument("--output",      default="runs/teacher_iter0.jsonl")
    parser.add_argument("--cont",        type=int, default=None, help="Continue from sample index")

    # Iter K≥1 context files
    parser.add_argument("--prev_teacher_jsonl", default=None)
    parser.add_argument("--prev_student_jsonl", default=None)
    parser.add_argument("--val_signals_jsonl",  default=None)
    parser.add_argument("--hint_strategy",
                        choices=["progressive", "always_solution", "none"],
                        default="progressive",
                        help="Hint strategy for recovering failed attempts (Contribution 1)")

    # HF push
    parser.add_argument("--hf_repo",         required=True)
    parser.add_argument("--hf_path_in_repo", default=None)
    parser.add_argument("--push_every_min",  type=int, default=30)
    parser.add_argument("--hf_private",      type=int, default=0)

    args = parser.parse_args()

    # Load dataset
    dataset_name = {
        "10k":  "MinTR-KIEU/NuminaMath-CoT-10k",
        "100k": "MinTR-KIEU/NuminaMath-CoT-100k",
    }[args.dataset]
    dataset = load_dataset(dataset_name, split="train")
    if args.cont is not None:
        dataset = dataset.select(range(args.cont, len(dataset)))

    # Rotate output file to avoid clobbering
    out_path = args.output
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_path)
        out_path  = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"Output exists — writing to: {out_path}")

    # Periodic HF push
    pusher = HFPusher(
        repo_id=args.hf_repo,
        local_path=out_path,
        path_in_repo=args.hf_path_in_repo or f"data/{os.path.basename(out_path)}",
        interval_sec=max(60, args.push_every_min * 60),
        private=bool(args.hf_private),
    )
    pusher.start()

    def _on_signal(signum, frame):
        print(f"\n[Signal {signum}] Finalizing push...")
        pusher.stop_and_final_push()
        os._exit(0)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEM,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    try:
        if args.iter == 0:
            run_iter0(dataset, model, tokenizer, out_path)
        else:
            run_iterK(
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                out_path=out_path,
                iteration=args.iter,
                hint_strategy=args.hint_strategy,
                prev_teacher_records=_load_jsonl_by_idx(args.prev_teacher_jsonl),
                prev_student_records=_load_jsonl_by_idx(args.prev_student_jsonl),
                val_error_signals=_load_val_signals(args.val_signals_jsonl),
            )
    finally:
        pusher.stop_and_final_push()


if __name__ == "__main__":
    main()
