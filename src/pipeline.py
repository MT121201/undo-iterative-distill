"""
Intelligent Distillation — Iterative Pipeline
==============================================
Orchestrates the full UNDO iterative knowledge distillation loop with our two
contributions active. This script is the single entry-point that sequences all
pipeline stages and records the exact commands to reproduce each run.

Algorithm per iteration K (1-indexed):
    1. Student validation   → val_error_signals  (one score per val question)
    2. Teacher generation   → distillation dataset  (with progressive hint cascade)
    3. Student distillation → SFT + student predictions on train set
    4. Save outputs         → JSONL + config.json  (fully reproducible)

Contributions applied here:
    Contribution 1 — Progressive Hint Cascade (--hint_strategy progressive):
        Failed teacher attempts are retried with graduated hints before being dropped.
        See src/hint.py for the cascade logic.

    Contribution 2 — Reasoning-Aware Error Signal (--scoring_signal reasoning_aware):
        Validation scores S = 0.6·A + 0.4·R replace binary {0,1}.
        See src/scoring.py for the signal formula and rubric.

Directory layout produced under --output_dir:
    config.json
    iter_0/
        teacher_output.jsonl   ← produced by teacher.py --iter 0
        student_output.jsonl   ← produced by student.py --mode train
        val_signals.jsonl      ← produced by student.py --mode test on val set
    iter_1/
        teacher_output.jsonl
        student_output.jsonl
        val_signals.jsonl
    ...

Run:
    python src/pipeline.py \\
        --dataset      MinTR-KIEU/NuminaMath-CoT-10k \\
        --val_dataset  MinTR-KIEU/NuminaMath-val \\
        --iterations   3 \\
        --hint_strategy   progressive \\
        --scoring_signal  reasoning_aware \\
        --output_dir   runs/exp_01 \\
        --hf_repo      MinTR-KIEU/intelligent-distill-outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

from scoring import compute_error_signal, format_signal_trajectory


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_jsonl_by_idx(path: Optional[str]) -> Dict[int, Dict]:
    if not path or not os.path.exists(path):
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[int(rec["idx"])] = rec
    return records


def _append_jsonl(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── Validation step ────────────────────────────────────────────────────────────

def build_val_error_signals(
    val_dataset,
    student_records: Dict[int, Dict],
    scoring_signal: str,
    judge_fn=None,
) -> List[Dict]:
    """
    Convert student inference records over the val set into error signals.

    Each signal: {question, student_response, score, answer_correctness,
                  reasoning_quality, signal_type, is_correct}

    The score uses our Reasoning-Aware formula (Contribution 2) when
    scoring_signal == "reasoning_aware"; otherwise binary {−1, +1}.

    In practice this function is called after running:
        python src/student.py --mode test --dataset <val_dataset> --output val_raw.jsonl
    and passing the resulting JSONL records as student_records.
    """
    signals = []
    for ex in val_dataset:
        idx     = int(ex.get("idx", 0))
        rec     = student_records.get(idx, {})
        s_resp  = rec.get("student_solution", "")
        gt      = ex.get("answer") or ex.get("gt") or ex.get("solution") or ""

        signal = compute_error_signal(
            student_response=s_resp,
            gt=gt,
            judge_fn=judge_fn if scoring_signal == "reasoning_aware" else None,
        )
        signals.append({
            "idx":                idx,
            "question":           ex.get("problem") or ex.get("question", ""),
            "student_response":   s_resp,
            **signal,
            "is_correct":         signal["answer_correctness"] > 0,
        })
    return signals


# ── Pipeline state summary ────────────────────────────────────────────────────

def _print_iteration_summary(
    iteration: int,
    out_dir: Path,
    iter_dir: Path,
    hint_strategy: str,
    scoring_signal: str,
    prev_teacher_n: int,
    prev_student_n: int,
    val_signal_n: int,
    args,
) -> None:
    """Print the exact shell commands needed to execute this iteration."""
    prev_dir = out_dir / f"iter_{iteration - 1}"

    print(f"\n{'=' * 60}")
    print(f"  Iteration {iteration}  |  hint={hint_strategy}  |  signal={scoring_signal}")
    print(f"{'=' * 60}")
    print(f"  Context loaded:")
    print(f"    prev teacher records : {prev_teacher_n}")
    print(f"    prev student records : {prev_student_n}")
    print(f"    val error signals    : {val_signal_n}")
    print()
    print(f"  Step 1 — Teacher generation:")
    print(f"    python src/teacher.py \\")
    print(f"      --iter {iteration} \\")
    print(f"      --dataset 10k \\")
    print(f"      --hint_strategy {hint_strategy} \\")
    print(f"      --prev_teacher_jsonl {prev_dir}/teacher_output.jsonl \\")
    print(f"      --prev_student_jsonl {prev_dir}/student_output.jsonl \\")
    print(f"      --val_signals_jsonl  {prev_dir}/val_signals.jsonl \\")
    print(f"      --output {iter_dir}/teacher_output.jsonl \\")
    print(f"      --hf_repo {args.hf_repo}")
    print()
    print(f"  Step 2 — Upload teacher output to HF, then student distillation:")
    print(f"    python src/student.py --mode train \\")
    print(f"      --dataset <hf_dataset_with_teacher_output_iter{iteration}> \\")
    print(f"      --save_dir {iter_dir}/student_ckpt \\")
    print(f"      --output   {iter_dir}/student_output.jsonl")
    print()
    print(f"  Step 3 — Student validation inference:")
    print(f"    python src/student.py --mode test \\")
    print(f"      --model_path {iter_dir}/student_ckpt \\")
    print(f"      --dataset {args.val_dataset} \\")
    print(f"      --output {iter_dir}/val_raw.jsonl")
    print()
    print(f"  Step 4 — Build val error signals (run pipeline.py again after step 3):")
    print(f"    [pipeline.py reads {iter_dir}/val_raw.jsonl → writes {iter_dir}/val_signals.jsonl]")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Distillation — full iterative pipeline orchestrator."
    )
    parser.add_argument("--dataset",      required=True, help="HF training dataset id")
    parser.add_argument("--val_dataset",  required=True, help="HF validation dataset id")
    parser.add_argument("--iterations",   type=int, default=3)
    parser.add_argument("--hint_strategy",
                        choices=["progressive", "always_solution", "none"],
                        default="progressive",
                        help="Hint strategy (Contribution 1)")
    parser.add_argument("--scoring_signal",
                        choices=["reasoning_aware", "binary"],
                        default="reasoning_aware",
                        help="Validation error signal type (Contribution 2)")
    parser.add_argument("--output_dir",   default="runs/exp_01")
    parser.add_argument("--hf_repo",      default=None)
    parser.add_argument("--hf_private",   type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config for full reproducibility
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"[Pipeline] Config saved → {out_dir}/config.json")

    val_ds = load_dataset(args.val_dataset, split="train")

    for iteration in range(1, args.iterations + 1):
        iter_dir = out_dir / f"iter_{iteration}"
        iter_dir.mkdir(exist_ok=True)
        prev_dir = out_dir / f"iter_{iteration - 1}"

        prev_teacher_records = _load_jsonl_by_idx(str(prev_dir / "teacher_output.jsonl"))
        prev_student_records = _load_jsonl_by_idx(str(prev_dir / "student_output.jsonl"))

        # Build val error signals from previous student's val inference
        val_raw_path = str(prev_dir / "val_raw.jsonl")
        val_student_records = _load_jsonl_by_idx(val_raw_path)
        val_signals = build_val_error_signals(
            val_dataset=val_ds,
            student_records=val_student_records,
            scoring_signal=args.scoring_signal,
        )

        # Write val signals for this iteration's teacher context
        val_signals_path = str(prev_dir / "val_signals.jsonl")
        if val_signals and not os.path.exists(val_signals_path):
            for sig in val_signals:
                _append_jsonl(val_signals_path, sig)
            print(f"[Pipeline] Val signals written → {val_signals_path}")

        # Print signal trajectory summary across iterations
        if val_signals:
            avg = sum(s["score"] for s in val_signals) / len(val_signals)
            print(f"\n[Pipeline] Iter {iteration} | avg val signal score: {avg:+.3f}")

        _print_iteration_summary(
            iteration=iteration,
            out_dir=out_dir,
            iter_dir=iter_dir,
            hint_strategy=args.hint_strategy,
            scoring_signal=args.scoring_signal,
            prev_teacher_n=len(prev_teacher_records),
            prev_student_n=len(prev_student_records),
            val_signal_n=len(val_signals),
            args=args,
        )


if __name__ == "__main__":
    main()
