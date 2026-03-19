# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **UNDO (Iterative Knowledge Distillation)** with extensions:
1. **Progressive Hint-Based Distillation** — recovers incorrect teacher attempts by progressively providing hints (final answer → full solution) instead of dropping them
2. **Reasoning-Aware Error Feedback** — replaces binary 0/1 scoring with `S = 0.6 × Answer correctness + 0.4 × Reasoning quality`

**Models:** Teacher = Qwen3-30B-A3B (~30B), Student = Qwen2.5-Math-1.5B-Instruct
**Dataset:** NuminaMath-CoT (10K samples for reproduction)
**Benchmarks:** GSM8K, SVAMP (easy), MATH, MMLU-Pro (hard), StrategyQA (out-of-domain)

## Working Constraints

**No model inference, training, or downloading.** No compute resources available. All work is code-only: write, structure, and make pipelines ready to run. Never attempt to load, infer, or fine-tune models. Never download model weights.

**Academic research code, not production engineering.** This is a conference paper submission. Prioritize:
- **Clarity over cleverness** — code should read like the paper's algorithm; a researcher unfamiliar with the codebase should be able to map each function to a section/equation in the paper
- **Reproducibility** — every experiment must be runnable from a single entry-point script with explicit arguments; no hidden state or undocumented flags
- **Honest naming** — variables and functions should use the paper's terminology (`progressive_hint`, `reasoning_score`, `val_error_signal`, etc.), not generic engineering names
- **No over-engineering** — no abstract base classes, factory patterns, or plugin systems; flat and direct is better
- **Experiment traceability** — all outputs (JSONL runs) must include enough metadata (iteration, model, dataset, config) to reconstruct what produced them

## Environment Setup

Two separate conda environments are required:

```bash
# Teacher environment
conda create -n teacher python=3.10 -y && conda activate teacher
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # GPU memory optimization

# Student environment
conda create -n student python=3.10 -y && conda activate student
pip install -r requirements.txt
```

Set HF token for pushing to Hugging Face:
```bash
export HF_TOKEN="your_huggingface_token"
```

## Commands

**Run tests:**
```bash
pytest                            # all tests
pytest test/test_evaluate.py      # single test file
pytest test/test_evaluate.py::test_evaluate_model_response_numeric  # single test
```

**Run student training:**
```bash
python src/student.py --mode train --dataset <HF_DATASET_ID> --save_dir ./student_ckpt
```

**Run student inference/evaluation:**
```bash
python src/student.py --mode test --dataset <HF_DATASET_ID> --output runs/student_out.jsonl
```

## Architecture

```
src/
  evaluate.py   — Answer extraction + correctness scoring for math reasoning
                  Extracts \boxed{} content and compares via: numeric/fraction,
                  ±symmetry, letter choice, string fallback
  student.py    — Student model (Qwen2.5-Math-1.5B-Instruct) train/test loop.
                  SFT on teacher_solution field; evaluates with evaluate.py
  utils.py      — HFPusher: background thread for periodic dataset pushes to HF Hub
  prompt/       — (reserved for prompt templates)

dataset/        — Scripts for sampling/building train and val sets, benchmark test sets
  sampling_trainset.py, sampling_val.py
  gsm8k_test.py, swamp_test.py, MMLU_PRO_math_test.py

utilities/      — push_to_hub.py: upload datasets/models to HF Hub

easydistill/    — Git submodule: upstream EasyDistill framework (teacher-side distillation)

runs/           — Output JSONL files from inference (e.g., student_svamp_i1.jsonl)
vis/            — Visualization images (UNDO pipeline diagrams)
```

## Key Data Flow (UNDO Iterative Loop)

Per iteration K:
1. **Validation**: Student infers on val set → generates error signals (scores per question)
2. **Teacher context**: Feed question + val error signals + prior student/teacher responses → teacher generates new CoT
3. **Evaluate**: If teacher answer is correct → add to distillation dataset; if wrong → apply progressive hinting (provide final answer, then full solution) before dropping
4. **Student distillation**: Fine-tune student on filtered/recovered dataset
5. **Repeat** for 3 iterations (OOD performance drops after iter 3 signal overfitting)

## Dataset Schema

**Train split fields:** `idx`, `problem`, `teacher_solution` (SFT target), `gt` (ground truth, eval only)
**Test split fields:** `idx`, `problem`/`question`, `gt`/`answer`/`output`/`solution`

## Answer Evaluation (`src/evaluate.py`)

`evaluate_model_response(response, expected)` returns dict with `has_boxed`, `extracted_answer`, `is_correct`, `comparison_mode`.

Comparison priority: `pm` (±) → `numeric`/fraction → `choice` (letter) → `string` fallback.
Handles: LaTeX fractions, degree/currency/percent normalization, circled numerals (①–⑩), factorial.

## Presentation Reference

`CLAUDE_CODE_DIRECTION.md` contains the full 33-slide specification for the academic presentation "Recovering Incorrect Teacher Attempts via Answer- and Solution-Level Hints" (MT Kieu, March 2026). It includes exact slide content, chart data, color palette (`#7B1D1D` maroon theme), and PptxGenJS + matplotlib generation instructions.
