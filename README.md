# Intelligent Distillation
**Preserving Teacher Reasoning with Progressive Hint-Based Distillation**

---

## Overview

Iterative knowledge distillation methods like [UNDO](https://arxiv.org/abs/2502.12051) teach a student model by having a large teacher repeatedly diagnose student failures and regenerate solutions. But **~40% of teacher attempts are silently discarded** — those where the teacher itself gets the answer wrong.

We ask: *are these failures truly uninformative?*

They are not. Dropped attempts contain partial reasoning, useful intermediate steps, and signals about where the teacher is uncertain. We recover them with two contributions that make distillation both more data-efficient and more epistemically honest.

---

## Pipeline

```
╔══════════════════════════════════════════════════════════════════════════╗
║              Intelligent Distillation  ·  Iteration K                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐  ║
║   │  Val Set    │───►│ Student Infer    │───►│   Error Signals  ★    │  ║
║   └─────────────┘    └──────────────────┘    │  S = 0.6·A + 0.4·R   │  ║
║                                              └──────────┬────────────┘  ║
║                                                         │               ║
║              ┌──────────────────────────────────────────┘               ║
║              │  Teacher Context per question:                            ║
║              │    · all val error signals                                ║
║              │    · student response at K−1                             ║
║              │    · teacher response at K−1                             ║
║              ▼                                                           ║
║         ┌─────────┐                                                      ║
║         │ Teacher │                                                      ║
║         └────┬────┘                                                      ║
║              │                                                           ║
║       ┌──────┴──────┐                                                    ║
║    CORRECT        WRONG                                                  ║
║       │              │                                                   ║
║       ▼              ▼                                                   ║
║  ┌─────────┐   ╔═══════════════════════════════╗                        ║
║  │ Dataset │   ║  Progressive Hint Cascade  ★  ║                        ║
║  └────┬────┘   ║                               ║                        ║
║       │        ║  Stage 1 · no hint   → WRONG  ║                        ║
║       │        ║          ↓                    ║                        ║
║       │        ║  Stage 2 · answer    → WRONG  ║                        ║
║       │        ║          ↓                    ║                        ║
║       │        ║  Stage 3 · solution  → WRONG  ║                        ║
║       │        ║          ↓                    ║                        ║
║       │        ║        Drop                   ║                        ║
║       │        ╚═══════════╤═══════════════════╝                        ║
║       │                    │ CORRECT (recovered)                         ║
║       └────────────────────┘                                             ║
║                    │                                                     ║
║                    ▼                                                     ║
║           ┌─────────────────┐                                            ║
║           │  Student SFT   │──────────────────────► Iteration K+1       ║
║           └─────────────────┘                                            ║
║                                                          ★ = our contrib ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Contribution 1 — Progressive Hint Cascade

Instead of discarding an incorrect teacher attempt, we retry it with the minimum hint necessary to produce a correct response:

```
  ┌─────────────────────────────────────────────────────────────┐
  │               Progressive Hint Cascade                      │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  attempt(no hint)          ──► CORRECT → save              │
  │       │ WRONG                                              │
  │       ▼                                                     │
  │  attempt(answer hint)      ──► CORRECT → save  ┐           │
  │       │ WRONG               teacher still builds│           │
  │       ▼                     its own reasoning  │           │
  │  attempt(solution hint)    ──► CORRECT → save  ┘           │
  │       │ WRONG                                              │
  │       ▼                                                     │
  │      Drop                                                   │
  └─────────────────────────────────────────────────────────────┘
```

**Why graduated?** A pilot study reveals a key asymmetry in how teachers use hints:

```
  Teacher Reasoning Fidelity  (BERTScore F1: teacher explanation vs GT solution)

  Correct hint given ──  ░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓  peak 0.80–0.90  → paraphrases
  Wrong / no hint   ──  ▓▓▓▓▓▓░░░░░░░░░░░░░░  peak 0.20–0.30  → reasons

                        0.0   0.25  0.50  0.75  1.0
                        ├─────┼─────┼─────┼─────┤
                        self-reasoning       copying
```

When given only the *final answer* (not the full solution), the teacher is guided to the right destination but constructs its own reasoning path. This is the core insight: **the answer hint preserves teacher-internal reasoning while still enabling recovery.**

---

## Contribution 2 — Reasoning-Aware Error Signal

UNDO's validation feedback is binary: **1** (correct) or **0** (wrong). We replace it with a continuous signal that tracks *how* the student is improving across iterations:

```
  S  =  0.6 × A  +  0.4 × R

  A ∈ {+1, −1}                    answer correctness
  R ∈ {−1, −0.5, 0, +0.5, +1}    reasoning quality (LLM-judged rubric)
  S ∈ [−1, +1]                    combined error signal
```

**Rubric for R:**

```
  +1.0  ──  correct logic, minimal gaps
  +0.5  ──  minor slip, structure good
   0.0  ──  unclear / missing key steps
  −0.5  ──  wrong approach but coherent
  −1.0  ──  nonsense / cannot reason
```

Instead of a snapshot, the teacher now sees a **trajectory**:

```
  Iteration:    init     1       2       3
  Score S :   [ −0.9,  −0.4,  +0.2,  +0.9 ]
                 ↑ wrong reasoning → slowly improving → correct
```

The teacher can recognise and reinforce structural progress even before the final answer is correct.

---

## Why It Works — The L / M / H Analysis

We measure BERTScore F1 similarity between each teacher explanation and the GT solution, and classify recovered samples into three groups:

```
  ┌──────────────────────────────────────────────────────────────┐
  │       Teacher Reasoning Fidelity Spectrum                    │
  ├────────────┬─────────────┬───────────────────────────────────┤
  │     L      │      M      │              H                    │
  │  < 0.50    │  0.50–0.75  │           ≥ 0.75                  │
  ├────────────┼─────────────┼───────────────────────────────────┤
  │ independent│   guided,   │  paraphrases / copies solution    │
  │  reasoning │ not copying │                                   │
  ├────────────┼─────────────┼───────────────────────────────────┤
  │  ↑ OOD     │             │  ↑ in-domain  ↓ OOD (overfits)   │
  └────────────┴─────────────┴───────────────────────────────────┘
```

Progressive hinting **concentrates the recovered dataset in L**.
Always-Solution hinting **concentrates it in M/H**.

Training students on L vs H directly confirms the mechanism:

| Group | GSM8K | MATH      | StrategyQA (OOD) |
|-------|-------|-----------|-----------------|
| L     | 67.4% | **37.4%** | **14.3%**       |
| H     | 63.1% | 31.2%     | 11.6%           |

**Verified hypothesis: teacher internal reasoning leads to better student adaptation and out-of-domain generalisation.**

---

## Results

### Main comparison — 3 iterations, NuminaMath-CoT 10K

| Benchmark          | UNDO   | Always-Solution | **Progressive (ours)** | Δ vs UNDO |
|--------------------|:------:|:---------------:|:----------------------:|:---------:|
| GSM8K              | 61.1%  | 65.2%           | **68.2%**              | +7.0 pp   |
| MATH               | 34.6%  | 37.1%           | **39.8%**              | +5.2 pp   |
| MMLU-Pro           | 14.1%  | 15.3%           | **15.6%**              | +1.5 pp   |
| SVAMP              | 88.6%  | 89.1%           | **89.6%**              | +1.0 pp   |
| StrategyQA (OOD)   | 13.4%  | 12.4% ↓         | **14.6%** ↑            | +1.2 pp   |

> Always-Solution degrades on OOD data. Progressive hinting improves both in-domain and out-of-domain — the only method to do so.

### Reasoning-Aware Error Signal comparison

| Validation signal    | Final accuracy |
|----------------------|:--------------:|
| **Our score (A+R)**  | **54.2%**      |
| Binary (0/1)         | 53.3%          |
| BERTScore            | 52.7%          |
| No score             | 52.5%          |

---

## Models & Data

| Role    | Model                        |
|---------|------------------------------|
| Teacher | Qwen3-30B-A3B-Instruct-2507  |
| Student | Qwen2.5-Math-1.5B-Instruct   |

**Training:** NuminaMath-CoT · 10K samples (resource-scaled reproduction)
**Evaluation:** GSM8K · SVAMP · MATH · MMLU-Pro · StrategyQA

---

## Reproduction

### Setup

```bash
conda create -n teacher python=3.10 -y && conda activate teacher
pip install -r requirements.txt && pip install flash-attn --no-build-isolation

conda create -n student python=3.10 -y && conda activate student
pip install -r requirements.txt

export HF_TOKEN="<your_token>"
```

### Full pipeline (orchestrated)

```bash
python src/pipeline.py \
  --dataset      MinTR-KIEU/NuminaMath-CoT-10k \
  --val_dataset  MinTR-KIEU/NuminaMath-val \
  --iterations   3 \
  --hint_strategy   progressive \
  --scoring_signal  reasoning_aware \
  --output_dir   runs/exp_01 \
  --hf_repo      MinTR-KIEU/intelligent-distill-outputs
```

The orchestrator generates `val_signals.jsonl` and prints the exact shell commands for every stage of every iteration.

### Step-by-step

```bash
# Iter 0 — baseline teacher pass
python src/teacher.py --iter 0 --dataset 10k \
  --output runs/iter_0/teacher_output.jsonl \
  --hf_repo MinTR-KIEU/intelligent-distill-outputs

# Student SFT  (after uploading teacher output to HF)
python src/student.py --mode train \
  --dataset  MinTR-KIEU/<teacher-iter0-dataset> \
  --save_dir runs/iter_0/student_ckpt \
  --output   runs/iter_0/student_output.jsonl

# Student validation inference
python src/student.py --mode test \
  --model_path runs/iter_0/student_ckpt \
  --dataset    MinTR-KIEU/NuminaMath-val \
  --output     runs/iter_0/val_raw.jsonl

# Iter 1 — with Progressive Hint Cascade
python src/teacher.py --iter 1 --dataset 10k \
  --hint_strategy      progressive \
  --prev_teacher_jsonl runs/iter_0/teacher_output.jsonl \
  --prev_student_jsonl runs/iter_0/student_output.jsonl \
  --val_signals_jsonl  runs/iter_0/val_signals.jsonl \
  --output runs/iter_1/teacher_output.jsonl \
  --hf_repo MinTR-KIEU/intelligent-distill-outputs
```

### Ablations

```bash
--hint_strategy   always_solution   # always provide full GT solution
--hint_strategy   none              # UNDO baseline — drop on failure
--scoring_signal  binary            # binary {0, 1} error signal
```

### Tests

```bash
pytest                        # 162 tests, all passing
pytest test/test_hint.py      # progressive hint cascade
pytest test/test_scoring.py   # reasoning-aware error signal
pytest test/test_evaluate.py  # LaTeX-aware answer extraction
```

---

## Code Map

```
src/
  pipeline.py        full iterative loop orchestrator  ← start here
  teacher.py         teacher inference · iter 0 and iter K≥1
  student.py         student SFT + benchmark evaluation
  hint.py            Contribution 1: progressive hint cascade
  scoring.py         Contribution 2: S = 0.6·A + 0.4·R
  similarity.py      BERTScore F1 · L/M/H group classification
  evaluate.py        LaTeX-aware \boxed{} answer extraction
  utils.py           HFPusher — background periodic HF dataset uploads
  prompt/
    teacher_prompt.py  iter0 · iterK · answer-hint · solution-hint templates
```

`hint.py` accepts any `prompt → response` callable, keeping it model-agnostic and independently testable. Each module maps to a section of the paper.

---

## Baseline

This work extends **UNDO** (*Iterative Knowledge Distillation via Unlearning and Distillation*, preprint — no official code released). We reproduced UNDO at 10K scale using a ~30B teacher, confirming performance trends consistent with the paper: accuracy peaks at iteration 3, then degrades — a convergence and overfitting signal that holds across both settings.
