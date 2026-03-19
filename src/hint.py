"""
Progressive Hint Cascade — Contribution 1
==========================================
Recovers incorrect teacher attempts via staged hint provision, preserving teacher
internal reasoning by giving the minimum hint necessary to produce a correct response.

Algorithm (Section 3.1):
    Stage 'none'     — teacher attempts with UNDO context only (no hint)
    Stage 'answer'   — provide the correct final answer as a hint
    Stage 'solution' — provide the full GT chain-of-thought as a hint
    Drop             — all stages failed; sample is discarded

Empirical motivation (pilot study, Section 3.2):
    When given a *correct* hint, BERTScore F1(teacher_response, GT_solution) peaks at
    0.6–0.9 → teacher paraphrases. When given a *wrong* or no hint, it drops to 0.1–0.4
    → teacher reasons independently.

    Progressive hinting shifts the recovered dataset toward the low-similarity (L) region,
    while always-solution hinting concentrates it in the mid/high (M/H) region.
    L-region training samples produce better student OOD generalisation (Section 4.4).

Usage:
    result = progressive_hint_cascade(
        problem, gt_answer, gt_solution,
        val_error_signals, student_response_prev, teacher_response_prev,
        teacher_infer_fn=lambda prompt: model_generate(prompt),
        tokenizer=tok,
    )
    if result is None:
        pass  # sample dropped
    else:
        print(result.hint_stage, result.teacher_response)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

from evaluate import evaluate_model_response
from prompt.teacher_prompt import (
    build_teacher_prompt_iterK,
    build_teacher_prompt_answer_hint,
    build_teacher_prompt_solution_hint,
)

HintStage = Literal["none", "answer", "solution"]


@dataclass
class HintResult:
    """Outcome of one successful stage in the progressive hint cascade."""
    teacher_response: str
    hint_stage: HintStage       # which stage produced a correct response
    extracted_answer: Optional[str]
    comparison_mode: Optional[str]


def progressive_hint_cascade(
    problem: str,
    gt_answer: str,
    gt_solution: str,
    val_error_signals: List[Dict],
    student_response_prev: Optional[str],
    teacher_response_prev: Optional[str],
    teacher_infer_fn: Callable[[str], str],
    tokenizer,
) -> Optional[HintResult]:
    """
    Attempt to recover a failed teacher response via progressive hinting.

    Tries three stages in order; returns at the first stage that produces a
    correct answer. Returns None if all stages fail (sample is dropped).

    Args:
        problem:               Training question text.
        gt_answer:             Ground-truth final answer (e.g. "\\boxed{42}").
        gt_solution:           Full GT chain-of-thought (for stage 'solution').
        val_error_signals:     [{question, student_response, score}, ...] from val set.
        student_response_prev: Student's previous attempt at this problem (K−1).
        teacher_response_prev: Teacher's previous attempt at this problem (K−1).
        teacher_infer_fn:      Callable: prompt str → teacher response str.
                               Encapsulates model loading; keeps this module model-agnostic.
        tokenizer:             Tokenizer for applying the chat template.

    Returns:
        HintResult on the first successful stage, or None if dropped.
    """
    stages: List[tuple] = [
        (
            "none",
            build_teacher_prompt_iterK(
                problem, val_error_signals,
                student_response_prev, teacher_response_prev, tokenizer,
            ),
        ),
        (
            "answer",
            build_teacher_prompt_answer_hint(
                problem, gt_answer, val_error_signals,
                student_response_prev, teacher_response_prev, tokenizer,
            ),
        ),
        (
            "solution",
            build_teacher_prompt_solution_hint(
                problem, gt_solution, val_error_signals,
                student_response_prev, teacher_response_prev, tokenizer,
            ),
        ),
    ]

    for stage, prompt in stages:
        response = teacher_infer_fn(prompt)
        eval_result = evaluate_model_response(response, gt_answer)
        if eval_result.get("is_correct"):
            return HintResult(
                teacher_response=response,
                hint_stage=stage,
                extracted_answer=eval_result.get("extracted_answer"),
                comparison_mode=eval_result.get("comparison_mode"),
            )

    return None  # all stages failed — sample dropped
