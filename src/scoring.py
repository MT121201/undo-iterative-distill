"""
Reasoning-Aware Error Signal — Contribution 2
=============================================
Replaces UNDO's binary {0, 1} validation score with a continuous signal that captures
both answer correctness and reasoning quality, giving the teacher a richer view of how
the student is improving across iterations.

Signal formula (Section 3.2):
    S = α · A + β · R,   α = 0.6,  β = 0.4

    A ∈ {+1, −1}                    — answer correctness
    R ∈ {−1, −0.5, 0, +0.5, +1}    — reasoning quality (LLM-judged rubric)
    S ∈ [−1, +1]

Example trajectory showing structural improvement before answer becomes correct:
    Iteration:   init    1      2      3
    Score S:   [−0.9,  −0.4,  +0.2,  +0.9]

Comparison of scoring signals (all using Progressive Hints pipeline, Section 4.2):
    Our score (A+R)  →  54.2%  final accuracy
    Binary (0/1)     →  53.3%
    BERTScore        →  52.7%
    No score         →  52.5%

Usage:
    signal = compute_error_signal(student_response, gt)             # binary fallback
    signal = compute_error_signal(student_response, gt, judge_fn)   # full reasoning score
    # signal = {"score": 0.2, "answer_correctness": -1.0, "reasoning_quality": 0.5, ...}
"""

import re
from typing import Callable, Dict, List, Literal, Optional

from evaluate import evaluate_model_response

# ── Signal weights (Section 3.2) ──────────────────────────────────────────────

ALPHA = 0.6   # weight for answer correctness  (A)
BETA  = 0.4   # weight for reasoning quality   (R)

# ── Reasoning quality rubric ──────────────────────────────────────────────────

RUBRIC: Dict[float, str] = {
     1.0: "correct logic, minimal gaps",
     0.5: "minor slip, overall structure good",
     0.0: "unclear or missing key steps",
    -0.5: "wrong approach but coherent",
    -1.0: "nonsense or cannot reason at all",
}

_RUBRIC_VALUES = sorted(RUBRIC.keys())   # [−1.0, −0.5, 0.0, 0.5, 1.0]

# ── LLM judge prompt ──────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating the reasoning quality of a student's math solution.

Ground-truth answer: {gt}

Student's solution:
{student_response}

Score the student's reasoning quality using EXACTLY one of these values:
  +1.0  correct logic, minimal gaps
  +0.5  minor slip, overall structure good
   0.0  unclear or missing key steps
  -0.5  wrong approach but coherent
  -1.0  nonsense or cannot reason at all

Respond with ONLY valid JSON (no markdown):
{{"reasoning_score": <value>, "justification": "<one sentence>"}}\
"""


def _parse_judge_response(response: str) -> Optional[float]:
    """Extract reasoning_score from judge model JSON output; snap to nearest rubric value."""
    match = re.search(r'"reasoning_score"\s*:\s*(-?[01](?:\.[05])?)', response)
    if not match:
        return None
    raw = float(match.group(1))
    return min(_RUBRIC_VALUES, key=lambda v: abs(v - raw))


# ── Core scoring functions ────────────────────────────────────────────────────

def compute_answer_correctness(student_response: str, gt: str) -> float:
    """A in S = α·A + β·R.  Returns +1.0 if correct, −1.0 otherwise."""
    result = evaluate_model_response(student_response, gt)
    return 1.0 if result.get("is_correct") else -1.0


def compute_reasoning_quality_llm(
    student_response: str,
    gt: str,
    judge_fn: Callable[[str], str],
) -> float:
    """
    R in S = α·A + β·R.
    Queries the judge model with the rubric prompt; parses the JSON score.
    Falls back to 0.0 (neutral) if the response is unparseable.
    """
    prompt   = _JUDGE_PROMPT.format(gt=gt, student_response=student_response)
    response = judge_fn(prompt)
    score    = _parse_judge_response(response)
    return score if score is not None else 0.0


def compute_error_signal(
    student_response: str,
    gt: str,
    judge_fn: Optional[Callable[[str], str]] = None,
) -> Dict:
    """
    Compute the Reasoning-Aware Error Signal: S = α·A + β·R.

    When judge_fn is None, falls back to binary scoring (R ≡ A), matching the
    original UNDO signal.  This lets the function be used in both modes with a
    single interface — the caller controls which signal type is active.

    Args:
        student_response: Full student solution string.
        gt:               Ground-truth answer (may include \\boxed{}).
        judge_fn:         Optional LLM callable: prompt → response string.
                          Pass None for binary fallback (no judge model needed).

    Returns:
        {
          "score":               float in [−1, +1],
          "answer_correctness":  float in {−1.0, +1.0},
          "reasoning_quality":   float in {−1.0, −0.5, 0.0, +0.5, +1.0},
          "signal_type":         "reasoning_aware" | "binary",
        }
    """
    A = compute_answer_correctness(student_response, gt)

    if judge_fn is not None:
        R           = compute_reasoning_quality_llm(student_response, gt, judge_fn)
        signal_type = "reasoning_aware"
    else:
        R           = A   # binary fallback: R ≡ A  →  S ∈ {−1, +1}
        signal_type = "binary"

    S = round(ALPHA * A + BETA * R, 4)
    return {
        "score":              S,
        "answer_correctness": A,
        "reasoning_quality":  R,
        "signal_type":        signal_type,
    }


def format_signal_trajectory(signals: List[Dict]) -> str:
    """
    Summarise a sequence of per-iteration error signals for inclusion in teacher context.

    Example output:
        "Scores: [−0.9 (init), −0.4 (iter 1), +0.2 (iter 2)]"
    """
    labels = ["init"] + [f"iter {i}" for i in range(1, len(signals))]
    parts  = [f"{s.get('score', 0.0):+.1f} ({lbl})" for lbl, s in zip(labels, signals)]
    return "Scores: [" + ", ".join(parts) + "]"
