"""Tests for Progressive Hint Cascade (Contribution 1, src/hint.py)."""

import pytest
from src.hint import HintResult, progressive_hint_cascade


# ── Minimal tokenizer stub ────────────────────────────────────────────────────

class _FakeTok:
    def apply_chat_template(self, messages, **kwargs):
        return " | ".join(m["content"] for m in messages)


_tok = _FakeTok()

VAL_SIGNALS = [
    {"question": "What is 2+2?", "student_response": "5", "score": -1.0}
]


def _always_correct(prompt: str) -> str:
    return r"Therefore: Final Answer: $\boxed{42}$"


def _always_wrong(prompt: str) -> str:
    return "I don't know. Final Answer: $\\boxed{0}$"


def _correct_on_stage(target_stage: str):
    """Return an infer_fn that succeeds only when the prompt contains the hint marker."""
    def infer(prompt: str) -> str:
        if target_stage == "none" and "HINT" not in prompt:
            return r"Final Answer: $\boxed{42}$"
        if target_stage == "answer" and "Correct Final Answer" in prompt:
            return r"Final Answer: $\boxed{42}$"
        if target_stage == "solution" and "Complete Solution" in prompt:
            return r"Final Answer: $\boxed{42}$"
        return r"Final Answer: $\boxed{0}$"
    return infer


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cascade_succeeds_at_stage_none():
    result = progressive_hint_cascade(
        problem="What is 6×7?",
        gt_answer=r"\boxed{42}",
        gt_solution="6 × 7 = 42",
        val_error_signals=VAL_SIGNALS,
        student_response_prev=None,
        teacher_response_prev=None,
        teacher_infer_fn=_always_correct,
        tokenizer=_tok,
    )
    assert result is not None
    assert result.hint_stage == "none"
    assert result.extracted_answer is not None


def test_cascade_succeeds_at_stage_answer():
    result = progressive_hint_cascade(
        problem="What is 6×7?",
        gt_answer=r"\boxed{42}",
        gt_solution="6 × 7 = 42",
        val_error_signals=VAL_SIGNALS,
        student_response_prev="I think it's 40.",
        teacher_response_prev="I got 40.",
        teacher_infer_fn=_correct_on_stage("answer"),
        tokenizer=_tok,
    )
    assert result is not None
    assert result.hint_stage == "answer"


def test_cascade_succeeds_at_stage_solution():
    result = progressive_hint_cascade(
        problem="What is 6×7?",
        gt_answer=r"\boxed{42}",
        gt_solution="6 × 7 = 42",
        val_error_signals=VAL_SIGNALS,
        student_response_prev=None,
        teacher_response_prev=None,
        teacher_infer_fn=_correct_on_stage("solution"),
        tokenizer=_tok,
    )
    assert result is not None
    assert result.hint_stage == "solution"


def test_cascade_returns_none_when_all_fail():
    result = progressive_hint_cascade(
        problem="What is 6×7?",
        gt_answer=r"\boxed{42}",
        gt_solution="6 × 7 = 42",
        val_error_signals=VAL_SIGNALS,
        student_response_prev=None,
        teacher_response_prev=None,
        teacher_infer_fn=_always_wrong,
        tokenizer=_tok,
    )
    assert result is None


def test_cascade_result_fields():
    result = progressive_hint_cascade(
        problem="What is 6×7?",
        gt_answer=r"\boxed{42}",
        gt_solution="6 × 7 = 42",
        val_error_signals=[],
        student_response_prev=None,
        teacher_response_prev=None,
        teacher_infer_fn=_always_correct,
        tokenizer=_tok,
    )
    assert isinstance(result, HintResult)
    assert result.teacher_response != ""
    assert result.hint_stage in ("none", "answer", "solution")


def test_cascade_hint_stage_ordering():
    """Cascade must try 'none' before 'answer' before 'solution'."""
    call_log = []

    def logging_infer(prompt: str) -> str:
        if "Complete Solution" in prompt:
            call_log.append("solution")
            return r"Final Answer: $\boxed{42}$"
        if "Correct Final Answer" in prompt:
            call_log.append("answer")
            return r"Final Answer: $\boxed{0}$"  # fail
        call_log.append("none")
        return r"Final Answer: $\boxed{0}$"  # fail

    progressive_hint_cascade(
        problem="p", gt_answer=r"\boxed{42}", gt_solution="sol",
        val_error_signals=[], student_response_prev=None, teacher_response_prev=None,
        teacher_infer_fn=logging_infer, tokenizer=_tok,
    )
    assert call_log == ["none", "answer", "solution"]
