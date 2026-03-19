"""Tests for Reasoning-Aware Error Signal (Contribution 2, src/scoring.py)."""

import pytest
from src.scoring import (
    ALPHA,
    BETA,
    RUBRIC,
    _RUBRIC_VALUES,
    _parse_judge_response,
    compute_answer_correctness,
    compute_error_signal,
    compute_reasoning_quality_llm,
    format_signal_trajectory,
)


# ── compute_answer_correctness ────────────────────────────────────────────────

@pytest.mark.parametrize("response,gt,expected", [
    (r"\boxed{42}",      r"\boxed{42}",  1.0),
    (r"\boxed{0}",       r"\boxed{42}", -1.0),
    (r"\boxed{B}",       r"\boxed{B}",   1.0),
    (r"No box",          r"\boxed{42}", -1.0),
    (r"\boxed{\frac{1}{2}}", r"\boxed{0.5}", 1.0),
])
def test_compute_answer_correctness(response, gt, expected):
    assert compute_answer_correctness(response, gt) == expected


# ── _parse_judge_response ─────────────────────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    ('{"reasoning_score": 1.0, "justification": "good"}',   1.0),
    ('{"reasoning_score": 0.5, "justification": "ok"}',     0.5),
    ('{"reasoning_score": 0.0, "justification": "meh"}',    0.0),
    ('{"reasoning_score": -0.5, "justification": "bad"}',  -0.5),
    ('{"reasoning_score": -1.0, "justification": "none"}', -1.0),
    ("unparseable garbage",                                  None),
    ("{}",                                                   None),
])
def test_parse_judge_response(response, expected):
    assert _parse_judge_response(response) == expected


# ── compute_reasoning_quality_llm ─────────────────────────────────────────────

def test_reasoning_quality_llm_valid():
    judge = lambda _: '{"reasoning_score": 0.5, "justification": "minor slip"}'
    score = compute_reasoning_quality_llm("student resp", r"\boxed{1}", judge)
    assert score == 0.5


def test_reasoning_quality_llm_fallback_on_bad_json():
    judge = lambda _: "I cannot decide."
    score = compute_reasoning_quality_llm("student resp", r"\boxed{1}", judge)
    assert score == 0.0   # neutral fallback


# ── compute_error_signal ──────────────────────────────────────────────────────

def test_error_signal_binary_correct():
    sig = compute_error_signal(r"\boxed{5}", r"\boxed{5}")
    assert sig["signal_type"] == "binary"
    assert sig["answer_correctness"] == 1.0
    # binary: R = A → S = α·1 + β·1 = 1.0
    assert sig["score"] == pytest.approx(ALPHA + BETA)


def test_error_signal_binary_wrong():
    sig = compute_error_signal(r"\boxed{0}", r"\boxed{5}")
    assert sig["signal_type"] == "binary"
    assert sig["answer_correctness"] == -1.0
    assert sig["score"] == pytest.approx(-(ALPHA + BETA))


def test_error_signal_reasoning_aware():
    judge = lambda _: '{"reasoning_score": 0.5, "justification": "minor slip"}'
    sig = compute_error_signal(r"\boxed{0}", r"\boxed{5}", judge_fn=judge)
    assert sig["signal_type"] == "reasoning_aware"
    assert sig["answer_correctness"] == -1.0
    assert sig["reasoning_quality"] == 0.5
    expected_score = ALPHA * (-1.0) + BETA * 0.5
    assert sig["score"] == pytest.approx(expected_score, abs=1e-4)


def test_error_signal_score_in_range():
    for resp, gt in [(r"\boxed{1}", r"\boxed{1}"), (r"\boxed{0}", r"\boxed{1}")]:
        sig = compute_error_signal(resp, gt)
        assert -1.0 <= sig["score"] <= 1.0


def test_error_signal_returns_all_keys():
    sig = compute_error_signal(r"\boxed{1}", r"\boxed{1}")
    assert set(sig.keys()) == {"score", "answer_correctness", "reasoning_quality", "signal_type"}


# ── format_signal_trajectory ──────────────────────────────────────────────────

def test_format_signal_trajectory():
    signals = [{"score": -0.9}, {"score": -0.4}, {"score": 0.2}]
    out = format_signal_trajectory(signals)
    assert "-0.9" in out
    assert "-0.4" in out
    assert "+0.2" in out
    assert "init" in out
    assert "iter 1" in out
    assert "iter 2" in out


def test_format_signal_trajectory_single():
    signals = [{"score": 1.0}]
    out = format_signal_trajectory(signals)
    assert "init" in out
    assert "+1.0" in out


# ── constants ─────────────────────────────────────────────────────────────────

def test_alpha_beta_sum_to_one():
    assert ALPHA + BETA == pytest.approx(1.0)


def test_rubric_values_are_valid():
    assert set(RUBRIC.keys()) == {1.0, 0.5, 0.0, -0.5, -1.0}
    assert _RUBRIC_VALUES == [-1.0, -0.5, 0.0, 0.5, 1.0]
