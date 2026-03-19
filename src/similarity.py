"""
Teacher Reasoning Fidelity via BERTScore F1
===========================================
Measures how closely a teacher's explanation resembles the GT solution, distinguishing
genuine internal reasoning from solution paraphrasing.

Similarity groups (Section 4.3):
    L  (score < 0.50)        — teacher uses own reasoning; hint had minimal influence
    M  (0.50 ≤ score < 0.75) — teacher is guided by the hint but not copying
    H  (score ≥ 0.75)        — teacher largely paraphrases / copies the GT solution

Verified hypothesis (Section 4.4):
    Students trained on L-group samples consistently outperform H-group students on both
    in-domain (GSM8K, SVAMP, MATH, MMLU-Pro) and OOD (StrategyQA) benchmarks.

    Performance on MATH:  L set 37.4%  vs  H set 31.2%  (+6.2 pp)
    OOD StrategyQA:       L set 14.3%  vs  H set 11.6%  (+2.7 pp)

    This confirms: teacher internal reasoning → better student adaptation and
    out-of-domain generalisation; solution copying → in-domain overfitting.

Pilot study (Section 3.2, see also slide 12-13):
    Correct hint  → BERTScore F1 ≈ 0.6–0.9  (teacher follows the solution)
    Wrong hint    → BERTScore F1 ≈ 0.1–0.4  (teacher resists, reasons independently)

Usage:
    scores = compute_bertscore_batch(teacher_responses, gt_solutions)
    records = annotate_similarity_groups(records, "teacher_response", "gt_solution")
    stats   = group_statistics(records)
"""

from typing import Dict, List, Literal, Optional

# ── Thresholds (Section 4.3) ──────────────────────────────────────────────────

L_THRESHOLD = 0.50   # below → L  (independent reasoning)
H_THRESHOLD = 0.75   # above → H  (solution copying)

SimilarityGroup = Literal["L", "M", "H"]


# ── Classification ────────────────────────────────────────────────────────────

def classify_similarity_group(score: float) -> SimilarityGroup:
    """
    Map a BERTScore F1 value to a similarity group (L / M / H).

    L (<0.50):  Teacher explanation diverges substantially from GT solution.
                Internal reasoning dominates; hint had minimal influence.
    M (0.50–0.75): Teacher follows the hint's structure but adds own steps.
    H (≥0.75):  Teacher closely paraphrases or reproduces the GT solution.
    """
    if score < L_THRESHOLD:
        return "L"
    if score < H_THRESHOLD:
        return "M"
    return "H"


# ── BERTScore computation ─────────────────────────────────────────────────────

def compute_bertscore_f1(hypothesis: str, reference: str, lang: str = "en") -> float:
    """
    BERTScore F1 between a teacher explanation (hypothesis) and GT solution (reference).

    Requires:  pip install bert-score
    Returns:   F1 score in [0, 1].
    """
    try:
        from bert_score import score as _bert_score
    except ImportError as exc:
        raise ImportError(
            "bert-score is required for similarity analysis. "
            "Install with: pip install bert-score"
        ) from exc
    _, _, F1 = _bert_score([hypothesis], [reference], lang=lang, verbose=False)
    return float(F1[0])


def compute_bertscore_batch(
    hypotheses: List[str],
    references: List[str],
    lang: str = "en",
    batch_size: int = 32,
) -> List[float]:
    """
    Compute BERTScore F1 for a list of (teacher_response, gt_solution) pairs.
    More efficient than calling compute_bertscore_f1 in a loop.
    """
    try:
        from bert_score import score as _bert_score
    except ImportError as exc:
        raise ImportError(
            "bert-score is required for similarity analysis. "
            "Install with: pip install bert-score"
        ) from exc
    _, _, F1 = _bert_score(
        hypotheses, references, lang=lang, verbose=False, batch_size=batch_size
    )
    return [float(f) for f in F1]


# ── Dataset annotation ────────────────────────────────────────────────────────

def annotate_similarity_groups(
    records: List[Dict],
    hypothesis_key: str = "teacher_response",
    reference_key:  str = "gt_solution",
    lang: str = "en",
) -> List[Dict]:
    """
    Add 'bertscore_f1' and 'similarity_group' fields to each record in-place (copy).

    Args:
        records:        List of dicts containing teacher responses and GT solutions.
        hypothesis_key: Field name for the teacher explanation.
        reference_key:  Field name for the GT solution.

    Returns:
        New list of records with two additional fields per record.
    """
    hypotheses = [r[hypothesis_key] for r in records]
    references  = [r[reference_key]  for r in records]
    scores      = compute_bertscore_batch(hypotheses, references, lang=lang)

    return [
        {**rec, "bertscore_f1": round(score, 4), "similarity_group": classify_similarity_group(score)}
        for rec, score in zip(records, scores)
    ]


def group_statistics(records: List[Dict], score_key: str = "bertscore_f1") -> Dict:
    """
    Summarise the L / M / H distribution over a dataset of annotated records.

    Returns:
        {
          "total": int,
          "L": {"count": int, "fraction": float, "mean_score": float},
          "M": { ... },
          "H": { ... },
        }
    """
    buckets: Dict[str, List[float]] = {"L": [], "M": [], "H": []}

    for rec in records:
        score = rec.get(score_key)
        if score is None:
            continue
        group = rec.get("similarity_group") or classify_similarity_group(score)
        if group in buckets:
            buckets[group].append(score)

    n_total = len(records)
    stats: Dict = {"total": n_total}
    for g, scores in buckets.items():
        n = len(scores)
        stats[g] = {
            "count":      n,
            "fraction":   round(n / n_total, 4) if n_total else 0.0,
            "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        }
    return stats
