import re
from fractions import Fraction
from typing import Optional, Dict, Any

_BOXED_RE = re.compile(r'\\boxed\{(.+?)\}', re.DOTALL)

def _latex_frac_to_fraction(s: str) -> Optional[Fraction]:
    """
    Convert simple latex fractions like \\frac{a}{b} to Fraction(a, b).
    Returns None if it doesn't match.
    """
    m = re.fullmatch(r'\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', s.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    return Fraction(a, b)

def _to_fraction_if_possible(s: str) -> Optional[Fraction]:
    """
    Try to interpret a string as a rational number.
    Supports integers, decimals, a/b, and \\frac{a}{b}.
    """
    s = s.strip()
    # \frac{a}{b}
    frac = _latex_frac_to_fraction(s)
    if frac is not None:
        return frac

    # a/b (plain)
    if re.fullmatch(r'[+-]?\d+\s*/\s*[+-]?\d+', s):
        a, b = re.split(r'/', s)
        b = int(b.strip())
        if b == 0:
            return None
        return Fraction(int(a.strip()), b)

    # integer or decimal
    if re.fullmatch(r'[+-]?(\d+(\.\d*)?|\.\d+)', s):
        # convert decimal to Fraction exactly
        return Fraction(s)

    return None

def _normalize_text(s: str) -> str:
    """
    Minimal normalization for text/latex: trim and collapse whitespace.
    """
    return re.sub(r'\s+', ' ', s.strip())

def evaluate_teacher_response(
    response: str,
    expected: Optional[str] = None,
    numeric_tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Evaluate a teacher response that should contain a final answer inside \\boxed{...}.

    Args:
        response: model-generated text.
        expected: ground-truth final answer (numeric or latex/text). If None, only checks for boxed presence.
        numeric_tolerance: absolute tolerance when comparing numeric answers.

    Returns:
        {
          "has_boxed": bool,
          "extracted_answer": str or None,
          "is_correct": bool or None,
          "comparison_mode": "numeric" | "string" | None,
          "details": str
        }
    """
    matches = list(_BOXED_RE.finditer(response))
    if not matches:
        return {
            "has_boxed": False,
            "extracted_answer": None,
            "is_correct": None,
            "comparison_mode": None,
            "details": "No \\boxed{...} found."
        }

    # Use the last boxed answer
    extracted = matches[-1].group(1).strip()

    if expected is None:
        return {
            "has_boxed": True,
            "extracted_answer": extracted,
            "is_correct": None,
            "comparison_mode": None,
            "details": "Found boxed answer; no expected value provided."
        }

    # Try numeric comparison first
    ext_frac = _to_fraction_if_possible(extracted)
    exp_frac = _to_fraction_if_possible(expected.strip())

    if ext_frac is not None and exp_frac is not None:
        # Numeric compare with tolerance
        diff = abs(float(ext_frac) - float(exp_frac))
        return {
            "has_boxed": True,
            "extracted_answer": extracted,
            "is_correct": diff <= numeric_tolerance,
            "comparison_mode": "numeric",
            "details": f"Numeric diff={diff}, tol={numeric_tolerance}"
        }

    # Fallback to string/latex normalized comparison
    is_equal = _normalize_text(extracted) == _normalize_text(expected)
    return {
        "has_boxed": True,
        "extracted_answer": extracted,
        "is_correct": is_equal,
        "comparison_mode": "string",
        "details": "Compared normalized strings."
    }

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    resp = "… Therefore the answer is \\boxed{10}."
    print(evaluate_teacher_response(resp, expected="10"))          # numeric -> True
    print(evaluate_teacher_response(resp, expected="\\frac{20}{2}")) # numeric -> True
    print(evaluate_teacher_response(resp, expected="ten"))         # string -> False
    print(evaluate_teacher_response("… no box here", expected="10"))
