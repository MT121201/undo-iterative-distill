import math
import re
from fractions import Fraction
from typing import Optional, Dict, Any, List, Tuple

# -----------------------------
# Balanced \boxed{...} extractor
# -----------------------------

def _find_last_boxed(text: str) -> Optional[str]:
    """
    Extract the LAST \\boxed{...} content using balanced-brace scanning.
    Returns the inner content string (without the outer { }) or None if not found.
    """
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        # Unbalanced; treat as not found
        return None
    return "".join(out).strip()

# -----------------------------
# Cleaning / normalization
# -----------------------------

# Remove $…$ or \( … \) wrappers around the whole string
_wrap_math_dollars = re.compile(r"^\$(.*)\$$", re.DOTALL)
_wrap_math_parens  = re.compile(r"^\\\((.*)\\\)$", re.DOTALL)

# Remove simple LaTeX text macros wrapping a single argument
_text_wrappers = [
    re.compile(r"^\\textbf\{(.*)\}$", re.DOTALL),
    re.compile(r"^\\mathrm\{(.*)\}$", re.DOTALL),
    re.compile(r"^\\text\{(.*)\}$", re.DOTALL),
]

# Strip surrounding parentheses like (D) or [A]
_paren_choice = re.compile(r"^\(\s*([A-Za-z])\s*\)$")

# Circled numerals mapping ①..⑩
_CIRCLED = {c: i for c, i in zip("①②③④⑤⑥⑦⑧⑨⑩", range(1, 11))}

def _strip_wrappers(s: str) -> str:
    s = s.strip()

    # strip $...$ or \( ... \)
    m = _wrap_math_dollars.fullmatch(s)
    if m:
        s = m.group(1).strip()
    m = _wrap_math_parens.fullmatch(s)
    if m:
        s = m.group(1).strip()

    # peel common text wrappers repeatedly
    changed = True
    while changed:
        changed = False
        for rx in _text_wrappers:
            m = rx.fullmatch(s)
            if m:
                s = m.group(1).strip()
                changed = True

    # Keep: strip trailing punctuation BEFORE bracket peeling
    s = s.rstrip(".;:,").strip()

    # --- NEW: normalize LaTeX spacing and clean stray backslashes
    s = s.replace(r"\ ", " ")
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\!", " ")
    s = s.replace(r"\:", " ")
    s = re.sub(r"\\+$", "", s).strip()

    # remove one layer of surrounding brackets/parentheses if it wraps a single letter
    m = re.fullmatch(r"\(\s*([A-Za-z])\s*\)", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"\[\s*([A-Za-z])\s*\]", s)
    if m:
        return m.group(1)

    return s



def _normalize_text(s: str) -> str:
    """Trim + collapse whitespace to a single space (e.g., '1 000' -> '1 000')."""
    return re.sub(r"\s+", " ", s.strip())

# -----------------------------
# Numeric parsing helpers
# -----------------------------

def _latex_frac_to_fraction(s: str) -> Optional[Fraction]:
    m = re.fullmatch(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", s.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    return Fraction(a, b)

def _strip_common_numerical_decorations(s: str) -> str:
    """
    Remove $, \$, thousands separators (',' or thin spaces like '\,' or regular spaces between digits),
    degree markers, percent signs, and LaTeX thin/medium spaces.
    """
    # Normalize common LaTeX spacing commands
    s = s.replace(r"\:", " ")       # medium space -> space
    s = s.replace(r"\,", "")        # thin space -> remove (often used as 1\,000)
    s = s.replace("\u00A0", " ")    # non-breaking space -> space
    s = s.strip()

    # Remove leading currency symbols ($ or \$, possibly repeated)
    s = re.sub(r"^(?:\\\$|\$)+\s*", "", s)

    # Remove thousands separators:
    s = s.replace(",", "")  # commas out
    # remove *spaces between digits* (handles "1 000", "1 000")
    s = re.sub(r"(?<=\d)[\s\u00A0](?=\d)", "", s)

    # Degrees: 60^{\circ} -> 60, also "60 degrees", "90°"
    s = re.sub(r"^([+-]?\d+(?:\.\d+)?)\s*\^\s*\\?{\\?circ\\?}", r"\1", s)
    s = re.sub(r"^([+-]?\d+(?:\.\d+)?)\s*(?:degrees?|°)$", r"\1", s, flags=re.I)

    # Trailing percent -> numeric
    s = re.sub(r"^([+-]?\d+(?:\.\d+)?)\s*%$", r"\1", s)

    # Collapse leftover whitespace at ends
    s = _normalize_text(s)
    return s

def _maybe_factorial_to_fraction(s: str) -> Optional[Fraction]:
    """
    Parse simple factorial like 'n!' (optionally with leading +/- and spaces) to a Fraction(int).
    Only handles a single integer factorial (no chained ops). Returns None if not factorial.
    """
    m = re.fullmatch(r'\s*([+-]?\d+)\s*!\s*', s)
    if not m:
        return None
    n = int(m.group(1))
    if n < 0:
        return None
    return Fraction(math.factorial(n))

def _to_fraction_if_possible(s: str) -> Optional[Fraction]:
    s = _strip_common_numerical_decorations(s)

    # factorial n!
    fac = _maybe_factorial_to_fraction(s)
    if fac is not None:
        return fac

    # \frac{a}{b}
    frac = _latex_frac_to_fraction(s)
    if frac is not None:
        return frac

    # a/b
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", s):
        a, b = re.split(r"/", s)
        b = int(b.strip())
        if b == 0:
            return None
        return Fraction(int(a.strip()), b)

    # integer or decimal
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", s):
        return Fraction(s)

    return None


# -----------------------------
# Candidate extraction from box
# -----------------------------


def _tokenize_box_candidates(s: str) -> List[str]:
    s0 = _strip_wrappers(s)

    # --- NEW: proactively surface a letter choice anywhere (e.g., \textbf{(B)} ... 2\sqrt{...})
    choice_hits: List[str] = []
    for rx in [
        re.compile(r"\\textbf\{\(\s*([A-Za-z])\s*\)\}"),
        re.compile(r"\\mathrm\{\(\s*([A-Za-z])\s*\)\}"),
        re.compile(r"\\text\{\(\s*([A-Za-z])\s*\)\}"),
    ]:
        m = rx.search(s0)
        if m:
            choice_hits.append(m.group(1).upper())
            break
    if not choice_hits:
        m = re.search(r"[\(\[]\s*([A-Za-z])\s*[\)\]]", s0)
        if m:
            choice_hits.append(m.group(1).upper())

    tokens = re.split(r"\s+", s0) if s0 else []

    expanded: List[str] = []
    # Seed proactively found choice first
    for ch in choice_hits:
        expanded.append(ch)

    for t in tokens:
        tt = _strip_wrappers(t)
        tt = tt.rstrip("\\")  # strip trailing backslashes

        # (D) -> D
        m = _paren_choice.fullmatch(tt)
        if m:
            tt = m.group(1)

        # circled numerals
        if len(tt) == 1 and tt in _CIRCLED:
            expanded.append(str(_CIRCLED[tt]))
            continue

        if re.fullmatch(r"\(?[A-Za-z]\)?", tt):
            expanded.append(tt.strip("()").upper())
            continue

        expanded.append(tt)

    seen = set()
    out = []
    for t in expanded:
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    return out or ([s0] if s0 else [])

def _parse_pm_number(t: str) -> Optional[Tuple[float, float]]:
    """
    Parse ± forms like '±3' or '\pm 3' => (-3, 3)
    """
    if re.fullmatch(r"[±]\s*([0-9]+(?:\.[0-9]+)?)", t):
        v = float(t.replace("±", "").strip())
        return (-v, v)
    if re.fullmatch(r"\\pm\s*([0-9]+(?:\.[0-9]+)?)", t):
        v = float(re.sub(r"\\pm\s*", "", t))
        return (-v, v)
    return None

def _first_letter_choice(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        if re.fullmatch(r"[A-Za-z]", t):
            return t.upper()
    return None

def _first_numeric_fraction(tokens: List[str]) -> Optional[Fraction]:
    for t in tokens:
        f = _to_fraction_if_possible(t)
        if f is not None:
            return f
    return None

def _first_pm(tokens: List[str]) -> Optional[Tuple[float, float]]:
    for t in tokens:
        pair = _parse_pm_number(t)
        if pair is not None:
            return pair
    return None

def _looks_like_list(expr: str) -> bool:
    """Heuristic: detect if expr is a list/interval of multiple values, not a single number."""
    if not expr:
        return False
    # If comma-separated values like "-3, -1, 1"
    if "," in expr:
        return True
    # If interval style (a, b)
    if re.match(r"^\(?\s*[+-]?\d+(\.\d+)?\s*,\s*[+-]?\d", expr):
        return True
    return False

# -----------------------------
# Public APIs
# -----------------------------
def evaluate_teacher_response(
    response: str,
    expected: Optional[str] = None,
    numeric_tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Evaluates a teacher's response string against an expected answer, supporting numeric, symbolic (±), letter choice, and string-based comparisons.
    Parameters:
        response (str): The teacher's response, typically containing a boxed answer (e.g., '\\boxed{...}').
        expected (Optional[str]): The expected answer for comparison. If None, only extraction is performed.
        numeric_tolerance (float): Tolerance for numeric comparisons (default: 1e-9).
    Returns:
        Dict[str, Any]: A dictionary with the following keys:
            - 'has_boxed' (bool): Whether a boxed answer was found in the response.
            - 'extracted_answer' (str or None): The extracted answer from the boxed content.
            - 'is_correct' (bool or None): Whether the extracted answer matches the expected answer (None if expected is not provided).
            - 'comparison_mode' (str or None): The mode of comparison used ('pm', 'numeric', 'choice', 'string', or None).
            - 'details' (str): Additional details about the comparison process.
    Comparison Modes:
        - 'pm': Handles ± (plus-minus) symmetry, accepting either sign if appropriate.
        - 'numeric': Compares numeric or fractional answers within the specified tolerance.
        - 'choice': Compares letter choices (e.g., 'A', 'B', 'C').
        - 'string': Falls back to normalized string comparison for other cases.
    Notes:
        - If the response or expected answer contains multiple numbers (e.g., lists or intervals), string comparison is used.
        - If no boxed answer is found, returns with 'has_boxed' set to False and 'is_correct' as None.
    """


    inner = _find_last_boxed(response)
    if inner is None:
        return {
            "has_boxed": False,
            "extracted_answer": None,
            "is_correct": None,
            "comparison_mode": None,
            "details": "No \\boxed{...} found."
        }

    inner_clean = _strip_wrappers(inner)
    tokens = _tokenize_box_candidates(inner)

    # Treat expected similar to response (BOXED -> clean -> tokens)
    expected_inner = expected_tokens = expected_clean = None
    if expected is not None:
        expected_inner = _find_last_boxed(expected)
        if expected_inner is not None:
            expected_clean = _strip_wrappers(expected_inner)
            expected_tokens = _tokenize_box_candidates(expected_inner)
        else:
            expected_clean = _strip_wrappers(expected)
            expected_tokens = _tokenize_box_candidates(expected)

    # ---- Helper lambdas for symmetric numeric/pm parsing from token lists
    def first_numeric(tokens_):
        if not tokens_:
            return None
        return _first_numeric_fraction(tokens_)

    def first_pm(tokens_):
        if not tokens_:
            return None
        return _first_pm(tokens_)

    # 1) ±x symmetry
    pm_resp = first_pm(tokens)
    pm_exp  = first_pm(expected_tokens) if expected_tokens else None
    if pm_resp is not None or pm_exp is not None:
        # If only one side is ±, accept if the other side matches either sign.
        if expected is None:
            return {
                "has_boxed": True,
                "extracted_answer": f"±{abs((pm_resp or pm_exp)[1])}",
                "is_correct": None,
                "comparison_mode": "pm",
                "details": "Extracted ± value; no expected provided."
            }

        # Build the allowed set(s)
        def pm_pair_to_set(pm_pair):
            return {pm_pair[0], pm_pair[1]}

        allowed_exp = None
        if pm_exp is not None:
            allowed_exp = pm_pair_to_set(pm_exp)

        allowed_resp = None
        if pm_resp is not None:
            allowed_resp = pm_pair_to_set(pm_resp)

        # Try to turn the non-± side (if any) into a number
        # response numeric?
        num_resp = first_numeric(tokens)
        # expected numeric?
        num_exp  = first_numeric(expected_tokens) if expected_tokens else None

        ok = False
        if pm_resp is not None and num_exp is not None:
            ok = (abs(float(num_exp) - pm_resp[0]) <= numeric_tolerance) or \
                 (abs(float(num_exp) - pm_resp[1]) <= numeric_tolerance)
        elif pm_exp is not None and num_resp is not None:
            ok = (abs(float(num_resp) - pm_exp[0]) <= numeric_tolerance) or \
                 (abs(float(num_resp) - pm_exp[1]) <= numeric_tolerance)
        elif pm_resp is not None and pm_exp is not None:
            # Any overlap between the two ± sets
            ok = len(pm_pair_to_set(pm_resp).intersection(pm_pair_to_set(pm_exp))) > 0
        else:
            # Could not parse numeric on the non-± side → fall back to string compare
            ok = _normalize_text(inner_clean) == _normalize_text(expected_clean or "")

        return {
            "has_boxed": True,
            "extracted_answer": f"±{abs((pm_resp or pm_exp)[1])}",
            "is_correct": ok,
            "comparison_mode": "pm",
            "details": f"Symmetric ± comparison with tol={numeric_tolerance}"
        }

    # 2) Numeric/fraction symmetry (e.g., '(C) 29' vs '29' should pass)
    num_resp = first_numeric(tokens)
    num_exp  = first_numeric(expected_tokens) if expected_tokens else None

    # --- NEW: detect multi-number answers (lists, intervals) → string compare only
    if _looks_like_list(inner_clean) or (expected_clean and _looks_like_list(expected_clean)):
        return {
            "has_boxed": True,
            "extracted_answer": _normalize_text(inner_clean),
            "is_correct": _normalize_text(inner_clean) == _normalize_text(expected_clean or ""),
            "comparison_mode": "string",
            "details": "Detected multiple numbers (list/interval) → treating as string."
        }
    if num_resp is not None or num_exp is not None:
        extracted = _normalize_text(inner_clean)
        if expected is None:
            return {
                "has_boxed": True,
                "extracted_answer": extracted,
                "is_correct": None,
                "comparison_mode": "numeric",
                "details": f"Parsed numeric={float(num_resp) if num_resp is not None else 'N/A'}"
            }

        if num_resp is not None and num_exp is not None:
            diff = abs(float(num_resp) - float(num_exp))
            return {
                "has_boxed": True,
                "extracted_answer": extracted,
                "is_correct": diff <= numeric_tolerance,
                "comparison_mode": "numeric",
                "details": f"Numeric diff={diff}, tol={numeric_tolerance}"
            }

        # If only one side numeric → try last‑resort numeric parse on the other side’s cleaned string
        if num_resp is not None and expected_clean:
            exp_frac = _to_fraction_if_possible(expected_clean.strip())
            if exp_frac is not None:
                diff = abs(float(num_resp) - float(exp_frac))
                return {
                    "has_boxed": True,
                    "extracted_answer": extracted,
                    "is_correct": diff <= numeric_tolerance,
                    "comparison_mode": "numeric",
                    "details": f"Numeric diff={diff}, tol={numeric_tolerance}"
                }
        if num_exp is not None:
            resp_frac = _to_fraction_if_possible(inner_clean.strip())
            if resp_frac is not None:
                diff = abs(float(resp_frac) - float(num_exp))
                return {
                    "has_boxed": True,
                    "extracted_answer": _normalize_text(inner_clean),
                    "is_correct": diff <= numeric_tolerance,
                    "comparison_mode": "numeric",
                    "details": f"Numeric diff={diff}, tol={numeric_tolerance}"
                }

        # Otherwise revert to string compare on cleaned inners
        return {
            "has_boxed": True,
            "extracted_answer": _normalize_text(inner_clean),
            "is_correct": _normalize_text(inner_clean) == _normalize_text(expected_clean or ""),
            "comparison_mode": "string",
            "details": "Only one side numeric; fell back to normalized string compare."
        }

    # 3) Letter choice symmetry
    letter_resp = _first_letter_choice(tokens)
    if letter_resp is not None:
        if expected is None:
            return {
                "has_boxed": True,
                "extracted_answer": letter_resp,
                "is_correct": None,
                "comparison_mode": "choice",
                "details": "Extracted letter choice."
            }
        expected_letter = _first_letter_choice(expected_tokens) if expected_tokens else None
        if expected_letter is not None:
            is_equal = _normalize_text(letter_resp) == _normalize_text(expected_letter)
        else:
            is_equal = _normalize_text(letter_resp) == _normalize_text(expected_clean or "")
        return {
            "has_boxed": True,
            "extracted_answer": letter_resp,
            "is_correct": is_equal,
            "comparison_mode": "choice",
            "details": "Compared letter choices."
        }

    # 4) Fallback: normalized string compare
    extracted = _normalize_text(inner_clean)
    if expected is None:
        return {
            "has_boxed": True,
            "extracted_answer": extracted,
            "is_correct": None,
            "comparison_mode": None,
            "details": "Non-numeric/choice expression."
        }
    return {
        "has_boxed": True,
        "extracted_answer": extracted,
        "is_correct": extracted == _normalize_text(expected_clean or ""),
        "comparison_mode": "string",
        "details": "Compared normalized strings."
    }


if __name__ == "__main__":
    # pass_response = r"$\boxed{\pm3}$"
    # print(evaluate_teacher_response(response=pass_response, expected="±3"))
    # debug = r"\boxed{\frac{1}{2}}"
    # print(evaluate_teacher_response(response=debug, expected=r"\boxed{0.5}"))
    pass
    