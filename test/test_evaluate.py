import pytest
from fractions import Fraction

from src.evaluate import (
    _find_last_boxed,
    _strip_wrappers,
    _normalize_text,
    _latex_frac_to_fraction,
    _strip_common_numerical_decorations,
    _to_fraction_if_possible,
    _tokenize_box_candidates,
    _parse_pm_number,
    _first_letter_choice,
    _first_numeric_fraction,
    _first_pm,
    evaluate_teacher_response,
)

# -----------------------------
# _find_last_boxed
# -----------------------------
@pytest.mark.parametrize("text,expected", [
    (r"foo \boxed{42}", "42"),
    (r"foo \boxed{bar \boxed{99}}", "99"),
    (r"no box here", None),
    (r"\boxed{a} and \boxed{b}", "b"),
    (r"\boxed{(C) 29}", "(C) 29"),
    (r"\boxed{unbalanced", None),
    (r"\boxed{outer \boxed{inner}}", "inner"),
    (r"before \boxed{a\boxed{b}} after", "b"),
])
def test_find_last_boxed(text, expected):
    assert _find_last_boxed(text) == expected

# -----------------------------
# _strip_wrappers
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    ("$42$", "42"),
    (r"\(42\)", "42"),
    (r"\textbf{42}", "42"),
    (r"\mathrm{42}", "42"),
    (r"\text{42}", "42"),
    (" (A) ", "A"),
    ("[B]", "B"),
    ("'C'", "'C'"),
    ("42.", "42"),
    ("42;", "42"),
    (" 42 ", "42"),
    ("(D).", "D"),
    ("[E]:", "E"),
])
def test_strip_wrappers(s, expected):
    assert _strip_wrappers(s) == expected

# -----------------------------
# _normalize_text
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    ("  foo   bar ", "foo bar"),
    ("\nfoo\tbar\n", "foo bar"),
    ("foo   bar   baz", "foo bar baz"),
])
def test_normalize_text(s, expected):
    assert _normalize_text(s) == expected

# -----------------------------
# _latex_frac_to_fraction
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    (r"\frac{1}{2}", Fraction(1, 2)),
    (r"\frac{-3}{4}", Fraction(-3, 4)),
    (r"\frac{5}{0}", None),
    (r"\frac{a}{b}", None),
    ("not a frac", None),
])
def test_latex_frac_to_fraction(s, expected):
    assert _latex_frac_to_fraction(s) == expected

# -----------------------------
# _strip_common_numerical_decorations
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    ("$1,000", "1000"),
    ("60^{\\circ}", "60"),
    ("45 degrees", "45"),
    ("90°", "90"),
    ("100%", "100"),
    ("$ 42", "42"),
    ("1\\:000", "1000"),
])
def test_strip_common_numerical_decorations(s, expected):
    assert _strip_common_numerical_decorations(s) == expected

# -----------------------------
# _to_fraction_if_possible
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    ("42", Fraction(42)),
    ("-3.5", Fraction("-3.5")),
    ("1/2", Fraction(1, 2)),
    (r"\frac{3}{4}", Fraction(3, 4)),
    ("0.25", Fraction("0.25")),
    ("not a number", None),
    ("1/0", None),
    (r"\frac{1}{0}", None),
])
def test_to_fraction_if_possible(s, expected):
    assert _to_fraction_if_possible(s) == expected

# -----------------------------
# _tokenize_box_candidates
# -----------------------------
@pytest.mark.parametrize("s,expected", [
    ("42", ["42"]),
    ("(C) 29", ["C", "29"]),
    (r"\textbf{(D)}", ["D"]),
    ("①", ["1"]),
    ("② ③", ["2", "3"]),
    ("A", ["A"]),
    ("(B)", ["B"]),
    ("", []),
])
def test_tokenize_box_candidates(s, expected):
    assert _tokenize_box_candidates(s) == expected

# -----------------------------
# _parse_pm_number
# -----------------------------
@pytest.mark.parametrize("t,expected", [
    ("±3", (-3.0, 3.0)),
    (r"\pm 2.5", (-2.5, 2.5)),
    ("+3", None),
    ("3", None),
    ("pm 3", None),
])
def test_parse_pm_number(t, expected):
    assert _parse_pm_number(t) == expected

# -----------------------------
# _first_letter_choice
# -----------------------------
@pytest.mark.parametrize("tokens,expected", [
    (["A", "42"], "A"),
    (["42", "B"], "B"),
    (["foo", "bar"], None),
    (["C"], "C"),
])
def test_first_letter_choice(tokens, expected):
    assert _first_letter_choice(tokens) == expected

# -----------------------------
# _first_numeric_fraction
# -----------------------------
@pytest.mark.parametrize("tokens,expected", [
    (["foo", "1/2", "bar"], Fraction(1, 2)),
    (["not", "a", "number"], None),
    (["42"], Fraction(42)),
    ([r"\frac{3}{4}"], Fraction(3, 4)),
])
def test_first_numeric_fraction(tokens, expected):
    assert _first_numeric_fraction(tokens) == expected

# -----------------------------
# _first_pm
# -----------------------------
@pytest.mark.parametrize("tokens,expected", [
    (["foo", "±3", "bar"], (-3.0, 3.0)),
    ([r"\pm 2"], (-2.0, 2.0)),
    (["42"], None),
])
def test_first_pm(tokens, expected):
    assert _first_pm(tokens) == expected

# -----------------------------
# evaluate_teacher_response
# -----------------------------
def test_evaluate_teacher_response_numeric():
    resp = r"Here is the answer: \boxed{42}"
    exp = r"\boxed{42}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "numeric"

def test_evaluate_teacher_response_numeric_fraction():
    resp = r"\boxed{\frac{1}{2}}"
    exp = r"\boxed{0.5}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True

def test_evaluate_teacher_response_letter_choice():
    resp = r"\boxed{(C)}"
    exp = r"\boxed{C}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "choice"

def test_evaluate_teacher_response_pm():
    resp = r"\boxed{±3}"
    exp = r"\boxed{3}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "pm"

def test_evaluate_teacher_response_no_boxed():
    resp = "No box here"
    result = evaluate_teacher_response(resp)
    assert result["has_boxed"] is False
    assert result["extracted_answer"] is None

def test_evaluate_teacher_response_string_fallback():
    resp = r"\boxed{foo bar}"
    exp = r"\boxed{foo bar}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "string"

def test_evaluate_teacher_response_circled():
    resp = r"\boxed{④}"
    exp = r"\boxed{4}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"]
    assert result["is_correct"]

def test_evaluate_teacher_response_currency():
    resp = r"\boxed{\$280}"
    exp = r"\boxed{280}"
    result = evaluate_teacher_response(resp, exp)
    assert result["is_correct"]
    assert result["comparison_mode"] == "numeric"

def test_evaluate_teacher_response_pm_reverse():
    resp = r"\boxed{3}"
    exp = r"\boxed{±3}"
    result = evaluate_teacher_response(resp, exp)
    assert result["is_correct"]

def test_evaluate_teacher_response_empty_box():
    resp = r"\boxed{}"
    result = evaluate_teacher_response(resp)
    assert result["has_boxed"]
    assert result["extracted_answer"] == ""

def test_evaluate_teacher_response_decimal():
    resp = r"""
    First, calculate the total number of children across all families:
    \[ 9 \times 3 = 27 \text{ children total.} \]

    Given that 3 families are childless, the number of families with children is:
    \[ 9 - 3 = 6 \text{ families with children.} \]

    Now, calculate the average number of children per family with children:
    \[ \frac{27}{6} = 4.5 \]

    Thus, the average number of children in the families with children is $\boxed{4.5}$.
    """
    exp = r"\boxed{4.5}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["extracted_answer"] == "4.5"
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "numeric"

def test_evaluate_teacher_response_factorial_string():
    # Both sides provide factorial symbolically
    resp = r"… therefore x is $\boxed{13!}$."
    exp  = r"\boxed{13!}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["extracted_answer"] in ("13!", r"13!")  # depending on wrappers
    # With both boxed strings equal after normalization
    assert result["is_correct"] is True
    assert result["comparison_mode"] in ( "numeric", "string")  # could be string if you don't coerce both
    # If your tokenizer yields numeric (after factorial patch), it will be numeric.

def test_evaluate_teacher_response_factorial_numeric_equivalence():
    # One side symbolic factorial, other side numeric 13! = 6227020800
    resp = r"\boxed{13!}"
    exp  = r"\boxed{6227020800}"
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["is_correct"] is True
    assert result["comparison_mode"] == "numeric"

def test_evaluate_teacher_response_non_numeric_string_fallback():
    # If expected is a narrative (no boxed / no numeric), we fall back to string compare of cleaned inners
    resp = r"\boxed{13!}"
    exp  = r"The value is \boxed{13!} given the setup."
    result = evaluate_teacher_response(resp, exp)
    assert result["has_boxed"] is True
    assert result["comparison_mode"] in ("numeric", "string")
    assert result["is_correct"] is True


def test_choice_simple_B():
    resp = r"The ratio of height to base ... The final answer is $\boxed{B}$."
    exp  = r"\boxed{B}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "choice"

def test_empty_box_inside_display_math():
    resp = r"""
    \[
    \boxed{}
    \]
    """
    out = evaluate_teacher_response(resp)  # no expected
    assert out["has_boxed"] is True
    # Empty box -> extracted_answer should be empty string after cleaning
    assert out["extracted_answer"] in ("",)  # tolerate empty
    assert out["is_correct"] is None

def test_interval_string_compare():
    resp = r"\[ x \in \boxed{(0.25, 0.5)} \]"
    exp  = r"\boxed{(0.25, 0.5)}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "string"

def test_list_of_numbers_string_compare():
    resp = r"This equation ... Therefore ... $\boxed{-3, -1, 1}$."
    exp  = r"\boxed{-3, -1, 1}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    # Lists aren't parsed as a single numeric -> should fall back to string
    assert out["comparison_mode"] == "string"

def test_choice_with_textbf_and_surrounding_math():
    resp = r"- The correct length of \(BC\) is \(2\sqrt{43}\). The final answer is $\boxed{\(\textbf{(B)}\ 2\sqrt{43}\)}$"
    exp  = r"\boxed{B}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "choice"

def test_pm_against_positive_value():
    resp = r"Therefore, the answer is: $\boxed{\pm3}$."
    exp  = r"\boxed{3}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "pm"

def test_numeric_choice_combo_13():
    resp = r"""
    6. **Conclude with the answer**:
    \[
    13
    \]
    The final answer is $\boxed{\textbf{(D)} \: 13}$
    """
    exp  = r"\boxed{13}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    # Our evaluator prioritizes numeric over choice when both exist
    assert out["comparison_mode"] == "numeric"

def test_degrees_cleaning():
    resp = r"So, the values are $B=\boxed{60^{\circ}}$, $C=\boxed{90^{\circ}}$, and $a=\boxed{\sqrt{3}}$."
    # Check degree normalization -> numeric compare for 60 and 90; sqrt(3) remains string
    outB = evaluate_teacher_response(r"$\boxed{60^{\circ}}$", r"\boxed{60}")
    assert outB["has_boxed"] is True and outB["is_correct"] is True and outB["comparison_mode"] == "numeric"

    outC = evaluate_teacher_response(r"$\boxed{90^{\circ}}$", r"\boxed{90}")
    assert outC["has_boxed"] is True and outC["is_correct"] is True and outC["comparison_mode"] == "numeric"

    outa = evaluate_teacher_response(r"$\boxed{\sqrt{3}}$", r"\boxed{\sqrt{3}}")
    assert outa["has_boxed"] is True and outa["is_correct"] is True and outa["comparison_mode"] == "string"

def test_circled_numeral_choice():
    resp = r"In summary, only $④$ is correct. Hence, the answer is: $\boxed{④}$."
    exp  = r"\boxed{4}"
    out = evaluate_teacher_response(resp, exp)
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    # mapped to '4' then parsed as numeric
    assert out["comparison_mode"] in ("numeric", "string")

def _wrap(resp: str) -> str:
    # Helper to test each \boxed in isolation (what the model would “return”)
    return resp

def test_choice_simple_B_split():
    resp = _wrap(r"The final answer is $\boxed{B}$.")
    out = evaluate_teacher_response(resp, r"\boxed{B}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "choice"

def test_empty_box_split():
    resp = _wrap(r"\[\boxed{}\]")
    out = evaluate_teacher_response(resp)  # no expected
    assert out["has_boxed"] is True
    assert out["extracted_answer"] in ("",)
    assert out["is_correct"] is None

def test_interval_string_compare_split():
    resp = _wrap(r"\[ x \in \boxed{(0.25, 0.5)} \]")
    out  = evaluate_teacher_response(resp, r"\boxed{(0.25, 0.5)}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "string"

def test_list_of_numbers_string_compare_split():
    resp = _wrap(r"This … $\boxed{-3, -1, 1}$.")
    out  = evaluate_teacher_response(resp, r"\boxed{-3, -1, 1}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "string"

def test_choice_with_textbf_and_surrounding_math_split():
    resp = _wrap(r"The final answer is $\boxed{\(\textbf{(B)}\ 2\sqrt{43}\)}$")
    out  = evaluate_teacher_response(resp, r"\boxed{B}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "choice"

def test_pm_against_positive_value_split():
    resp = _wrap(r"Therefore, the answer is: $\boxed{\pm3}$.")
    out  = evaluate_teacher_response(resp, r"\boxed{3}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "pm"

def test_numeric_choice_combo_13_split():
    resp = _wrap(r"The final answer is $\boxed{\textbf{(D)} \: 13}$")
    out  = evaluate_teacher_response(resp, r"\boxed{13}")
    assert out["has_boxed"] is True
    assert out["is_correct"] is True
    # numeric should win if both numeric and choice present
    assert out["comparison_mode"] == "numeric"

def test_degrees_cleaning_60_split():
    resp = _wrap(r"$\boxed{60^{\circ}}$")
    out  = evaluate_teacher_response(resp, r"\boxed{60}")
    assert out["has_boxed"] and out["is_correct"] and out["comparison_mode"] == "numeric"

def test_degrees_cleaning_90_split():
    resp = _wrap(r"$\boxed{90^{\circ}}$")
    out  = evaluate_teacher_response(resp, r"\boxed{90}")
    assert out["has_boxed"] and out["is_correct"] and out["comparison_mode"] == "numeric"

def test_sqrt3_string_split():
    resp = _wrap(r"$\boxed{\sqrt{3}}$")
    out  = evaluate_teacher_response(resp, r"\boxed{\sqrt{3}}")
    assert out["has_boxed"] and out["is_correct"] and out["comparison_mode"] == "string"

def test_circled_numeral_split():
    resp = _wrap(r"Hence, the answer is: $\boxed{④}$.")
    out  = evaluate_teacher_response(resp, r"\boxed{4}")
    assert out["has_boxed"] and out["is_correct"]
    assert out["comparison_mode"] in ("numeric", "string")  # either acceptable

def test_final_primes_list_split():
    resp = _wrap(r"\[\boxed{2, 3, 5, 7, 13}\]")
    out  = evaluate_teacher_response(resp, r"\boxed{2, 3, 5, 7, 13}")
    assert out["has_boxed"] is True
    assert out["extracted_answer"] == "2, 3, 5, 7, 13"
    assert out["is_correct"] is True
    assert out["comparison_mode"] == "string"