"""
Teacher prompt templates for all pipeline stages.

Iteration 0:   build_teacher_prompt_iter0        — no context, pure problem solving
Iteration K≥1: build_teacher_prompt_iterK        — UNDO context (val signals + prev responses)
Hint stage 2:  build_teacher_prompt_answer_hint  — add GT final answer as hint
Hint stage 3:  build_teacher_prompt_solution_hint — add full GT solution as hint

Each builder shares the same compact, equation-first system prompt and differs only
in how much supervision is injected into the user message. This graduated design is
the key mechanism of Contribution 1 (progressive hint cascade): the teacher sees the
minimum hint necessary to produce a correct response, preserving internal reasoning.
"""

from typing import Dict, List, Optional

# ── Shared system prompt ───────────────────────────────────────────────────────

_SYSTEM = """You are a meticulous math expert and proof assistant.
Write compact, equation-first solutions that look like the user's examples.

STRICT formatting (auto-evaluated):
1) Keep reasoning short:
   - Routine/algebra/arithmetic: ≤3 concise lines before the final line.
   - Piecewise/monotonicity/logic: ≤5 concise lines before the final line.
2) No restating the problem or listing givens.
3) Prefer symbol-heavy lines (equalities/implications). Combine trivial steps.
4) No \\boxed{...} until the very end.
5) End with EXACTLY ONE line with EXACTLY ONE boxed answer:
   Final Answer: $\\boxed{...}$
   - Multiple-choice: only the letter (A–E) inside the box.
   - Numeric/fraction: simplest exact form (e.g., \\tfrac{13}{6}, \\sqrt{2}, 7, 0.5).
   - Percentages: include % only if asked explicitly.
   - ± values: \\boxed{\\pm 3}.
   - Lists/sets/tuples: comma-separated, increasing order when applicable.
6) Balance braces in the final \\boxed{...}. Do not re-box or repeat anywhere else.

Concise style guide:
- Use one computation line for simple sums/products.
- Chain steps with =, ⇒, ⟹ when clear; avoid prose.
- For piecewise/monotone: state each condition in one line; intersect once.
- Choose the shortest correct method.

Mini examples (style + brevity):

Ex 1 (arithmetic seq):
(y+2)−(−\\tfrac13)=4y−(y+2) ⇒ y+\\tfrac73=3y−2 ⇒ \\tfrac{13}{3}=2y ⇒ y=\\tfrac{13}{6}
Final Answer: $\\boxed{\\tfrac{13}{6}}$

Ex 2 (percent share):
Area=12·500=6000; Land=5·6000=30000; Build=60000; Total=90000; Partner=90000−54000=36000 ⇒ 36000/90000=40\\%
Final Answer: $\\boxed{40\\%}$

Ex 3 (MC derivatives):
f'(x)=−\\cos x, f''(x)=\\sin x ⇒ matches choice B
Final Answer: $\\boxed{B}$

Ex 4 (cube edges):
Each face ≥2 black; feasible with 8; any fewer fails some face
Final Answer: $\\boxed{D}$

Ex (piecewise decreasing):
x<0: a^x ↓ ⇒ 0<a<1; x≥0: slope (\\tfrac14−a)<0 ⇒ a>\\tfrac14; at 0: f(0^-)=1>f(0^+)=2a ⇒ a<\\tfrac12 ⇒ a∈(\\tfrac14,\\tfrac12)
Final Answer: $\\boxed{\\left(\\tfrac14,\\tfrac12\\right)}$"""


# ── Helper: format val error signals ─────────────────────────────────────────

def _format_val_error_signals(val_error_signals: List[Dict]) -> str:
    """
    Format the student's validation performance for inclusion in teacher context.

    Each signal dict: {question, student_response, score}
    The score is either binary {0, 1} (UNDO baseline) or continuous S ∈ [−1, 1]
    from our Reasoning-Aware Error Signal (Contribution 2, see scoring.py).
    """
    if not val_error_signals:
        return "(no validation data available)"
    lines = []
    for i, sig in enumerate(val_error_signals, 1):
        score = sig.get("score", sig.get("is_correct", 0))
        score_str = f"{score:+.2f}" if isinstance(score, float) else str(int(score))
        lines.append(
            f"[Val Q{i}] {sig['question'].strip()}\n"
            f"  Student: {sig['student_response'].strip()}\n"
            f"  Score: {score_str}"
        )
    return "\n\n".join(lines)


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_teacher_prompt_iter0(problem: str, tokenizer) -> str:
    """
    Iteration 0: teacher solves from scratch with no context (UNDO baseline first pass).
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                "Follow the rules strictly and end with exactly one line:\n"
                "Final Answer: $\\boxed{...}$"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_teacher_prompt_iterK(
    problem: str,
    val_error_signals: List[Dict],
    student_response_prev: Optional[str],
    teacher_response_prev: Optional[str],
    tokenizer,
) -> str:
    """
    Iteration K≥1, no hint: teacher receives the full UNDO context.

    The teacher sees the student's validation performance (error signals), its own
    previous attempt at this problem, and the student's previous attempt. This context
    lets the teacher diagnose and target the student's specific weaknesses.
    """
    val_block     = _format_val_error_signals(val_error_signals)
    student_prev  = student_response_prev or "(not available — first iteration)"
    teacher_prev  = teacher_response_prev or "(not available — first iteration)"

    user_content = (
        "Your student's current performance on the validation set:\n\n"
        f"{val_block}\n\n"
        "---\n"
        f"Problem:\n{problem}\n\n"
        f"Your previous response to this problem:\n{teacher_prev}\n\n"
        f"Student's previous response to this problem:\n{student_prev}\n\n"
        "Analyse the student's weaknesses shown above, then write a new, improved solution "
        "that will help the student learn better. Follow the formatting rules strictly.\n"
        "Final Answer: $\\boxed{...}$"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_teacher_prompt_answer_hint(
    problem: str,
    gt_answer: str,
    val_error_signals: List[Dict],
    student_response_prev: Optional[str],
    teacher_response_prev: Optional[str],
    tokenizer,
) -> str:
    """
    Progressive hint stage 2: teacher receives only the correct final answer.

    Called when the no-hint attempt (stage 1) failed. Providing only the final
    answer — not the full solution — is intentional: it leaves the reasoning chain
    entirely to the teacher. Pilot study (similarity.py) confirms that answer-hint
    samples retain lower BERTScore F1 than solution-hint samples, meaning the teacher
    still reasons independently rather than paraphrasing.
    """
    val_block    = _format_val_error_signals(val_error_signals)
    student_prev = student_response_prev or "(not available)"
    teacher_prev = teacher_response_prev or "(not available)"

    user_content = (
        "Your student's current performance on the validation set:\n\n"
        f"{val_block}\n\n"
        "---\n"
        f"Problem:\n{problem}\n\n"
        f"Your previous response to this problem:\n{teacher_prev}\n\n"
        f"Student's previous response to this problem:\n{student_prev}\n\n"
        f"[HINT — Correct Final Answer]: {gt_answer}\n\n"
        "Your previous attempt was incorrect. Using the correct final answer above as a guide, "
        "rethink your reasoning from scratch. Do not simply restate the answer — construct a "
        "clear chain-of-thought that leads the student to understand how to arrive at it.\n"
        "Final Answer: $\\boxed{...}$"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_teacher_prompt_solution_hint(
    problem: str,
    gt_solution: str,
    val_error_signals: List[Dict],
    student_response_prev: Optional[str],
    teacher_response_prev: Optional[str],
    tokenizer,
) -> str:
    """
    Progressive hint stage 3: teacher receives the complete GT chain-of-thought.

    Used only when both no-hint and answer-hint attempts failed. This is the strongest
    hint and empirically shifts BERTScore F1 toward the high-similarity (H) region,
    meaning the teacher partially paraphrases the solution. Samples recovered here
    are labelled hint_stage='solution' in output JSONL for downstream analysis.
    """
    val_block    = _format_val_error_signals(val_error_signals)
    student_prev = student_response_prev or "(not available)"
    teacher_prev = teacher_response_prev or "(not available)"

    user_content = (
        "Your student's current performance on the validation set:\n\n"
        f"{val_block}\n\n"
        "---\n"
        f"Problem:\n{problem}\n\n"
        f"Your previous response to this problem:\n{teacher_prev}\n\n"
        f"Student's previous response to this problem:\n{student_prev}\n\n"
        f"[HINT — Complete Solution]:\n{gt_solution}\n\n"
        "Both previous attempts were incorrect. Use the complete solution above as a reference. "
        "Write an explanation that conveys the key reasoning steps clearly for the student. "
        "Ensure correctness and maintain the formatting rules.\n"
        "Final Answer: $\\boxed{...}$"
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
