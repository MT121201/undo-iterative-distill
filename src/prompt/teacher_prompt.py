# prompts.py

# -------------------------
# Teacher Prompt: Iteration 0 (first round)
# -------------------------
from transformers import AutoTokenizer

def build_teacher_prompt_iter0(problem_text: str, tokenizer):
    messages = [
        {
            "role": "system",
            "content": """You are a meticulous math expert and proof assistant.
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
Final Answer: $\\boxed{\\left(\\tfrac14,\\tfrac12\\right)}$
"""
        },
        {
            "role": "user",
            "content": f"""Problem:
{problem_text}

Follow the rules strictly and end with exactly one line:
Final Answer: $\\boxed{{...}}$"""
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )





# -------------------------
# Teacher Prompt: Iteration 1+ (subsequent rounds, with history)
# -------------------------
