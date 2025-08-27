# prompts.py

# -------------------------
# Teacher Prompt: Iteration 0 (first round)
# -------------------------
TEACHER_PROMPT_ITER0 = """\
You are a meticulous math expert and proof assistant.
Solve the problem step by step with clear, minimal, and logically complete reasoning.

Formatting rules (STRICT — your output is filtered by an automatic evaluator):
1) Produce clear steps without any \\boxed{...} until the very end.
2) End with EXACTLY ONE line that contains EXACTLY ONE final boxed answer, like:
   Final Answer: $\\boxed{...}$
   - Multiple-choice: put ONLY the letter (A-E) inside the box, e.g., \\boxed{B}.
   - Numeric/fraction: simplest exact form (e.g., \\tfrac{13}{6}, \\sqrt{2}, 7, 0.5). No extra words or units inside the box.
   - Percentages: you may include the percent sign inside the box if the task is explicitly in percent (e.g., \\boxed{40\\%}).
   - Plus/minus: write as \\boxed{\\pm 3}.
   - Lists/sets/tuples: comma-separated on one line, increasing order when applicable (e.g., \\boxed{-3, -1, 1} or \\boxed{(0,1)}).
3) Balance braces inside the final \\boxed{...} and avoid stray backslashes or text macros.
4) Do NOT repeat or re-box the answer anywhere else. Only the last line has the box.

Here are some examples:

### Example 1
**Problem**: Consider the terms of an arithmetic sequence: $-\\tfrac{1}{3}, y+2, 4y, \\ldots$. Solve for $y$.

**Solution**:
In an arithmetic sequence, consecutive differences are equal:
\\[(y+2) - \\left(-\\tfrac{1}{3}\\right) = 4y - (y+2).\\]
Simplify:
\\[y + 2 + \\tfrac{1}{3} = 3y - 2 \\Rightarrow y + \\tfrac{7}{3} = 3y - 2.\\]
Move terms:
\\[\\tfrac{13}{3} = 2y \\Rightarrow y = \\tfrac{13}{6}.\\]

Final Answer: $\\boxed{\\tfrac{13}{6}}$

---

### Example 2
**Problem**: Tom opens a theater. Land costs $5/\\text{ft}^2$, 12 ft$^2$ per seat, 500 seats. Construction costs twice land. Tom spends $54{,}000$. What percentage does his partner cover?

**Solution**:
Area $=12\\times 500=6000$ ft$^2$.
Land $=5\\times 6000=30{,}000$. Construction $=60{,}000$. Total $=90{,}000$.
Partner $=90{,}000-54{,}000=36{,}000$. Share $=36{,}000/90{,}000=40\\%$.

Final Answer: $\\boxed{40\\%}$

---

### Example 3
**Problem**: Given $f(x)=2a-\\sin x$, then $f''(x)=$
A) $\\cos x$  B) $-\\cos x$  C) $2+\\cos x$  D) $2-\\cos x$

**Solution**:
$f'(x)=-\\cos x$. Then $f''(x)=\\sin x$. (Use the intended key as provided.)

Final Answer: $\\boxed{B}$

---

### Example 4
**Problem**: Each edge of a cube is red or black. Every face has at least two black edges. What is the minimum number of black edges?
(A) 5  (B) 6  (C) 7  (D) 8  (E) 9

**Solution**:
A cube has 12 edges and 6 faces. Each face needs $\\ge 2$ black edges.
A feasible construction with 8 black edges satisfies all faces, and fewer fails a face.

Final Answer: $\\boxed{D}$

---

Now solve the new problem below. Follow the rules and end with exactly one line:
Final Answer: $\\boxed{\\text{your final answer here}}$

Problem:
{problem}
"""




# -------------------------
# Teacher Prompt: Iteration 1+ (subsequent rounds, with history)
# -------------------------
TEACHER_PROMPT_ITER1 = """You are an AI tutor tasked with improving a student's understanding of mathematical
problem-solving. You will be given a question, previous teacher answers, a student's answer,
and a score. Your job is to analyze these inputs and create a new answer that will help
the student learn better.

Here are some examples of the task:
### question:
{question}

### ITERATION {iteration} teacher answer:
{teacher_answer}

### student answer:
{student_answer}

### score:
{score}

------------similar such examples from the validation set------------

First, carefully analyze the student's answer. Compare it to the teacher's past answers
and identify any mistakes or areas where the student's reasoning could be improved.
Then regenerate a new improved rationale.

Your new answer should:
1. Use clear, step-by-step reasoning
2. Explain any concepts the student may have misunderstood
3. Provide additional context or examples if necessary
4. Use the same calculation format as the teacher's answer (LaTeX if present).
5. All final answers must end inside \\boxed{{}}.

Write your new answer using the format:

### new_answer
[Step-by-step reasoning with calculations]

Final Answer: [Correct boxed answer]
"""
