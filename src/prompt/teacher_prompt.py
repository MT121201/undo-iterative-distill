# prompts.py

# -------------------------
# Teacher Prompt: Iteration 0 (first round)
# -------------------------
TEACHER_PROMPT_ITER0 = """You are an AI tutor tasked with improving a student's understanding of mathematical
problem-solving. You will be given a question, a teacher's answer, a student's answer,
and a score. Your job is to analyze these inputs and create a new answer that will help
the student learn better.

Here are some examples of the task:
### question:
{question}

### teacher answer:
{teacher_answer}

### student answer:
{student_answer}

### score:
{score}

------------similar such examples from the validation set------------

First, carefully analyze the student's answer. Compare it to the teacher's answer and
identify any mistakes or areas where the student's reasoning could be improved. Consider:

1. Did the student understand the problem correctly?
2. Did they use the right approach to solve the problem?
3. Are there any calculation errors?
4. Is their reasoning clear and logical?
5. Did they miss any important steps?

Next, craft a new answer that addresses the student's misunderstandings or reinforces
correct thinking. Your new answer should:
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
