# Claude Code Direction: Edit F_Hinted_UNDO PPTX

## Overview

This document gives **exact, step-by-step instructions** for a Claude Code agent to edit the uploaded `F_Hinted_UNDO_pptx__1_.pdf` (source reference) and produce a polished `.pptx` file. The presentation is an academic research deck titled **"Recovering Incorrect Teacher Attempts via Answer- and Solution-Level Hints"** by MT Kieu, March 16, 2026.

---

## 0. Environment Setup

```bash
# Install all required dependencies first
pip install "markitdown[pptx]" Pillow defusedxml --break-system-packages
npm install -g pptxgenjs

# Confirm LibreOffice and poppler are available
which soffice || echo "soffice missing"
which pdftoppm || echo "pdftoppm missing"
```

---

## 1. Locate & Copy the Source File

The uploaded file is a PDF preview of the original PPTX. Since no `.pptx` source is available, **create the presentation from scratch using PptxGenJS**. The PDF gives the full content reference.

```bash
# Working directory
mkdir -p /home/claude/undo_pptx
cd /home/claude/undo_pptx
```

---

## 2. Design Specification

### Color Palette (match the original maroon/dark-red academic theme)

| Role | Hex | Usage |
|---|---|---|
| Primary (maroon) | `#7B1D1D` | Slide numbers, headers, accent borders |
| Dark (near-black) | `#1A1A1A` | Title slide background, section dividers |
| Light background | `#FFFFFF` | Content slide backgrounds |
| Accent red | `#C0392B` | Bold inline highlights |
| Subtle gray | `#F5F5F5` | Card backgrounds, secondary panels |
| Text dark | `#2C2C2C` | Body text |
| Gold/orange accent | `#E67E22` | Chart bar (A2/Progressive Hints bars) |
| Blue accent | `#2980B9` | Chart bar (A1/Always Solution bars) |

### Typography

| Element | Font | Size | Weight |
|---|---|---|---|
| Slide title | `Calibri` | 36pt | Bold |
| Section number badge | `Calibri` | 18pt | Bold |
| Body text | `Calibri` | 16pt | Regular |
| Sub-bullets | `Calibri` | 14pt | Regular |
| Caption / footnote | `Calibri` | 11pt | Regular, muted |
| Slide number (footer) | `Calibri` | 10pt | Regular |

### Layout Rules

- Slide dimensions: **10 × 7.5 inches** (standard widescreen 4:3 is fine, or 13.33 × 7.5 for 16:9)
- Left margin: **0.5 in**, Right margin: **0.5 in**, Top: **1.2 in** (below title bar), Bottom: **0.4 in**
- Every content slide has: a **maroon left-side badge** with the slide number + the slide title in bold at the top
- A thin **horizontal rule** (`#7B1D1D`, 2pt) separates the title area from content
- Slide page numbers appear bottom-right in muted gray

---

## 3. Complete Slide Inventory

Reproduce **all 22 content slides** (+ 3 section dividers + title + end slide + backup slides = ~28 total). Below is the full mapping.

---

### Slide 1 — Title Slide

**Layout:** Dark background (`#1A1A1A`), centered content.

**Content:**
```
Main title (large, white, bold):
  "Recovering Incorrect Teacher Attempts via
   Answer- and Solution-Level Hints"

Subtitle (white, lighter):
  "Preserving Teacher Reasoning
   with Progressive Hint-Based Distillation"

Date (muted gray): "March 16th, 2026"
Author (white):    "MT Kieu"
```

**Visual:** Add a subtle geometric pattern (thin maroon diagonal lines or a dark gradient overlay) to avoid plain black.

---

### Slide 2 — Table of Contents

**Layout:** Light background. Three numbered sections in large text.

**Content:**
```
1. INTRODUCTION
2. EXPERIMENTAL SCENARIOS
3. DISCUSSION & CONCLUSION
```

**Visual:** Three large numbered blocks in maroon circles (diameter ~0.8 in), each next to its section label. Use a 3-row vertical stack with generous spacing.

---

### Slide 3 — Section Divider: "1. INTRODUCTION"

**Layout:** Full dark maroon background (`#7B1D1D`). Centered large white text.

**Content:**
```
"1. INTRODUCTION"
```

---

### Slide 4 — Background: UNDO (Slide number badge: 1)

**Layout:** Two-column. Left = text bullets. Right = flowchart diagram.

**Left column content:**
```
Header: "What is UNDO?"

Bullet 1 (bold maroon label):
  "UNDO" is a "knowledge distillation method"
  where a teacher model generates reasoning
  traces to train a student model.

Bullet 2:
  UNDO is currently available as a "preprint",
  and "official code is not released".

(Highlighted phrases: knowledge distillation method,
 preprint, official code is not released — use maroon bold)
```

**Right column content — Flowchart diagram (draw with PptxGenJS shapes):**

```
Nodes and positions (approximate, left-to-right / top-to-bottom):

[Question]        (blue fill #AED6F1, top-left)
[GT Answer]       (green fill #A9DFBF, top-right)
[Teacher Response (with COT)] (blue fill, middle-left)
[Evaluate]        (yellow diamond #F9E79F, middle-center)
  → CORRECT label (green) → [Student Learning] (pink fill)
  → WRONG label (red) → [Drop] (dashed red box)
Dashed red arrow loop from Drop back to Teacher Response
Label at bottom: "Feedback learning gaps"

Caption below diagram: "Simplify UNDO method"
```

Use `addShape`, `addText`, and `addLine` in PptxGenJS to build this. Key connections:
- Solid black arrows: Question→Teacher, GT Answer→Evaluate, Teacher→Evaluate, Evaluate→Student Learning (CORRECT path), Evaluate→Drop (WRONG path)
- Dashed red arrow: Drop → back to Teacher Response (loop)

---

### Slide 5 — Reproducing UNDO (Badge: 2)

**Layout:** Two-column. Left = text. Right = 2×2 grid of line charts.

**Left column:**
```
"We reproduced UNDO" following the preprint's
algorithm and training settings, however we
used the "small-scale training set".

Results:
• Our reproduced UNDO model shows "highly
  consistent performance trends" with the results
  reported in the UNDO paper.
  – Performance "improvement" up to the 3rd
    iteration then degradation

→ This confirms that our reproduction "follows the
UNDO training dynamics", despite using smaller-scale
settings.
```

**Right column — 4 line charts (2×2 grid):**

Each chart: X-axis = Iteration (1–4), Y-axis = Accuracy. Two lines: Paper (blue) and Reproduce (orange).

Chart data:

| Chart | Paper values | Reproduce values | Y-range |
|---|---|---|---|
| GSM8k | 54, 56, 57, 55 | 51, 53, 54.2, 53 | 51–57 |
| MATH | 32.5, 36, 38, 37.5 | 31, 34, 35, 36 | 31–38 |
| MMLU PRO | 14, 16, 15.5, 16 | 12, 14, 14, 14.5 | 12–16 |
| SVAMP | 76, 82, 86, 87 | 70, 80, 85, 87 | 70–88 |

Use PptxGenJS `addChart` with type `line`. Add legend: Paper (blue dot), Reproduce (orange square).

---

### Slide 6 — Motivation: Why Go Beyond UNDO? (Badge: 3)

**Layout:** Single-column with two highlighted callout boxes.

**Content:**
```
Header: "Key observation when reproduce UNDO method"

• UNDO discards ~40% of teacher attempts (incorrect answers)
• These discarded attempts often contain:
    – Partial reasoning
    – Useful intermediate steps
    – Signals about where the teacher is uncertain

[Callout box — red border]:
"Problem with UNDO"
• Data inefficient: only ~60% of training questions are used
• Given solution hints may recover them, but may also cause copying

[Arrow callout at bottom]:
→ To go beyond UNDO, we need the careful mechanism
  to recover failed teacher attempts.
```

**Visual:** Use a maroon-bordered callout rectangle for the "Problem with UNDO" section. Use a right-arrow shape (maroon fill, white text) for the conclusion statement.

---

### Slide 7 — Research Questions and Goals (Badge: 4)

**Layout:** Two-column. Left = Research Questions (numbered). Right = Research Goals (bullets).

**Left column:**
```
"Research questions:"

1. Can we recover incorrect teacher attempts
   by providing additional supervision?

2. Does the teacher actually improve its reasoning,
   or does it simply copy the provided solution?

3. Can we recover incorrect teacher attempts without
   overriding the teacher's own reasoning?
```

**Right column:**
```
"Research goals:"

• Extend UNDO to address its limitation in
  inefficient use of training data

• Demonstrate that preserving teacher internal
  reasoning benefits student learning more than
  overriding it by giving full solution.

• A method to preserve native teacher reasoning
  when it already exists
```

**Visual:** Separate the two columns with a thin vertical maroon line.

---

### Slide 8 — Section Divider: "2. EXPERIMENTAL SCENARIOS"

Same dark maroon full-slide treatment as Slide 3.

---

### Slide 9 — Baseline Workflow (Badge: 5)

**Layout:** Full-slide flowchart diagram with brief description above.

**Title bar:** "Baseline Workflow"

**Diagram — reproduce the full UNDO workflow:**

Top band (validation loop):
```
[Validation Questions] → [Student Inference] → [Val-set error signals box]
                                                  (contains 3 sub-rows:
                                                   Validation Question,
                                                   Student Response,
                                                   Score (0 or 1 for T/F))
                                              → [ALL val-set error signals at K]
```

Bottom band (teacher + distillation loop):
```
[Teacher Context at Iteration K]  (stacked box with fields:
  Question n+2 / n+1 / n
  ALL val-set error signals at K
  Student's Response at K-1 / None if K=1
  Teacher's Response at K-1 / None if K=1)

→ [Question (n)] → [Teacher Prompt box] → [Teacher Inference]
                     (text: "Your job is to analyze these inputs
                      and create a new answer that will help the
                      student learn better.")

Teacher Inference:
  → [Save for K+1]
  → [Evaluate] (diamond)
      CORRECT → [Iteration K Distillation Dataset] → [Student Distillation]
                                                          → Update weight
      WRONG → [Skip in this iteration] (dashed)

[New Student Inference] → [Save for K+1]  "End of iteration K"
```

Use color coding: blue boxes for questions/inputs, yellow diamond for Evaluate, pink for distillation dataset, red dashed for wrong-path items.

---

### Slide 10 — Experimental Setup (Badge: 6)

**Layout:** Two clean content blocks.

**Content:**
```
"Training Dataset"
NuminaMath-CoT: 10K samples (scaled reproduction setting).

"Evaluation Benchmarks"
We evaluate across different difficulty levels:

• Easy:            GSM8K + SVAMP
• Hard:            MATH + MMLU-Pro
• Out-of-Domain:   StrategyQA
  (above all train/test datasets are in same domain: Mathematics)
```

**Visual:** Use three colored horizontal pill/badge labels for Easy (green), Hard (red), Out-of-Domain (purple/blue) next to the benchmark names.

---

### Slide 11 — Diagnostic Analysis: Quantifying Teacher Dependence (Badge: 7)

**Layout:** Two-column. Left = bullets + hypothesis box. Right = small diagram + color gradient bar.

**Left column:**
```
• Metric: "BERTScore F1" (Contextual Embedding Similarity).
• Target: Similarity between Teacher Explanation (E) and
  Ground-Truth Solution (S).
• Purpose: To detect if the teacher model is actually
  "thinking" or just performing sophisticated paraphrasing.
```

**Right column:**

Top: Two boxes side-by-side
```
[GT Solution]    [Teacher Response]
       ↘              ↙
    [BERTScore F1 (similarity score)]
```

Middle: Horizontal gradient bar (green → yellow → red, labeled):
```
"Teacher Reasoning Behavior Spectrum"
Low BERTScore = Self-reasoning ←————→ High BERTScore = Copy solution
```

Bottom: Hypothesis box (bordered rectangle):
```
"Hypothesis: Models trained on low-similarity
(high-reasoning) samples will exhibit superior
out-of-domain robustness compared to those trained
on high-similarity (copy solution) samples."
```

---

### Slide 12 — Pilot Study: How Do Teachers Use Hints? (Badge: 8)

**Layout:** Two-column. Left = text. Right = two side-by-side diagrams.

**Left column:**
```
"RQ2:" Does the teacher "actually improve its reasoning",
or does it simply "copy the provided solution"?

We generate teacher explanations under "two conditions"

❖ "Correct-hint condition:"
  ➢ Provide the "correct solution"

❖ "Wrong-hint condition:"
  ➢ Provide an "incorrect solution or answer"

"Why this test matters"
→ This diagnostic helps us understand "teacher behavior
  when receiving solution."
→ "Does this method always override teacher's reasoning?"
```

**Right column — Two mini flowcharts side-by-side:**

Left mini-chart (Correct condition):
```
[Question] [Correct Solution]
       ↘        ↙
  [Teacher Response]     [GT Solution]
         ↓
  [Similarity Score]
```

Right mini-chart (Wrong condition):
```
[Question] [Wrong Solution]
       ↘        ↙
  [Teacher Response]     [GT Solution]
         ↓
  [Similarity Score]
```

---

### Slide 13 — Pilot Study Results: Teacher Follows Correct Hints, Resists Wrong Hints (Badge: 9)

**Layout:** Two-column. Left = findings text. Right = KDE distribution chart.

**Left column:**
```
"Key observations"

1. "Correct hint → high similarity" (mostly 0.6–0.9)
2. "Wrong hint → low similarity" (mostly 0.1–0.4)

"Interpretation"
• The teacher "follows a correct solution strongly"
• But the teacher "does not blindly copy an incorrect one"

"This suggests that: Given hints do not always overwrite
reasoning — careful hint-strategy can help reserve
teacher reasoning"
```

**Right column — KDE Distribution Chart:**

X-axis: 0.0 to 1.0 (BERTScore F1)
Y-axis: Density 0 to 4

Two distributions:
- Orange histogram + red dashed KDE: `solution_hint (wrong-forced)` — peak around 0.2–0.3
- Blue histogram + green KDE: `solution_hint (correct)` — peak around 0.8–0.9

X-axis label strip at bottom:
```
Self-reasoning ←————————————————→ Copy solution
```

Legend: solution_hint (correct), solution_hint (wrong-forced), KDE (correct), KDE (wrong-forced)

Use `addChart` with type `bar` or render as an SVG image embedded via `addImage`. If generating chart with code, use `matplotlib` to create and embed as PNG.

---

### Slide 14 — Hint-Strategy: From UNDO to Progressive Hinting (Badge: 10)

**Layout:** Header text + 3-column diagram comparison.

**Header:** "Progressive hinting is designed to help reserve teacher reasoning"

**Three columns (equal width):**

Column 1 — "UNDO Baseline":
```
Flowchart:
[Question] → [Teacher Inference] ← [GT Answer]
                    ↓
               [Evaluate] ◇
              CORRECT → [Distillation Dataset]
              WRONG → [Drop]
```

Column 2 — "Always-Solution Hinting":
```
Flowchart:
[Full Solution] [Question]
         ↘         ↙
    [Teacher Inference]
           ↓
      [Evaluate] ◇
      CORRECT → [Distillation Dataset]
      WRONG → [Drop]

Note: "Simplify the evaluation block"
```

Column 3 — "Progressive Hinting (our)":
```
Flowchart:
[Question] → [Teacher] ← CORRECT → [Distillation dataset]
                ↓ WRONG
         [Final Answer] → [Teacher] → CORRECT ↗
                              ↓ WRONG
                     [Full Solution] → [Teacher] → CORRECT ↗
                                            ↓ WRONG
                                         [Drop]

Note: "Simplify the evaluation block"
```

Footer bullets (below all 3 columns):
```
• GT Answer = Final Answer: The correct answer of that question
• Full Solution: The human CoT to solve that question (included the final answer)
```

---

### Slide 15 — Experimental Results (Badge: 11)

**Layout:** 2×2 bar chart grid (left 2/3 of slide) + results summary text (right 1/3).

**Charts — 4 grouped bar charts:**

Each chart has 3 bars per group: UNDO (gray), A1 Always Solution (dark blue), A2 Progressive Hints (orange).

| Chart | UNDO | A1 | A2 |
|---|---|---|---|
| GSM8K | 61.1 | 65.2 | 68.2 |
| SVAMP | 88.6 | 89.1 | 89.6 |
| MATH | 34.6 | 37.1 | 39.8 |
| MMLU-Pro | 14.1 | 15.3 | 15.6 |

Legend: UNDO (gray), A1 Always Solution (dark blue/navy), A2 Progressive Hints (orange)

**Right column text:**
```
"Progressive Hints (A2) consistently outperform
UNDO baseline"

• GSM8K: "+7.0"
• MATH: "+5.2"
• MMLU-Pro: "+1.5"
• SVAMP: "+1.0"

"Always-Solution (A1) helps, but less than A2"
⇒ Indicates that "structured progressive guidance
   matters"

This supports a hypothesis that: "Teacher's internal
reasoning lead to better student adaptation".
```

Caption at bottom-left: "Performance comparison across GSM8K, SVAMP, MATH, and MMLU-Pro"

---

### Slide 16 — Why Progressive Hinting Works (Badge: 12)

**Layout:** Single column, text explanation + 3-group definition boxes.

**Content:**
```
"To understand, first we divide teacher's solutions into three
groups based on BERTScore F1:"

[L box — green border]:
"L — Low similarity (<0.5)"
• Teacher uses mostly its own reasoning
• Minimal copying from the provided solution

[M box — yellow border]:
"M — Mid similarity (0.5–0.75)"
• Teacher is guided by the solution, but not copying
• Mixed teacher reasoning and hint structure

[H box — red border]:
"H — High similarity (>0.75)"
• Teacher largely paraphrases or copies the solution
• Strong dependence on the provided hint

Hypothesis box (bordered):
"If internal reasoning truly helps student adaptation, then
low-similarity samples should improve students more."
```

**Visual:** Three side-by-side colored cards (L=green, M=amber, H=red) with border and title, each containing their bullet points.

---

### Slide 17 — Progressive Hints vs Always-Solution: Similarity Comparison (Badge: 13)

**Layout:** Left = text findings. Right = two stacked KDE distribution charts.

**Left column:**
```
"Always Solution"
• Similarity distribution is "concentrated in M and H"
• Teacher mostly "paraphrases or copies" the provided solution

"Progressive Hinting"
• Similarity distribution shifts "strongly toward L"
• Teacher often generates "independent or partially guided
  reasoning"

"Hypothesis:"
"Teacher's Internal Reasoning Leads to Better Student Adaptation"
```

**Right column — Two KDE plots stacked:**

Top plot (Approach 1 — Always Solution):
- X: 0.0 to 1.0, Y: Density 0–2.0
- Blue histogram + green KDE curve
- Peak around 0.5–0.7 (M/H region)
- Red arrows showing L / M / H boundaries at ~0.5 and ~0.75

Bottom plot (Approach 2 — Progressive Hinting):
- Same axes
- Peak around 0.3–0.4 (L region)
- Same boundary markers

X-axis footer: `Self-reasoning ←——→ Copy solution`

---

### Slide 18 — Verification Setup (Badge: 14)

**Layout:** Two-column. Left = text. Right = 3-row table diagram.

**Left column:**
```
"To test this, we use the same:"
• Original no-hint teacher's response

"We construct 3 sets from hinted teacher's response:"
• "H-set": No-hint response + "High" similarity response
• "L-set": No-hint response + "Low" similarity response
• "L+M-set": No-hint response + "Low + Medium" similarity response

* No-hint responses are the same in 3 sets
* Same number of additional samples
```

**Right column — 3-row visual table:**

```
Row 1: [No-hint (gray)]  [High similarity (red/pink)]
Row 2: [No-hint (gray)]  [Low similarity (green)]
Row 3: [No-hint (gray)]  [Low similarity (green)]  [Medium similarity (yellow)]
```

Draw as colored rectangles arranged in 3 horizontal rows with labels.

---

### Slide 19 — Verification Test Results (Badge: 15)

**Layout:** 4 bar charts in a row (1×4) at top 60% of slide. Findings text at bottom.

**Charts — H set vs L set vs L+M set:**

Each chart has 3 bars: H set (dark blue), L set (orange), L+M set (light orange/yellow).

| Chart | H set | L set | L+M set |
|---|---|---|---|
| GSM8K | 63.1 | 67.4 | 66.8 |
| SVAMP | 88.5 | 89.3 | 89.2 |
| MATH | 31.2 | 37.4 | 36.9 |
| MMLU-Pro | 12.1 | 15.8 | 15.4 |

Legend: H set (dark blue), L set (orange), L+M set (gold)

**Bottom findings text:**
```
- "L set consistently outperforms H set" across all datasets.
  Largest gains appear on hard reasoning (MATH)
- L+M performs close to L, but slightly diluted
- High similarity samples underperform (Always hints cannot prevent this)
- Progressive hints similarity distribution mostly concentrated in Low-similarity
  region — that's why this method performs better than Always hints

⇒ "Hypothesis verified:" "Teacher's Internal Reasoning Leads to
   Better Student Adaptation"
```

---

### Slide 20 — Out-of-Domain Evaluation (Badge: 16)

**Layout:** Left = text findings. Right = single bar chart (StrategyQA).

**Left column:**
```
"Out-of-domain benchmark: StrategyQA"

"Key Observations:"
☐ "Always-hint and H-set outperform UNDO on in-domain
   benchmarks", but perform worse than UNDO on OOD.
     □ This suggests that "solution-copying behavior may
       improve in-domain accuracy but leads to overfitting
       to the math distribution."

☐ "Internal teacher reasoning (L-based variants)"
   maintains strong performance both in-domain and OOD.

⇒ "Internal teacher reasoning improves student
   adaptation and transfers beyond the training domain."
```

**Right column — StrategyQA bar chart:**

6 bars: UNDO* (gray), Approach 1 (dark navy), A2-Full (orange), H set (blue), L set (dark orange/red), L+M set (light orange)

Values (approximate from slide):
```
UNDO*:      13.4
Approach 1: 12.4
A2-Full:    14.6
H set:      11.6
L set:      14.3
L+M set:    14.0
```

Chart title: "StrategyQA (OOD)"
Y-axis range: 10–16

Legend: UNDO*, A2-Full, L set, Approach 1, H set, L+M set

---

### Slide 21 — Chosen k-iteration: When to Stop? (Badge: 17)

**Layout:** Full-slide line chart with explanation text below.

**Chart — "When to Stop: In-domain vs OOD Trend":**

X-axis: Iteration 1–5
Y-axis: Accuracy 0–90

Two lines:
- Blue (In-domain): ~51, 53, 55, 54, 55 (rises then plateaus)
- Orange (OOD / StrategyQA): ~12, 13, 14, 13, 10 (rises then drops)

Annotations:
- Dashed vertical line at iteration 3: label "Stop @ Iter 3"
- Arrow at iter 4-5 of OOD line: "OOD drop → Overfit signal"

**Below chart:**
```
After 3rd iteration, stopping signal appears:
- Performance improves until iteration 3; later iterations show overfitting signal
- OOD drops clearly if training continues
- A signal that student stops learning or is overfitting

"=> Stop at Iteration 3 is justified"
"=> In UNDO preprint, authors confirm that iteration 3 is good point to stop."
```

---

### Slide 22 — Section Divider: "Contribution 2: Reasoning-Aware Error Feedback"

**Layout:** Full dark maroon background. Two lines of large white centered text.

```
"Contribution 2:"
"Reasoning-Aware Error Feedback"
```

---

### Slide 23 — Motivation: Improving the Error Signal in UNDO (Badge: 18)

**Layout:** Left = mini workflow diagram. Right = text explanation.

**Left: Repeat the baseline workflow diagram (simplified)**
Highlight the "Val-set error signals" and "Score (0 or 1 for T/F)" box prominently with a red dashed border to draw attention to the binary score limitation.

**Right column:**
```
- This distillation framework relies on "validation error signals",
  which help the teacher understand the student's weaknesses.

- However, the current signal uses a "very simple score":
    - "1" for correct
    - "0" for incorrect

- This raises an important concern:
  "[Highlighted box] Does such a simple binary score
   provide enough information for the teacher to guide
   student improvement?"

"Research Questions"
1. Can we design a more informative scoring signal to
   replace this simple metric?
```

---

### Slide 24 — Proposed Student Error Feedback Signal (Badge: 19)

**Layout:** Single column with formula display + comparison section + visual signal bar.

**Content:**
```
"Reasoning-Aware Error Signal Score:"

[Large centered formula]:
  S = 0.6 × Answer correctness + 0.4 × Reasoning quality

- This makes "correct answer" dominate, but still tells the teacher
  if the student is improving structurally.

"Compare:"
• Original UNDO: [0 or 1]
    e.g. at iteration 3: Score=[0/1]

• "Proposed method: [Init, Iter1, Iter2, ...]"
    e.g. at iteration 3: Score=[-0.9, -0.4, 0.2]

"This allows the teacher to observe the student's reasoning
improvement (or not improve) across iterations."
```

**Visual — "Student Improvement Signal Across Iterations" bar:**

Horizontal segmented bar (4 segments):
```
[-0.9 (red)]  [-0.4 (orange)]  [0.2 (light green)]  [0.9 (dark green)]
  Init             Iter 1           Iter 2                Iter 3
```

Below bar: scale `-1.0 ——— -0.5 ——— 0.0 ——— 0.5 ——— 1.0`
Caption: `(-1← incorrect reasoning + incorrect answer || correct reasoning + correct answer →1)`

---

### Slide 25 — Proposed Student Error Feedback Signal (cont.) (Badge: 20)

**Layout:** Two-column. Left = explanation text. Right = scoring rubric table.

**Left column:**
```
"Reasoning-Aware Error Signal:"

We measure student performance using two components:

1. "Answer correctness (A):"
   - A = +1 if final answer matches GT (or passes verifier).
   - A = -1 otherwise.

2. "Reasoning quality (R):"
   - Student reasoning is evaluated by an LLM
     using a fixed rubric prompt

"This allows the teacher to observe how reasoning quality
evolves during distillation, not only final correctness."
```

**Right column — Rubric table:**

| Meaning | Value |
|---|---|
| correct logic / minimal gaps | +1.0 |
| minor slip, structure good | +0.5 |
| unclear / missing key steps | 0.0 |
| wrong approach but coherent | -0.5 |
| nonsense / cannot reason | -1.0 |

Style: alternating row colors (white / light gray). Bold header row.

---

### Slide 26 — Test Result: Error Signal Score (Badge: 21)

**Layout:** Full-width horizontal stacked bar chart + comparison settings list below.

**Chart title:** "How Fast the Student Improves (Progressive Hints pipeline) under Different Validation Signals"

**Chart — 4 rows of horizontal bars:**

Each row: one scoring method. Bars show: Iter 1 avg (blue) + Δ to Iter 2 (orange) + Δ to Iter 3 (green). Total = final accuracy at iter 3.

| Method | Iter 1 | Δ Iter 2 | Δ Iter 3 | Final |
|---|---|---|---|---|
| Our score (A+R) | ~52.5 | ~1.0 | ~0.7 | 54.2 |
| BERTScore | ~52.0 | ~0.2 | ~0.5 | 52.7 |
| Binary (0/1) | ~52.5 | ~0.1 | ~0.7 | 53.3 |
| No score | ~52.0 | ~0.0 | ~0.5 | 52.5 |

X-axis: Average Accuracy (%) 46–56
Legend: Iter 1 (Avg) [blue], Δ to Iter 2 [orange], Δ to Iter 3 [green]

**Below chart:**
```
"Compared Scoring Signal Settings (all using our Propose Hint):"
1. No score: Remove score entirely
2. Binary score: original pipeline
3. Semantic Similarity Score: "BERTScore" between student answer and ground truth
4. "Proposed Signal": Combined "Answer Correctness + Reasoning Quality"
```

---

### Slide 27 — Section Divider: "3. DISCUSSION & CONCLUSION"

Full dark maroon background, centered white text.

---

### Slide 28 — Discussion: Why Does Reasoning Preservation Work? (Badge: 22)

**Layout:** Three numbered large points with icons.

**Content:**
```
1. "Breaking the 'Mimicry Trap'"
   ☐ Always given hints often forces the teacher to parrot the answer.
   ☐ Our Progressive-Hints prevents this by giving the minimum information
     necessary to trigger the teacher's internal logic

2. "Robustness vs. Memorization"
   ☐ The L-set consistently outperformed the H-set.
   ☐ This proves that students learn better when the teacher provides diverse
     reasoning paths rather than just a paraphrased solution.

3. "Informative Feedback"
   ☐ The Reasoning-Aware Signal proves that LLM-based teachers can benefit
     from a "Score Assistant" role: analyzing how a student failed, not just
     that they failed.
```

**Visual:** Use three large maroon numbered circle badges (1, 2, 3) on the left of each section, with the title in bold maroon and description in regular dark text.

---

### Slide 29 — End Slide

**Layout:** Dark background. Centered.

```
"The End"
```

Add any acknowledgment or thank-you if desired. Same aesthetic as title slide.

---

### Slide 30 — Backup Slide Divider

**Layout:** Maroon background.

```
"Backup slide"
```

---

### Slide 31 — Reproducing UNDO (Backup, Badge: 2)

**Layout:** Single column.

```
• UNDO is currently available as a preprint, and official code is not released.
• We reproduced UNDO following the preprint's algorithm and training settings:
  – Same distillation procedure and student model setup
  – Same evaluation datasets

"Resource-scaled reproduction"
• Due to compute constraints, we run a scaled-down reproduction:
  – Teacher: ~30B (vs 72B in the paper)
  – Training data: ~10% of the paper's dataset size

"Why this is still useful"
• The goal is to verify the UNDO pipeline and establish a consistent baseline
  under our compute budget.
• This reproduced UNDO baseline will be used as the reference point for
  comparing our method.
```

---

### Slide 32 — Reproducing UNDO Analysis (Backup, Badge: 4)

**Layout:** Two-column. Left = text. Right = 2×2 grid of line charts.

**Left column:**
```
• The performance drop after the 3rd iteration is consistent with the UNDO paper:
  [Quoted box]: "These iteration 4 are lower than iteration 3 results, we conclude
  that the model has converged and additional regeneration from the teacher is
  no longer beneficial"

• As training proceeds, the student model begins to:
  ☐ Fit "more complex patterns" from harder datasets
  ☐ Gradually exhibit "overfitting behavior", especially on easier and OOD
    benchmarks

→ This behavior further confirms that our reproduced UNDO follows the
"same learning and overfitting patterns" as the original method.
```

**Right column — 4 line charts (2×2 grid):**

| Chart | Paper | Reproduce | Notes |
|---|---|---|---|
| GSM8k (Easy) | 54, 56, 57, 55 | 51, 53, 54.2, 53 | Easy benchmark |
| MMLU PRO (Hard) | 14.5, 16, 15.8, 16.5 | 12, 14, 14.5, 14.8 | Hard benchmark |
| StrategyQA (OOD) | N/A | 12.5, 13, 12.6, 11 | OOD |
| Theorem QA (OOD) | N/A | 6.3, 6.4, 6.5, 6.0 | OOD |

---

### Slide 33 — Compare with Other Distillation Methods (Backup, Badge: 22)

**Layout:** Two-column. Left = 2 line charts. Right = grouped bar chart.

**Left charts — Accuracy over training steps:**

Two line charts (GSM8K, MATH). X-axis: Training Step 2–10. Three lines each:
- Standard Dist. (gray dashed): starts ~49, peaks ~54 at step 2, drops to ~44
- Black-box Dist. (gray dashed darker): starts ~48, peaks ~52, drops lower
- Ours (UNDO) (orange solid): peaks earlier and higher than others

**Right chart — Time to Peak (normalized):**

3 stacked bars: Standard Dist., Black-box Dist., Progressive Hints
Each bar has: Student optimization (blue) + Teacher interaction (orange) + Iteration overhead (green)

Values:
```
Standard Dist.:   Peak @ epoch ~5, total height 1.1
Black-box Dist.:  Peak @ epoch ~5, total height 1.8
Progressive Hints: Peak @ iter 3, total height 3.0
```

**Below charts:**
```
Standard Distillation: Student mimics a "frozen" snapshot of teacher knowledge.
Black-box Distillation: Student engages in "live dialogue" with the teacher during training.
Our method: Teacher "diagnoses" student failures to provide targeted, high-reasoning guidance.
```

---

## 4. Implementation Instructions for Claude Code

### Step 1: Create the PptxGenJS script

```bash
cd /home/claude/undo_pptx
cat > package.json << 'EOF'
{
  "name": "undo-presentation",
  "version": "1.0.0",
  "dependencies": {
    "pptxgenjs": "^3.12.0"
  }
}
EOF

npm install
```

Create `generate.js` and implement all slides per the specifications above. Use the following structure:

```javascript
const pptx = require("pptxgenjs");
const pres = new pptx();

// Set slide size (16:9)
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 inches

// ── MASTER DEFAULTS ──────────────────────────────────────
const MAROON   = "7B1D1D";
const DARK     = "1A1A1A";
const WHITE    = "FFFFFF";
const GRAY_BG  = "F5F5F5";
const TEXT     = "2C2C2C";
const ORANGE   = "E67E22";
const BLUE     = "2980B9";
const NAVY     = "1F3864";

// ── HELPER: add standard content slide header ────────────
function addSlideHeader(slide, badgeNum, title) {
  // Maroon badge (left)
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.2, y: 0.15, w: 0.55, h: 0.55,
    fill: { color: MAROON }, line: { color: MAROON }
  });
  slide.addText(String(badgeNum), {
    x: 0.2, y: 0.15, w: 0.55, h: 0.55,
    fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle"
  });
  // Title
  slide.addText(title, {
    x: 0.85, y: 0.15, w: 12.0, h: 0.55,
    fontSize: 28, bold: true, color: MAROON, valign: "middle"
  });
  // Horizontal rule
  slide.addShape(pptx.ShapeType.line, {
    x: 0.2, y: 0.78, w: 12.9, h: 0,
    line: { color: MAROON, width: 2 }
  });
  // Page number (bottom right)
  slide.addText(String(badgeNum), {
    x: 12.5, y: 7.1, w: 0.6, h: 0.3,
    fontSize: 10, color: "999999", align: "right"
  });
}

// ... [implement each slide function here]

pres.writeFile({ fileName: "UNDO_Hinted_Presentation.pptx" })
  .then(() => console.log("Done: UNDO_Hinted_Presentation.pptx"))
  .catch(err => console.error(err));
```

### Step 2: Handle charts using matplotlib for complex KDE plots

For KDE distribution charts (Slides 13, 17) that cannot be rendered natively in PptxGenJS, generate them as PNG images first:

```bash
pip install matplotlib numpy scipy --break-system-packages
```

```python
# chart_kde.py — generate KDE chart images
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Slide 13: Correct vs Wrong hint distribution
fig, ax = plt.subplots(figsize=(5, 3.5))

correct_sim = np.random.beta(8, 2, 300)    # peaks near 0.8
wrong_sim   = np.random.beta(2, 6, 300)    # peaks near 0.25

ax.hist(correct_sim, bins=15, alpha=0.4, color='steelblue', label='solution_hint (correct)', density=True)
ax.hist(wrong_sim,   bins=15, alpha=0.4, color='orange',    label='solution_hint (wrong-forced)', density=True)

kde_c = gaussian_kde(correct_sim); xs = np.linspace(0,1,200)
kde_w = gaussian_kde(wrong_sim)
ax.plot(xs, kde_c(xs), color='green',  linewidth=2, label='KDE (correct)')
ax.plot(xs, kde_w(xs), color='red',    linewidth=2, linestyle='--', label='KDE (wrong-forced)')

ax.annotate('Wrong-hint', xy=(0.22, 3.5), fontsize=10, color='darkorange')
ax.annotate('Correct-hint', xy=(0.72, 1.6), fontsize=10, color='steelblue')
ax.set_xlabel('Similarity score (BERTScore F1)')
ax.set_ylabel('Density')
ax.legend(fontsize=7)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig('chart_kde_slide13.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart_kde_slide13.png")
```

Repeat similarly for Slide 17 (two stacked KDE plots). Save all chart images to `/home/claude/undo_pptx/charts/`.

Then reference in PptxGenJS:
```javascript
slide.addImage({ path: "./charts/chart_kde_slide13.png", x: 5.5, y: 1.0, w: 7.0, h: 4.5 });
```

### Step 3: Generate the PPTX

```bash
cd /home/claude/undo_pptx
python chart_kde.py         # generate chart images
python chart_bars.py        # generate bar chart images for complex charts
node generate.js            # build the PPTX
```

### Step 4: Visual QA

```bash
cd /home/claude/undo_pptx
python /mnt/skills/public/pptx/scripts/office/soffice.py --headless \
  --convert-to pdf UNDO_Hinted_Presentation.pptx

rm -f slide-*.jpg
pdftoppm -jpeg -r 150 UNDO_Hinted_Presentation.pdf slide
ls -1 "$PWD"/slide-*.jpg
```

Then view each slide image and check for:
- Overlapping text / shapes
- Text cut off at edges
- Chart labels legible
- Color contrast sufficient
- Badge numbers correct and matching slide content
- Section dividers dark maroon (not just gray)

Fix issues → re-run node generate.js → re-run soffice + pdftoppm → re-inspect.

### Step 5: Copy to output

```bash
cp /home/claude/undo_pptx/UNDO_Hinted_Presentation.pptx \
   /mnt/user-data/outputs/UNDO_Hinted_Presentation.pptx
```

---

## 5. Priority Checklist

Before declaring success, verify ALL of the following:

- [ ] All 28+ slides present and in correct order
- [ ] Title slide has dark background + correct title/author/date
- [ ] Three section dividers are dark maroon, not just dark gray
- [ ] Every content slide has: maroon number badge, bold title, horizontal rule
- [ ] All flowchart diagrams rendered (Slides 4, 9, 12, 14) — not placeholder text
- [ ] All bar/line charts rendered with correct data values and legends (Slides 5, 11, 13, 15, 17, 19, 20, 21, 26, 32, 33)
- [ ] KDE distribution charts embedded as images (Slides 13, 17)
- [ ] Signal improvement bar (Slide 24) rendered correctly with color gradient
- [ ] Hypothesis boxes use bordered rectangle style
- [ ] Formula on Slide 24 (S = 0.6×A + 0.4×R) is large and legible
- [ ] Rubric table on Slide 25 is clean with alternating rows
- [ ] Slide numbers bottom-right are correct on all slides
- [ ] No text overflow beyond slide boundaries
- [ ] Font is Calibri throughout (fallback: Arial)
- [ ] Visual QA with soffice+pdftoppm completed at least once

---

## 6. Key Academic Content to Preserve Exactly

These items must appear verbatim (or near-verbatim) as they are findings/formulas:

1. **Formula:** `S = 0.6 × Answer correctness + 0.4 × Reasoning quality`
2. **Hypothesis 1:** *"Models trained on low-similarity (high-reasoning) samples will exhibit superior out-of-domain robustness compared to those trained on high-similarity (copy solution) samples."*
3. **Hypothesis 2 (verified):** *"Teacher's Internal Reasoning Leads to Better Student Adaptation"*
4. **Score example:** `Score = [-0.9, -0.4, 0.2]` at iteration 3
5. **Rubric table** (Slide 25) — all 5 rows with exact values
6. **Performance gains** (Slide 15): GSM8K +7.0, MATH +5.2, MMLU-Pro +1.5, SVAMP +1.0
7. **Three groups:** L (<0.5), M (0.5–0.75), H (>0.75)
