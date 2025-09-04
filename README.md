
# INTERACTIVE DISTILLATION VIA UNDO METHOD

![UNDO Overview](vis/UNDO.png)

This project implements and extends the [UNDO framework](https://arxiv.org/pdf/2504.02521), which introduces **interactive knowledge distillation** as an iterative optimization process.
Instead of relying on a one-shot teacher–student transfer, UNDO repeatedly:

1. Distills knowledge from the teacher,
2. Evaluates the student,
3. Identifies student errors,
4. Regenerates refined teacher rationales conditioned on those errors,
5. Retrains the student with improved *teacher-distilled datasets*.

This feedback loop makes the student model learn more efficiently and adaptively, particularly for challenging **mathematical and reasoning tasks**.

Our implementation follows two goals:

1. **Validate** whether UNDO works as described in the paper.
2. **Enhance** UNDO with practical improvements (better prompts, tooling, and efficiency).

![Pipeline](vis/UNDO_pipe.png)

---

##  Updates

* **2025/09/02** – New teacher prompt for higher-quality rationales ([code](src/prompt/teacher_prompt.py))
* **2025/09/03** – Teacher model **QWEN3-30B-A3B** running to generate the first distilled dataset on 10k samples (\~160 GPU hours). Progress: [Teacher\_CoT\_NuminaMath\_10k\_I0](https://huggingface.co/datasets/MinTR-KIEU/Teacher_CoT_NuminaMath_10k_I0)

---

## Progress Checklist

| Status | Task                                    |
| ------ | --------------------------------------- |
| ✔️     | Prepare training dataset                |
| ✔️     | Implement teacher model (1st iteration) |
| ✔️     | Evaluation method for teacher responses |
| ✔️     | Prompt teacher model for good answers   |
| In Progress      | Generate 1st teacher-distilled dataset  |
| ⬜      | Implement student model                 |
| ⬜      | Train student on distilled dataset      |
| ⬜      | Validate student model                  |
| ⬜      | Generate 2nd iteration dataset          |
| ⬜      | ...         |


---

##  Datasets

| Type               | Name                              | Link                                                                             |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------------- |
| Train              | NuminaMath-CoT-10k                | [link](https://huggingface.co/datasets/MinTR-KIEU/NuminaMath-CoT-10k)            |
| Train              | NuminaMath-CoT-100k               | [link](https://huggingface.co/datasets/MinTR-KIEU/NuminaMath-CoT-100k)           |
| Distilled (Iter 0) | Teacher\_CoT\_NuminaMath\_10k\_I0 | [link](https://huggingface.co/datasets/MinTR-KIEU/Teacher_CoT_NuminaMath_10k_I0) |
| Distilled (Iter 1) | –                                 | \[]                                                                              |
| Distilled (Iter 2) | –                                 | \[]                                                                              |
| Test               | Math500                           | \[]                                                                              |
| Test               | GSM8K                             | \[]                                                                              |
| Test               | MMLU PRO                          | \[]                                                                              |
| Test               | SVAMP                             | \[]                                                                              |

---

## 💻 GPU Usage

| Model         | Recommended VRAM | Current Setup               | Platform | Notes                            |
| ------------- | ---------------- | --------------------------- | -------- | -------------------------------- |
| QWEN3-30B-A3B | 64 GB            | 6× RTX 5060 Ti (16 GB)     | GPU2     | \~4× RTX 3090 (24 GB) equivalent |

---

## ⚙️ Installation

See [doc/installation.md](doc/installation.md) for detailed setup instructions.

---

## 🚀 Running

### Teacher Inference

Set your Hugging Face token first:

```bash
export HF_TOKEN="your_huggingface_token"
```

Run teacher inference to generate a **teacher-distilled dataset**:

```bash
python src/teacher.py \
    --dataset NUMINA_10K_or_100K \
    --iter ITERATION_ID \
    --output output.jsonl \
    --hf_repo MinTR-KIEU/Teacher_CoT_NuminaMath_10k_I0 \
    --hf_private 0 \
    --push_every_min 0
```

Parameters:

* `--dataset` : Choose between `10k` or `100k`.
* `--iter` : `0` = first distillation, `1` = second, etc.
* `--output` : Local output file.
* `--hf_repo` : Hugging Face repo for dataset.
* `--hf_private` : 0 = public, 1 = private.
* `--push_every_min` : Auto-push interval (0 = disabled).

---

## 📝 Notes for Reproduction

* Evaluation is handled with our custom extractor ([code](src/evaluate.py)) that parses final answers inside `$\boxed{...}$`.
* This works well for most math problems, but may fail if:

  * The teacher outputs long strings instead of numbers,
  * Multiple answers are given instead of one boxed result.

