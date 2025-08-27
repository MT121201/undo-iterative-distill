import argparse
import os
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt.teacher_prompt import TEACHER_PROMPT_ITER0, TEACHER_PROMPT_ITER1
import datetime
from src.evaluate import evaluate_teacher_response

MODEL_NAME = "Qwen/Qwen3-8B"

def _save_to_new_dataset(data, output_file):
    """
    Save correct model response to new dataset
    """
    with open(output_file, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="10k", help="Dataset size 10k/100k'")
    parser.add_argument("--iter", default="0", help="Iteration number")
    parser.add_argument("--output", default="new_dataset.jsonl", help="Path to save new dataset")
    parser.add_argument("--cont", type=int, default=None, help="Continue from this sample index")
    args = parser.parse_args()

    # Load dataset
    if args.dataset == "10k":
        print("Loading 10k dataset")
        dataset_name = "MinTR-KIEU/NuminaMath-CoT-10k"
    elif args.dataset == "100k":
        print("Loading 100k dataset")
        dataset_name = "MinTR-KIEU/NuminaMath-CoT-100k"
    else:
        raise ValueError("Unknown dataset: choose '10k' or '100k'")

    dataset = load_dataset(dataset_name, split="train")

    # Continues sampling from index
    if args.cont is not None:
        dataset = dataset.select([i for i in list(range(args.cont, len(dataset)))])

    # Load prompts
    if args.iter == "0":
        print("Using Iteration 0 Prompt")
        TEACHER_PROMPT = TEACHER_PROMPT_ITER0
    else:
        print("Using Iteration 1+ Prompt")
        TEACHER_PROMPT = TEACHER_PROMPT_ITER1

    # load the tokenizer and the model
    print("Using ", MODEL_NAME, "model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
)

    # Prepare output file
    if os.path.exists(args.output):
        base, ext = os.path.splitext(args.output)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base}_{timestamp}{ext}"
        print(f"Output file exists. Saving to new file: {args.output}")

    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        # NOTE: Add a timer here calculate remaining time

        # Dataset keys:  dict_keys(['source', 'problem', 'solution', 'messages'])
        # NOTE: Fix this after have prompt template
        prompt = prompts[idx % len(prompts)].format(**example)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Evaluate response
        gt = example.get("answer") or example.get("output") or example.get("gt") or example.get("solution")
        results = evaluate_teacher_response(response, gt)
        # results:
        # Dict[str, Any]: A dictionary with the following keys:
        #     - 'has_boxed' (bool): Whether a boxed answer was found in the response.
        #     - 'extracted_answer' (str or None): The extracted answer from the boxed content.
        #     - 'is_correct' (bool or None): Whether the extracted answer matches the expected answer (None if expected is not provided).
        #     - 'comparison_mode' (str or None): The mode of comparison used ('pm', 'numeric', 'choice', 'string', or None).
        #     - 'details' (str): Additional details about the comparison process.
        if results.get("has_boxed"):
            if results.get("is_correct"):
                # If the response is correct, we can save it
                _save_to_new_dataset([{
                    "input": example.get("problem"),
                    "response": response,
                    "gt": gt,
                    "idx": idx
                }], args.output)

if __name__ == "__main__":
    main()
