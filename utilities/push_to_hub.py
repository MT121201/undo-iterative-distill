from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_DIR = "./student_ckpt"   # path where your trained model/tokenizer is saved
HF_REPO_ID = "MinTR-KIEU/qwen2.5-student-I1"  # replace with your repo
PRIVATE = True  # False if you want public

print(f"Loading from {LOCAL_DIR}...")
model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)

print(f"Pushing to https://huggingface.co/{HF_REPO_ID}...")
model.push_to_hub(HF_REPO_ID, private=PRIVATE, token=True)
tokenizer.push_to_hub(HF_REPO_ID, private=PRIVATE, token=True)

print("✅ Model pushed successfully!")
