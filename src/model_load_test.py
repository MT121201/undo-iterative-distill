import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Make sure all 6 GPUs are visible to this process
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Leave ~1 GiB headroom per GPU
max_mem = {i: "15GiB" for i in range(6)}  # i = 0..5 for your six GPUs
# max_mem["cpu"] = "48GiB"                  # optional fallback


tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_mem,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)


print(model.hf_device_map)  # verify all cuda:0..5 are used
