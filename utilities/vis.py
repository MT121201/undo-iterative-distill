import matplotlib.pyplot as plt
import os
import subprocess

# Data: accuracy per iteration for Qwen2.5 1.5B
data = {
    "Qwen2.5-1.5B": {
        "Iteration": [0, 1, 2, 3],
        "Math500": [9.6, None, None, None],
        "GSM8K": [31.16,None, None, None],
        "MMLU_PRO": [13.76, 14.53, None, None],
        "SVAMP": [75.79, 83.33, None, None],
    }
}


datasets = ["Math500", "GSM8K", "MMLU_PRO", "SVAMP"]
models = list(data.keys())

fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 subplots

for i, dataset in enumerate(datasets):
    ax = axes[i]
    for model in models:
        iterations = data[model]["Iteration"]
        accuracies = data[model][dataset]
        ax.plot(iterations, accuracies, marker="o", label=model)
    ax.set_title(dataset)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([0, 1, 2, 3, 4])  # integer ticks only
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# Ensure save directory exists
os.makedirs("vis", exist_ok=True)
save_path = "vis/student_vis.png"
plt.savefig(save_path)
print(f"Plot saved to {save_path}")