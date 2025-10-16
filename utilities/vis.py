import matplotlib.pyplot as plt
import os
import subprocess

# Data: accuracy per iteration for Qwen2.5 1.5B
data = {
    "Qwen2.5-1.5B": {
        "Iteration": [0, 1, 2, 3, 4, 5],
        "Math500": [9.6, 32.4, 35.80, 37.85, 33.20, 33.20],
        "GSM8K": [31.16, 47.65, 54.22, 54.80, 52.62, 51.33],
        "MMLU_PRO": [13.76, 14.7, 15.32, 15.30, 15.10, 15.28],
        "SVAMP": [75.79, 83.33, 84.40, 885.42, 85.10, 84.66],
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