# Installation
## Environment Setup
We recommend using Conda for managing your Python environments. 
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
```bash
source ~/miniconda3/bin/activate
```
```bash
conda init --all
```
Then accept their requirements and restart your terminal.
## Teacher Model - QWEN3-30B-A3B
Better to create a separate environment for the teacher model:
```bash
conda create -n teacher python=3.10 -y
conda activate teacher
```
Installation of dependencies:
```bash
pip install -r requirements.txt
```

### HF token
For pushing to Hugging Face
```bash
export HF_TOKEN="your_huggingface_token"
```


### Flash-attn
For saving GPU memory, install the following package for
```bash
pip install flash-attn --no-build-isolation
```

## Student Model Qwen2.5-1.5B
```bash
conda create -n student python=3.10 -y
conda activate student
```
Installation of dependencies:
```bash
pip install -r requirements.txt
```

### HF token
For pushing to Hugging Face
```bash
export HF_TOKEN="your_huggingface_token"
```