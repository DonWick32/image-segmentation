# 🛠️ Continual Surgical Tool Segmentation with LoRA-SAM2
Robust segmentation of surgical tools in robot-assisted procedures is vital for patient safety and automation. This repository implements a domain-incremental continual learning (CL) framework using the Segment Anything Model 2 (SAM2) combined with Low-Rank Adaptation (LoRA). The method addresses challenges such as smoke, blood, and illumination changes while avoiding catastrophic forgetting and preserving privacy.

## ⚙️ Installation & Setup
1. SAM2 Environment Setup
Please follow the official SAM2 GitHub repository instructions to clone and set up the SAM2 development environment, including creating the conda environment and installing SAM2 core dependencies.

2. Download SAM2 Base Model
Download the SAM2 base model checkpoint:
```
bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -P checkpoints/
```

Make sure your config.yaml contains the model checkpoint path:
```yaml
model:
  # checkpoint: "checkpoints/sam2.1_hiera_small.pt"
  checkpoint: "checkpoints/sam2.1_hiera_base_plus.pt"
```

3. Install Additional Dependencies
Install the required Python packages:

```bash
pip install omegaconf wandb safetensors opencv-python tqdm
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Log in to Weights & Biases using
```bash
wandb login
```

# 🗂️ Project Structure
```
SAM2-Surgical-CL/
│
├── annotations/              
├── data/                     
├── jobs/, jobs2/             
│
├── baseline.ipynb            
├── inference.ipynb           
│
├── baseline.py               
├── inference.py              
├── train.py                  
├── train_single_domain.py    
├── train_cl.py               
├── train_cl_kmean.py         
├── train_lora_sam.py         
│
├── evaluate.py               
├── sam2_model.py             
├── sam2_dataset.py           
├── utils.py                  
│
├── lora_qkv.py               
├── config.yaml               
├── lora_sam_config.yaml      
│
├── sbatch_run.sh             
├── sbatch_run2.sh            
├── sbatch_run3.sh            
│
├── requirements.txt          
├── .gitignore
└── README.md
```

# 📂 Dataset
The dataset used is SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarially Generated.

Place the dataset anywhere on your system, for example:


```bash

~/segstrong/
├── SegSTRONGC_test/
├── SegSTRONGC_val/
```
Update your config.yaml with the dataset root path:

```yaml
dataset:
  path: "~/segstrong"
```

# 🚀 Training
1. Configure dataset and model paths
Ensure that the dataset path and model checkpoint are correctly set in config.yaml.

2. Run continual learning training
```bash
sbatch sbatch_run.sh
```



