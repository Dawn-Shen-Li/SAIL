#!/bin/bash

# Name your conda environment
ENV_NAME="mllm"

# Create the conda environment with Python 3.10 for best compatibility
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
conda init
conda activate $ENV_NAME

# Install PyTorch 2.2.1 with CUDA 12.1 from PyTorch channel
# Also installs compatible torchvision and torchaudio
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install numpy==1.22.4 transformers==4.37.2 nltk peft wandb jsonlines

# MM-related packages
pip install -U openmim
mim install mmcv==2.2.0 mmengine==0.10.5

# Vision/Detection utilities
pip install timm pycocotools shapely terminaltables scipy

# Large model & optimization tools
pip install deepspeed fairscale lvis 

# Done
echo "✅ Conda environment '$ENV_NAME' created and ready."
