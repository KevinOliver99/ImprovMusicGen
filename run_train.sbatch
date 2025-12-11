#!/bin/bash
#SBATCH --job-name=jam_full
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=80G
#SBATCH --time=120:00:00
#SBATCH --output=jam_full_%j.out
#SBATCH --error=jam_full_%j.err

source env/bin/activate

echo "Starting full fine-tuning for ALL roles on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

accelerate launch --mixed_precision="bf16" train_jam.py
