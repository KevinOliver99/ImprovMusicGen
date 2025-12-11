#!/bin/bash
#SBATCH --job-name=eval_jam_full
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=eval_jam_full_%j.out
#SBATCH --error=eval_jam_full_%j.err

source env/bin/activate

echo "Starting evaluation on $(hostname)"
python eval_jam.py
