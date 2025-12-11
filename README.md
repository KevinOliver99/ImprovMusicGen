# ImprovMusicGen

**ImprovMusicGen** is a deep learning project focused on improvisational music generation using the MusicGen architecture and the Slakh dataset. This repository contains the code for training and evaluating the model, designed to generate role-aware musical improvisation based on a provided musical context.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data & Models](#data--models)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Outputs](#outputs)
- [Configuration](#configuration)

## 🛠 Prerequisites

This code was originally developed and tested on an HPC environment with the following specifications:
- **GPU:** Single NVIDIA A100 (80GB VRAM)
- **CPU:** 50 Cores

**Note:** Hyperparameters in `train_jam.py` and `eval_jam.py`, as well as computational resources defined in `.sbatch` files, may need to be adapted if running on different hardware.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KevinOliver99/ImprovMusicGen.git
   cd ImprovMusicGen
   ```

2. **Set up the Python environment:**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

## 💾 Data & Models

To run the training or evaluation, you need to download the Slakh dataset and the MusicGen small model.

1. **Dataset:**
   Download the Slakh dataset from Hugging Face:
   [WhatzInTheGrass/ImprovMusicGen-Slakh](https://huggingface.co/datasets/WhatzInTheGrass/ImprovMusicGen-Slakh)
   

2. **Models:**
   Download the MusicGen small model from Hugging Face:
   [WhatzInTheGrass/ImprovMusicGen-Models](https://huggingface.co/WhatzInTheGrass/ImprovMusicGen-Models)
   

   *Place the downloaded `DATA/` and `models/` directories in the working directory, along with the code you cloned from the GitHub repository.

## 🚀 Usage

### Training
To start the training process using the SLURM workload manager:

```bash
sbatch run_train.sbatch
```
*Training produces a checkpoint after every epoch.*

### Evaluation
To evaluate the model and generate samples:

```bash
sbatch run_eval.sbatch
```

## 🎵 Outputs

During validation/evaluation, the model produces 3 separate audio files for each sample:

1.  **Context (`*_context.wav`)**: 
    The raw musical context fed to the model. This includes all stems *other than* the target stem, plus the first 20 beats (context length) of the target stem. 
    *Note: There is usually an audible dropout of the target stem around the middle of the audio where the generation is supposed to begin.*

2.  **Target (`*_target.wav`)**: 
    Only the target stem. This includes the same first 20 beats of context, followed by the model's autoregressive output for the subsequent 20 beats.

3.  **Mix (`*_mix.wav`)**: 
    The final result merging both the musical context and the model's generated output into a single cohesive track.

## 🔧 Configuration

- **Hyperparameters:** Adjustable in `train_jam.py` and `eval_jam.py`.
- **Compute Resources:** Adjustable in `run_train.sbatch` and `run_eval.sbatch`.

---
*Author: Kevin Bretz*
*Email: k.o.bretz@umail.leidenuniv.nl*