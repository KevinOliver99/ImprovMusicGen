Environment Setup:

python -m venv env
source env/bin/activate
pip install -r requirements.txt

This code was run on an HPC using a single A100 GPU (80GB VRAM, 50 CPU cores). Parameters in the configs (hyperparameters in train_jam.py and eval_jam.py, computational resources in run_train.sbatch and run_eval.sbatch) may need to be adapted for varying setups.

Start training with:
sbatch run_train.sbatch

Start evaluation with:
sbatch run_eval.sbatch


Training produces a checkpoint after every epoch.


Validation produces 3 seperate audio files:
1. The first file is the raw musical context fed to the model. This includes all stems other than the target stem, as well as the first 20 beats (context length) of the target stem. There is usually an audible dropout of the target stem around the middle of the audio.
2. The second file is only the target stem. This includes the same first 20 beats of context that were included in the above audio file, followed by the models autoregressive output for another 20 beats.
3. The third file merges both the musical context and the models autoregressive output into a single file.