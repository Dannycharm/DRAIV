#!/bin/bash
# ---------------- ORCA / Slurm header ----------------
#SBATCH -J unet_bdd100k_preprocess
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu
# ────────────────────────────────────────────

source /scratch/dannycharm-alt-REU/DRAIV/virtual_env/bin/activate

python3 /scratch/dannycharm-alt-REU/DRAIV/scripts/unet_scripts/prepare_unet_dataset.py
