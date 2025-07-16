#!/bin/bash
#SBATCH -J  smoke_test_TwinLiteNetPlus    # Job name
#SBATCH --time=3-00:00:00             # 3 day wall-clock
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=26                 # 26 CPU cores total  
#SBATCH --gres=gpu:l40s:1          # 1 L40S GPU  
#SBATCH --mem=90G                    
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu

module load apptainer/1.1.9-gcc-13.2.0

# ───────────── host-side paths ─────────────
DATA_DIR=/scratch/dannycharm-alt-REU/DRAIV/datasets
IMG=/scratch/dannycharm-alt-REU/DRAIV/draiv_ml.sif
REPO=/scratch/dannycharm-alt-REU/DRAIV           # git repo root
# ────────────────────────────────────────────

apptainer exec --nv \
  --bind "$DATA_DIR":/datasets,"$REPO":/workspace \
  "$IMG" \
    python /workspace/scripts/TwinLiteNetPlus_scripts/train.py --config medium --max_epochs 100 --savedir /workspace/runs/TwinLiteNetPlus_lane/full_train_02 --ema
