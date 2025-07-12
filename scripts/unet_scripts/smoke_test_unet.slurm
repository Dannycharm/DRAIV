#!/bin/bash
#SBATCH -J smoke_test_unet          # Job name
#SBATCH --time=03:00:00             # 18 hour wall-clock
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=10                 # 10 CPU cores total  (8 dataloader workers)
#SBATCH --gres=gpu:l40s:1          # 1 L40S GPU  (change to a30 etc.)
#SBATCH --mem=64G                    # all mem on the node is mem=0 but long queue (or set e.g. 120G)
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
  python /workspace/scripts/unet_scripts/train_unet.py \
     --data_root /datasets/unet/processed \
     --epochs 3 --batch_size 16 --lr 1e-3 \
     --save_dir /workspace/runs/unet_lane/unet_smoke_test

