#!/bin/bash
# ---------------- ORCA / Slurm header ----------------
#SBATCH -J yolo12s_smoke_test
#SBATCH --time=1:00:00
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=8                 # 8 CPU cores total  (8 dataloader workers)
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu
# ────────────────────────────────────────────

# ───────────── host-side paths ─────────────
DATA_DIR=/scratch/dannycharm-alt-REU/DRAIV/datasets
IMG=/scratch/dannycharm-alt-REU/DRAIV/draiv-dev_0.1.sif
REPO=/scratch/dannycharm-alt-REU/DRAIV           # git repo root
RUN_HOST=$REPO/runs
# ────────────────────────────────────────────

export ULTRALYTICS_WORKERS=8          # silence worker warning
cd "$REPO"

# ───────────── quick sanity checks ─────────
[ -d "$DATA_DIR/bdd100k/images/100k/train" ] || echo "  DATA_DIR not found"
[ -f "$REPO/configs/bdd100k_det.yaml" ]               || echo "  YAML missing"
# ────────────────────────────────────────────

apptainer exec --nv --cleanenv \
  -B "$DATA_DIR":/datasets \
  -B "$REPO":/workspace \
  -B "$RUN_HOST":/workspace/runs \
  "$IMG" \
   yolo detect train \
      model=yolo12s.pt \
      project=/workspace/runs/my_yolo_experiments \
      name=smoke_test_yolo12s_01 \
      epochs=3 \
      batch=48 imgsz=640 \
      half=True cache=labels workers=8 \
      cos_lr=True \
      save_period=5 \
      data=/workspace/configs/bdd100k_det.yaml      # ← no “\” here
