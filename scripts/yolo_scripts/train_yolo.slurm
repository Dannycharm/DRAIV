#!/bin/bash
# ---------------- ORCA / Slurm header ----------------
#SBATCH -J yolo12m-001                # Job name
#SBATCH --time=3-00:00:00             # 3 day wall-clock
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=10                 # 10 CPU cores total  (8 dataloader workers)
#SBATCH --gres=gpu:l40s:1          # 1 L40S GPU  (change to a30 etc.)
#SBATCH --mem=90G                    # all mem on the node is mem=0 but long queue (or set e.g. 120G)
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu

# ───────────── host-side paths ─────────────
DATA_DIR=/scratch/dannycharm-alt-REU/DRAIV/datasets
IMG=/scratch/dannycharm-alt-REU/DRAIV/draiv-dev_0.1.sif
REPO=/scratch/dannycharm-alt-REU/DRAIV           # git repo root
RUN_HOST=$REPO/runs                                   
# ────────────────────────────────────────────

export ULTRALYTICS_WORKERS=10          # silence worker warning
cd "$REPO"

# ───────────── quick sanity checks ─────────
[ -d "$DATA_DIR/bdd100k/images/100k/train" ] || echo "⚠️ DATA_DIR not found"
[ -f "$REPO/configs/bdd100k_det.yaml" ]               || echo "⚠️ YAML missing"
# ────────────────────────────────────────────

module load apptainer/1.1.9-gcc-13.2.0

apptainer exec --nv --cleanenv \
  -B "$DATA_DIR":/datasets \
  -B "$REPO":/workspace \
  -B "$RUN_HOST":/workspace/runs \
  "$IMG" \
   yolo detect train \
      model=yolo12m.pt \
      project=/workspace/runs/my_yolo_experiments \
      name=yolo12m_01 \
      epochs=100 \
      batch=48 imgsz=640 \
      half=True cache=labels workers=10 \
      cos_lr=True warmup_epochs=3 \
      save_period=5 \
      data=/workspace/configs/bdd100k_det.yaml      # ← no “\” here

