#!/bin/bash
#SBATCH -J yolo-exp_v12s_05
#SBATCH -p secondary                 # H100 partition
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8                         # dataloader workers
#SBATCH --mem=60G
#SBATCH -t 4:00:00
#SBATCH -o /u/pdanie20/scratch/DRAIV_altREU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu

# ───────────── host-side paths ─────────────
DATA_DIR=/u/pdanie20/scratch/DRAIV_altREU/datasets
IMG=/u/pdanie20/scratch/DRAIV_altREU/draiv-dev_0.1.sif
REPO=/u/pdanie20/scratch/DRAIV_altREU/DRAIV           # git repo root
RUN_HOST=$REPO/runs                                   # contains exp0_1
CKPT_HOST=$RUN_HOST/my_yolo_experiments/exp0_1/weights/last.pt
# ────────────────────────────────────────────

export ULTRALYTICS_WORKERS=8          # silence worker warning
cd "$REPO"

# ───────────── quick sanity checks ─────────
[ -d "$DATA_DIR/bddk100k/bdd100k/images/100k/train" ] || echo "⚠️ DATA_DIR not found"
[ -f "$REPO/configs/bdd100k_det.yaml" ]               || echo "⚠️ YAML missing"
[ -f "$CKPT_HOST" ] || { echo "🚫 Checkpoint missing: $CKPT_HOST"; exit 1; }
# ────────────────────────────────────────────

apptainer exec --nv --cleanenv \
  -B "$DATA_DIR":/datasets \
  -B "$REPO":/workspace \
  -B "$RUN_HOST":/workspace/runs \
  "$IMG" \
   yolo detect train \
      model=/workspace/runs/my_yolo_experiments/exp_v12s_04/weights/last.pt \
      project=/workspace/runs/my_yolo_experiments \
      name=exp_v12s_05 \
      epochs=60 \
      batch=48 imgsz=640 \
      half=True cache=labels workers=8 \
      cos_lr=True warmup_epochs=0 \
      save_period=5 nbs=48 \
      data=/workspace/configs/bdd100k_det.yaml      # ← no “\” here

