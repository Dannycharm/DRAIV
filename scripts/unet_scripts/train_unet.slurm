#SBATCH -J unet_lane                # Job name
#SBATCH --time=3-00:00:00             # 3 day wall-clock
#SBATCH --nodes=1                  # one node is enough for 1 GPU
#SBATCH --ntasks=10                 # 10 CPU cores total  (8 dataloader workers)
#SBATCH --gres=gpu:l40s:1          # 1 L40S GPU  (change to a30 etc.)
#SBATCH --mem=90G                    # all mem on the node is mem=0 but long queue (or set e.g. 120G)
#SBATCH -o /scratch/dannycharm-alt-REU/DRAIV/runs/logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdanie20@illinois.edu

"""
Train a 2-class U-Net (lane vs background) on the pre-processed BDD100K dataset.

Usage
-----

python train_unet.py \
  --data_root /datasets/unet/processed \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --save_dir runs/unet_lane

"""

import argparse, os, time, json, math
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp   # pip install segmentation-models-pytorch
from unet_dataset import UNetSegmentationDataset

# ────────────────────────────────────────────────────────────────────────────────

def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="processed/ directory with images/{train,val} and masks/{train,val}")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", default="runs/unet_lane")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────

def dice_coefficient(pred, target, eps=1e-6):

    pred = pred.softmax(1)[:,1]          # probability of class 1
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps) / (union + eps)

# ────────────────────────────────────────────────────────────────────────────────

def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data ----------------------------------------------------------------------
   
 tf_img = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],

                    std=[0.229,0.224,0.225])

    ])

    tf_mask = T.Compose([
        T.PILToTensor(),
        lambda x: x.squeeze(0).long()

    ])

    train_ds = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/train",
        mask_dir=f"{args.data_root}/masks/train",
        transform=tf_img,
        target_transform=tf_mask
    )

    val_ds   = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/val",
        mask_dir=f"{args.data_root}/masks/val",
        transform=tf_img,
        target_transform=tf_mask
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Model ---------------------------------------------------------------------

    model = smp.Unet(encoder_name="resnet34",
                     encoder_weights="imagenet",
                     classes=2, activation=None).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_iou, start = 0.0, time.time()
    history = []

    # Training loop -------------------------------------------------------------

    for epoch in range(1, args.epochs + 1)
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation ------------------------------------------------------------

        model.eval(); val_loss = 0.0; val_iou = 0.0

        with torch.no_grad():

            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss += criterion(out, masks).item() * imgs.size(0)
                val_iou  += dice_coefficient(out, masks.float()).item() * imgs.size(0)

        val_loss /= len(val_ds)
        val_iou  /= len(val_ds)

        # Logging ---------------------------------------------------------------

        history.append({

            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_iou

        })

        print(f"[{epoch:03}/{args.epochs}] "

              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={val_iou:.4f}")

        # Checkpoint ------------------------------------------------------------

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")

    # Save training log ---------------------------------------------------------

    json.dump(history, open(f"{args.save_dir}/history.json","w"), indent=2)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min. Best Dice: {best_iou:.4f}")

if __name__ == "__main__":

    main()

