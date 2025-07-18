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
  --amp 
"""

import argparse, os, time, json, math
from tqdm import tqdm
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
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
    p.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP) training")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────────

def dice_coefficient(pred, target, eps=1e-6):

    pred = pred.softmax(1)[:,1]          # probability of class 1
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps) / (union + eps)

# ────────────────────────────────────────────────────────────────────────────────

def soft_dice_loss(pred, target, eps=1e-6):

    prob = pred.softmax(1)[:, 1]          # (B,H,W)
    target = target.float()
    inter = (prob * target).sum(dim=[1, 2])
    union = prob.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

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
        mask_dir=f"{args.data_root}/labels/masks/train",
        transform=tf_img,
        target_transform=tf_mask
    )

    val_ds   = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/val",
        mask_dir=f"{args.data_root}/labels/masks/val",
        transform=tf_img,
        target_transform=tf_mask
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

# ─── AMP & scaler ─────────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp)

# Model ---------------------------------------------------------------------

    model = smp.Unet(encoder_name="resnet34",
                     encoder_weights="imagenet",
                     classes=2, activation=None).to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warm-up
        cosine_epoch = epoch - warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * cosine_epoch / (args.epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




    best_dice, start = 0.0, time.time()
    history = []

    patience, bad = 3, 0

    # Training loop -------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                out = model(imgs)
                loss = 0.6 * ce(out, masks) + 0.4 * soft_dice_loss(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        
        scheduler.step()

        # Validation ------------------------------------------------------------

        model.eval(); val_loss = 0.0; val_dice = 0.0

        with torch.no_grad():

            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss += (0.6 * ce(out, masks) + 0.4 * soft_dice_loss(out, masks)).item() * imgs.size(0)
                val_dice  += dice_coefficient(out, masks.float()).item() * imgs.size(0)

        val_loss /= len(val_ds)
        val_dice  /= len(val_ds)

        # Logging ---------------------------------------------------------------

        history.append({

            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice

        })

        print(f"[{epoch:03}/{args.epochs}] "

              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={val_dice:.4f}")

        # Checkpoint ------------------------------------------------------------

        if val_dice > best_dice:
            best_dice = val_dice
            bad = 0
            torch.save(model.state_dict(), f"{args.save_dir}/best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stop: no Dice improvement for 3 epochs.")
                break

    # Save training log ---------------------------------------------------------

    json.dump(history, open(f"{args.save_dir}/history.json","w"), indent=2)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min. Best Dice: {best_dice:.4f}")

if __name__ == "__main__":

    main()
