"""
Train a 2-class U-Net (lane vs background) on the pre-processed BDD100K dataset.

Usage
-----
python train_unet.py \
  --data_root /datasets/unet/processed_no_reshape \
  --epochs 40 \
  --batch_size 4 \
  --accum_steps 2 \
  --lr 5e-4 \
  --amp \
  --save_dir runs/unet_lane_full

"""

import argparse, os, time, json, math, cv2
from tqdm import tqdm
from pathlib import Path
import torch, torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp   # pip install segmentation-models-pytorch
from unet_dataset import UNetSegmentationDataset

# ────────────────────────────────────────────────────────────────────────────────

def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="processed/ directory with images/{train,val} and masks/{train,val}")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", default="runs/unet_lane")
    p.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision (AMP) training")
    p.add_argument("--accum_steps", type=int, default=2,
               help="number of mini-batches to accumulate")
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

def focal_loss(logit, target, gamma=2.0, alpha=0.25):

    prob = logit.sigmoid()
    pt   = torch.where(target==1, prob, 1-prob)
    w    = alpha*target + (1-alpha)*(1-target)
    return (w * (1-pt).pow(gamma) *

            binary_cross_entropy_with_logits(logit, target.float(),

                                             reduction='none')).mean()

def total_loss(out, target):

    logit_lane = out[:,1]              # lane channel
    dice  = soft_dice_loss(out, target)
    foc   = focal_loss(logit_lane, target, alpha=0.75)
    return 0.6*dice + 0.4*foc


# ────────────────────────────────────────────────────────────────────────────────
def squeeze_and_long(x, **kwargs):
        return x.squeeze(0).long()
# ────────────────────────────────────────────────────────────────────────────────

def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    # =========================================================
    # 1.  Cropping parameters
    # =========================================================
    FULL_H, FULL_W = 720, 1280          # original BDD frame size
    CROP_TOP_FRAC  = 0.30               # cut away 30 % of the height
    crop_y = int(FULL_H * CROP_TOP_FRAC)   # 216 pixels

    # Data ----------------------------------------------------------------------
    tf_img = A.Compose([
        A.Crop(x_min=0, x_max=FULL_W,
           y_min=crop_y, y_max=FULL_H),   # remove class imbalance caused by sky
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.92, 1.08),                 # ±8 % like before
            translate_percent=(-0.02, 0.02),    # ±2 % shift
            rotate=(-2, 2),                     # ±2°
            shear=(-2, 2),                      # tiny shear
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            fit_output=False,                   # keeps 1280 × 720
            fill=0, fill_mask=0,                # same as fill / fill_mask
            p=0.7
        ),
        A.RandomBrightnessContrast(0.2,0.2,p=0.5),
        A.GaussianBlur(blur_limit=3,p=0.2),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2(),
        A.Lambda(mask=squeeze_and_long)
    ])

    # ---------- validation aug (deterministic) ----------
    val_tf = A.Compose([
        A.Crop(x_min=0, x_max=FULL_W,
           y_min=crop_y, y_max=FULL_H),# remove class imbalance caused by sky
        A.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ToTensorV2(),
        A.Lambda(mask=squeeze_and_long)
    ])

    train_ds = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/train",
        mask_dir=f"{args.data_root}/labels/masks/train",
        transform=tf_img,
        target_transform=None
    )

    val_ds   = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/val",
        mask_dir=f"{args.data_root}/labels/masks/val",
        transform=val_tf,
        target_transform=None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

# ─── AMP & scaler ─────────────────────────────────────────────
    scaler = GradScaler('cuda', enabled=args.amp)

# Model ---------------------------------------------------------------------

    model = smp.Unet(encoder_name="resnet101",
                     encoder_weights="imagenet",
                     classes=2, activation=None).to(device)

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
        optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")):
            imgs, masks = imgs.to(device), masks.to(device)
            
            with autocast("cuda", enabled=args.amp):
                out = model(imgs)
                batch_loss = total_loss(out, masks)  # also calculates mean over batch
            loss = batch_loss / args.accum_steps          # scale loss for accumulation
            scaler.scale(loss).backward()
            running_loss += batch_loss.item() * imgs.size(0)

        # ----- step / update every accum_steps -----
            if (i + 1) % args.accum_steps == 0 or (i+1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = running_loss / len(train_ds)
        scheduler.step()

        # Validation ------------------------------------------------------------

        model.eval(); val_loss = 0.0; val_dice = 0.0

        with torch.no_grad():

            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss += total_loss(out, masks) * imgs.size(0)
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
