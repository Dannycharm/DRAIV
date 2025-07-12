import argparse, os, time, json, math, cv2
from tqdm import tqdm
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
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
def squeeze_and_long(x, **kwargs):
        return x.squeeze(0).long()


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data ----------------------------------------------------------------------
    tf_img = A.Compose([
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

    train_ds = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/train",
        mask_dir=f"{args.data_root}/labels/masks/train",
        transform=tf_img,
        target_transform=None
    )

    val_ds   = UNetSegmentationDataset(
        image_dir=f"{args.data_root}/images/val",
        mask_dir=f"{args.data_root}/labels/masks/val",
        transform=tf_img,
        target_transform=None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

# ─── AMP & scaler ─────────────────────────────────────────────
    scaler = GradScaler('cuda', enabled=args.amp)

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



    print("Starting debug...")
    imgs, masks = next(iter(train_loader))
    print("Loaded batch")
    print(torch.unique(masks), masks.dtype)
    print("image shape is:", imgs.shape)
    print("mask shape is:", masks.shape)
    if scaler.is_enabled():
        print("AMP is on")

if __name__ == "__main__":
    main()
