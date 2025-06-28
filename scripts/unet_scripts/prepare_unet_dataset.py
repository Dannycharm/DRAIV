# This Python script will:
# Resize images and masks
# Remap pixel values in masks (e.g., 4, 6, 22 → 1; 255 → 0)
# Save resized, remapped results to output folders
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# === CONFIG ===
INPUT_IMG_DIR = '/datasets/bddk100k/bdd100k/images/100k/val/'
INPUT_MASK_DIR = '/datasets/bddk100k/bdd100k/labels/lane/masks/val/'
OUT_IMG_DIR = '/datasets/unet/processed/images/val'
OUT_MASK_DIR = '/datasets/unet/processed/labels/masks/val'

#And when I process your validation set, I can use:
#   INPUT_IMG_DIR = '/datasets/bddk100k/bdd100k/images/100k/val/'
#   INPUT_MASK_DIR = '/datasets/bddk100k/bdd100k/labels/lane/masks/val/'
#   OUT_IMG_DIR = '/datasets/unet/processed/images/val'
#   OUT_MASK_DIR = '/datasets/unet/processed/labels/masks/val'

RESIZE_SHAPE = (512, 512)

# === CREATE OUTPUT FOLDERS ===
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# === GET ALL IMAGES ===
image_filenames = [f for f in os.listdir(INPUT_IMG_DIR) if f.endswith('.jpg')]

# === RESUME BY SKIPPING ALREADY PROCESSED FILES ===
already_processed = set(os.listdir(OUT_IMG_DIR))
to_process = [f for f in image_filenames if f not in already_processed]

for fname in tqdm(to_process, desc="Resuming image/mask processing"):
    img_path = os.path.join(INPUT_IMG_DIR, fname)
    mask_path = os.path.join(INPUT_MASK_DIR, fname.replace('.jpg', '.png'))

    if not os.path.exists(mask_path):
        print(f"Skipping {fname}, no corresponding mask.")
        continue

    # Resize and save image
    img = Image.open(img_path).convert('RGB').resize(RESIZE_SHAPE, Image.BILINEAR)
    img.save(os.path.join(OUT_IMG_DIR, fname))

    # Resize and save remapped mask
    raw_mask = np.array(Image.open(mask_path))
    remapped = np.where(raw_mask == 255, 0, 1).astype(np.uint8)
    resized_mask = Image.fromarray(remapped).resize(RESIZE_SHAPE, Image.NEAREST)
    resized_mask.save(os.path.join(OUT_MASK_DIR, fname.replace('.jpg', '.png')))


