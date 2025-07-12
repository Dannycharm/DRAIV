# This will load the processed .jpg images and remapped .png masks.

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

def is_albumentations_transform(transform):
    return isinstance(transform, A.BasicTransform) or isinstance(transform, A.Compose)

class UNetSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            if is_albumentations_transform(self.transform):
                augmented = self.transform(image=np.array(image), mask=np.array(mask))
                image = augmented['image']
                mask = augmented['mask']
            else:  # Assume torchvision or similar
                image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

