# This will load the processed .jpg images and remapped .png masks.

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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

            image = self.transform(image)

        if self.target_transform:

            mask = self.target_transform(mask)

        return image, mask

# === Usage Example ===

if __name__ == "__main__":

    transform = T.Compose([

        T.ToTensor(),

        T.Normalize(mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225])

    ])

    target_transform = T.Compose([

        T.PILToTensor(),           # Returns ByteTensor (H, W)

        lambda x: x.squeeze(0).long()  # Remove channel dimension, convert to Long

    ])

    dataset = UNetSegmentationDataset(

        image_dir='./processed/images',

        mask_dir='./processed/masks',

        transform=transform,

        target_transform=target_transform

    )

    img, mask = dataset[0]

    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}, Mask unique values: {torch.unique(mask)}")

