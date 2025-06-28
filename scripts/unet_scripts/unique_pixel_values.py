# This Will Scan The Entire Mask Dataset for Unique Values
# Do this for the train and val data
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

mask_dir = '/datasets/bddk100k/bdd100k/labels/lane/masks/val/'
all_values = set()

for fname in tqdm(os.listdir(mask_dir), desc="Scanning mask values"):
    if fname.endswith('.png'):
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        unique_vals = np.unique(mask)
        all_values.update(unique_vals)

print("All unique pixel values in dataset:", sorted(all_values))

