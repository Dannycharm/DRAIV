import numpy as np
from PIL import Image

m = np.array(Image.open("/scratch/dannycharm-alt-REU/DRAIV/datasets/unet/processed_no_reshape/labels/masks/train/0000f77c-62c2a288.png"))
print(np.unique(m), m.sum())   # should print [0 1]  and a non-zero sum

bright = (m * 255).astype(np.uint8)
Image.fromarray(bright).save("mask_vis.png")

