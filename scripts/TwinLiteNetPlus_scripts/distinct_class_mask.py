import numpy as np
import pathlib
import argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="image I'm testing")
ap2 = ap.parse_args()
mask = np.array(Image.open(pathlib.Path(ap2.image)))
print("Unique pixel values:", np.unique(mask))

