import sys

import torch

print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__)
print("CUDA OK:", torch.cuda.is_available())
print(
    "GPU    :",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
)
