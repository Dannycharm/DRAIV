
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='torchscript', device='cpu')
# ➜ best.torchscript.ptj

