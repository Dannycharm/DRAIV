import argparse
import os
import cv2
import time
import sys
from pathlib import Path
import torch
import numpy as np
import zmq
from skimage.morphology import thin      # or skeletonize
# Tracking and Detection
import supervision as sv
from ultralytics import YOLO


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# ----------------------------- CLI ---------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video path or camera index")
ap.add_argument("--weights", required=True, help=".pt checkpoint")
ap.add_argument("--out", default="demo_tracked_twin.avi",
                help="output file (extension sets container)")
ap.add_argument("--device", default="0", help="0, cuda:0, -1, cpu …")
args = ap.parse_args()

# ------------------------- Device Handling -------------------------
device = torch.device(f"cuda:{args.device}" if args.device.isdigit()
                      else args.device)
print("Using device:", device)

# --------------------- Model & Tracker Setup -----------------------
model = YOLO(args.weights, verbose=False).to(device)
tracker  = sv.ByteTrack()
box_anno = sv.BoxAnnotator()
lbl_anno = sv.LabelAnnotator()

# ----------------------- Video Input Check -------------------------
if not os.path.exists(args.source):
    raise FileNotFoundError(args.source)
info   = sv.VideoInfo.from_video_path(args.source)
frames = sv.get_video_frames_generator(args.source)
print("VideoInfo:", info)

# ---------------------- VideoSink with fallback --------------------
ext = os.path.splitext(args.out)[1].lower()
if   ext in (".mp4", ".m4v"): codec_pref = ["mp4v", "avc1", "H264"]
elif ext in (".avi",):        codec_pref = ["MJPG", "XVID"]
else: raise ValueError("Unsupported output extension; use .mp4 or .avi")

for fourcc in codec_pref:
    try:
        sink = sv.VideoSink(args.out, video_info=info, codec=fourcc)
        sink.__enter__()               # test writer immediately
        print(f"✓ Writing {args.out} with FOURCC '{fourcc}'")
        break
    except Exception:
        continue
else:
    raise RuntimeError(
        "OpenCV cannot encode the requested container; "
        "install opencv-python wheels or switch to .avi")

# ----------------------- Processing Loop ---------------------------
t0 = time.time()
for idx, frame in enumerate(frames, 1):

    # One resize (BGR)
    padded, ratio, pad = letterbox_for_img(frame, new_shape=(384, 640), auto=True)

    # Detect & Track
    det = sv.Detections.from_ultralytics(model(padded, device=args.device, verbose=False)[0])
    det = tracker.update_with_detections(det)
    labels = [f"{model.names[c]} #{tid}"
                for c, tid in zip(det.class_id, det.tracker_id)]

fps = idx / (time.time() - t0)
print(f"✓ Saved {args.out} · {idx} frames · {fps:.1f} FPS")

