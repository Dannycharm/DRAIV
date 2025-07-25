# This runs in real time, produces symbolic facts from each frame such as object detections + ByteTrack IDs + segmentation masks. This is also used for visualizing the lane and drivable area segmentation masks and detection annotations on videos

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

# Segmentation Model Imports
parent_dir = (Path(__file__).parent.parent / 'TwinLiteNetPlus_scripts').resolve()
sys.path.append(str(parent_dir))
from model.model import TwinLiteNetPlus

#  JSON publishing stub (ZeroMQ)
def setup_pub(port: int = 5555):
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{port}")
    return pub


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
#--------------------------------------------------------------------------

# def get_lane_centerline(lane_mask: np.ndarray) -> np.ndarray:
#     """
#     Args
#     ----
#     lane_mask : np.uint8 HxW   # 0 background, 1 lane pixels

#     Returns
#     -------
#     centerline : np.uint8 HxW  # skeletonised mask (0/255)
#     """
#     # Ensure binary bool image for scikit-image
#     lane_bool = lane_mask.astype(bool)

#     # Zhang-Suen thinning; max_num_iter=np.inf == full skeleton
#     skel_bool = thin(lane_bool, max_num_iter=np.inf)

#     # Back to the format your downstream expects (0/255 uint8)
#     centerline = (skel_bool * 255).astype(np.uint8)
#     return centerline

# def compute_lane_offset(lane_mask: np.ndarray,
#                         bottom_rows: int = 20) -> float | None:
#     skel = get_lane_centerline(lane_mask)
#     # Consider only the rows closest to the ego vehicle
#     cols = np.where(skel[-bottom_rows:, :] > 0)[1]
#     if len(cols) < 2:          # not enough skeleton pixels visible
#         return None
#     lane_center = int(cols.mean())
#     img_center  = lane_mask.shape[1] // 2
#     # + offset  => drift right,  – offset => drift left
#     return img_center - lane_center

#--------------------------------------------------------------------------

def main():

    #  JSON publishing stub
    pub = setup_pub()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/datasets/videos/0000f77c-6257be58.mp4", help="video path or camera index")
    parser.add_argument("--yolo_weight", type=str, default="/workspace/models/yolo_models/best.pt", help="YOLO .pt checkpoint")
    parser.add_argument("--seg_weight", type=str, default="/workspace/models/TwinLiteNet_models/large.pth", help="Segmentation model .pth")
    parser.add_argument("--out", type=str, default="demo_scene_6.avi")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hyp", type=str, default="/workspace/scripts/TwinLiteNetPlus_scripts/hyperparameters/twinlitev2_hyper.yaml")
    parser.add_argument("--config", default="large")
    args = parser.parse_args()

    # Segmentation model, pre-load once
    seg_model_args = argparse.Namespace(**vars(args))
    seg_model_args.weight = args.seg_weight
    seg_model = TwinLiteNetPlus(seg_model_args).to(args.device)
    seg_model.half()
    seg_model.load_state_dict(torch.load(seg_model_args.weight, map_location=args.device))
    seg_model.eval()

    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)
    info   = sv.VideoInfo.from_video_path(args.source)
    frames = sv.get_video_frames_generator(args.source)
    print("The video info is: ", info)
    info.width = 640
    info.height = 384
    print("The video info after change is: ", info)

    t0 = time.time()
    for idx, frame in enumerate(frames, 1):

         # One resize (BGR)
        padded, ratio, pad = letterbox_for_img(frame, new_shape=(384, 640), auto=True)

        padded_rgb = padded[:, :, ::-1].transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(padded_rgb).to(seg_model_args.device).unsqueeze(0)
        tensor = tensor.half() / 255.0 if getattr(seg_model_args, 'half', True) else tensor.float() / 255.0
        da_seg_out, ll_seg_out = seg_model(tensor)

        _, _, H, W = tensor.shape
        pad_w, pad_h = map(int, pad)
        da_pred = da_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]
        ll_pred = ll_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]
        da_mask = torch.nn.functional.interpolate(da_pred, scale_factor=1/ratio[0], mode='bilinear')
        ll_mask = torch.nn.functional.interpolate(ll_pred, scale_factor=1/ratio[0], mode='bilinear')
        da_mask = torch.argmax(da_mask, 1).int().squeeze().cpu().numpy()
        ll_mask = torch.argmax(ll_mask, 1).int().squeeze().cpu().numpy()

        # # Compute lane offset using the lane line mask (ll_mask)
        # lane_mask = ll_mask.astype(np.uint8)  # ensure it's binary 0/1 as uint8

        # # Skeletonize the lane mask to get thin lines and get lane offset
        # lane_offset_px =  compute_lane_offset(lane_mask, 180)

        # # Compute drivable area ratio
        # drivable_mask = da_mask  # (already a binary 0/1 array)
        # drivable_ratio = float(drivable_mask.sum() / drivable_mask.size)

    fps = idx / (time.time() - t0)
    print(f"✓ Saved {args.out} · {idx} frames · {fps:.1f} FPS")

if __name__ == "__main__":
    main()
