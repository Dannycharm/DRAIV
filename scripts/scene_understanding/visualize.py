
# This is used for visualizing the lane and drivable area segmentation masks and detection annotations on videos 

import argparse
import os
import cv2
import time
import sys
from pathlib import Path
import torch
import numpy as np

# Tracking and Detection
import supervision as sv
from ultralytics import YOLO

# Segmentation Model Imports
parent_dir = (Path(__file__).parent.parent / 'TwinLiteNetPlus_scripts').resolve()
sys.path.append(str(parent_dir))
from model.model import TwinLiteNetPlus

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

def show_seg_result(img, result, palette=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [255, 0, 0]
    color_seg = color_area[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img = img.astype(np.float32)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return img

@torch.no_grad()
def segment_frame(frame, args, model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = TwinLiteNetPlus(args).to(device)
        if getattr(args, 'half', True):
            model.half()
        model.load_state_dict(torch.load(args.weight, map_location=device))
        model.eval()
    img_bgr = frame.copy()
    img, ratio, pad = letterbox_for_img(img_bgr, new_shape=(384, 640), auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    img = img.half() / 255.0 if getattr(args, 'half', True) else img.float() / 255.0
    da_seg_out, ll_seg_out = model(img)
    _, _, H, W = img.shape
    pad_w, pad_h = map(int, pad)
    da_pred = da_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]
    ll_pred = ll_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]
    da_mask = torch.nn.functional.interpolate(da_pred, scale_factor=1/ratio[0], mode='bilinear')
    ll_mask = torch.nn.functional.interpolate(ll_pred, scale_factor=1/ratio[0], mode='bilinear')
    da_mask = torch.argmax(da_mask, 1).int().squeeze().cpu().numpy()
    ll_mask = torch.argmax(ll_mask, 1).int().squeeze().cpu().numpy()
    return da_mask, ll_mask, img_bgr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/datasets/videos/0000f77c-6257be58.mp4", help="video path or camera index")
    parser.add_argument("--yolo_weight", type=str, default="/workspace/models/yolo_models/best.pt", help="YOLO .pt checkpoint")
    parser.add_argument("--seg_weight", type=str, default="/workspace/models/TwinLiteNet_models/large.pth", help="Segmentation model .pth")
    parser.add_argument("--out", type=str, default="demo_scene.avi")
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

    # YOLO and Tracking Setup
    model = YOLO(args.yolo_weight).to(args.device)
    tracker  = sv.ByteTrack()
    box_anno = sv.BoxAnnotator()
    lbl_anno = sv.LabelAnnotator()

    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)
    info   = sv.VideoInfo.from_video_path(args.source)
    frames = sv.get_video_frames_generator(args.source)
    ext = os.path.splitext(args.out)[1].lower()
    if   ext in (".mp4", ".m4v"): codec_pref = ["mp4v", "avc1", "H264"]
    elif ext in (".avi",):        codec_pref = ["MJPG", "XVID"]
    else: raise ValueError("Unsupported output extension; use .mp4 or .avi")

    for fourcc in codec_pref:
        try:
            sink = sv.VideoSink(args.out, video_info=info, codec=fourcc)
            sink.__enter__()
            print(f"✓ Writing {args.out} with FOURCC '{fourcc}'")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("OpenCV cannot encode the requested container; install opencv-python wheels or switch to .avi")

    t0 = time.time()
    for idx, frame in enumerate(frames, 1):
        # Detect & Track
        det = sv.Detections.from_ultralytics(model(frame, device=args.device, verbose=False)[0])
        det = tracker.update_with_detections(det)
        labels = [f"{model.names[c]} #{tid}"
                  for c, tid in zip(det.class_id, det.tracker_id)]

        # Segmentation
        da_mask, ll_mask, img_bgr = segment_frame(frame, seg_model_args, model=seg_model)

        # Visualize segmentation
        vis = show_seg_result(img_bgr, (da_mask, ll_mask))
        vis = box_anno.annotate(vis, det)
        vis = lbl_anno.annotate(vis, det, labels)
        sink.write_frame(vis)

    sink.__exit__(None, None, None)
    fps = idx / (time.time() - t0)
    print(f"✓ Saved {args.out} · {idx} frames · {fps:.1f} FPS")

if __name__ == "__main__":
    main()



