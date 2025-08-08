#Perception.py

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
    pub.setsockopt(zmq.SNDHWM, 10000)
    pub.bind(f"tcp://127.0.0.1:{port}")
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

def compute_lane_offset(lane_mask: np.ndarray, bottom_rows: int = 20):
    lane_cols = np.where(lane_mask[-bottom_rows:, :] > 0)[1]
    # If detection failed
    if len(lane_cols) == 0:
        return None

    lane_width = max(lane_cols) - min(lane_cols)
    lane_center = int(lane_cols.mean())
    img_center = lane_mask.shape[1] // 2
    
    offset = (img_center - lane_center)
    return offset, lane_center, lane_width
#--------------------------------------------------------------------------

def main():
    time.sleep(2)  # Wait for subscriber to connect for 15 seconds
    # write output
    log_file = open("log_perception.txt", 'w')

    #  JSON publishing stub
    pub = setup_pub()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/datasets/videos/unsafe_distance/dashcam_video_one.mp4", help="video path or camera index")
    parser.add_argument("--yolo_weight", type=str, default="/workspace/models/yolo_models/best.pt", help="YOLO .pt checkpoint")
    parser.add_argument("--seg_weight", type=str, default="/workspace/models/TwinLiteNet_models/large.pth", help="Segmentation model .pth")
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
    tracker  = sv.ByteTrack(lost_track_buffer=120, minimum_matching_threshold=0.7, )
    box_anno = sv.BoxAnnotator()
    lbl_anno = sv.LabelAnnotator(text_scale=0.2, text_thickness=0, text_padding=5)

    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)
    info   = sv.VideoInfo.from_video_path(args.source)
    frames = sv.get_video_frames_generator(args.source)

    t0 = time.time()
    for idx, frame in enumerate(frames):

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

        # Detect & Track
        det = sv.Detections.from_ultralytics(model(frame, device=args.device, verbose=False)[0])
        det = tracker.update_with_detections(det)
        labels = [f"{model.names[c]} #{tid}"
                  for c, tid in zip(det.class_id, det.tracker_id)]

        # Compute lane offset using the lane line mask (ll_mask)
        lane_mask = ll_mask.astype(np.uint8)  # ensure it's binary 0/1 as uint8

        # lane offset
        lane_result = compute_lane_offset(lane_mask, bottom_rows=40)
        if lane_result is None:
            lane_offset_px, lane_center, lane_width = None, None, None
        else:
            lane_offset_px, lane_center, lane_width = lane_result
            lane_width = int(lane_width) 

        # Compute drivable area ratio
        drivable_mask = da_mask  # (already a binary 0/1 array)
        drivable_ratio = float(drivable_mask.sum() / drivable_mask.size)

        # vx
        if not hasattr(main, "prev_centroids"):
            main.prev_centroids = {}

        centroids = det.xyxy[:, :4].copy()
        centroids = (centroids[:, 0:2] + centroids[:, 2:4]) / 2   # (x,y)


        # Prepare object list from detections
        objects_list = []
        for (bbox, cls_id, track_id, (cx,cy)) in zip(det.xyxy, det.class_id, det.tracker_id, centroids):
            x1, y1, x2, y2 = map(int, bbox)
            class_name = model.names[int(cls_id)]
            obj_id = int(track_id) if track_id is not None else None

            # vx is a raw measure of how many pixels an object shifts left or right from one video frame to the next.
            vx = None
            if track_id is not None:
                if track_id in main.prev_centroids:
                    dx = cx - main.prev_centroids[track_id][0]
                    # vx in pixels/frame
                    vx = float(dx)
                main.prev_centroids[track_id] = (cx, cy)
            cx, cy = float(cx), float(cy)

            # This checks if a pedestrian is within the overall boundary of the drivable area or lane boundary
            if class_name == 'pedestrian':
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)
                    # clamp indices to image size
                    h, w = lane_mask.shape
                    foot_x = np.clip(foot_x, 0, w-1)
                    foot_y = np.clip(foot_y, 0, h-1)
                    is_on_lane = lane_mask[foot_y, foot_x] > 0
                    is_on_drivable = da_mask[foot_y, foot_x] > 0
                    if is_on_lane or is_on_drivable:
                        class_name = 'pedestrian_on_road'

            objects_list.append({
                "id": obj_id,
                "cls": class_name,
                "bbox": [x1, y1, x2, y2],
                "vx": vx,
                "object_center_coord": [cx, cy]
            })

        # Create and send/print JSON message
        msg = {
            "t": idx,
            "lane_center": lane_center,
            "ego_lane_offset_px": lane_offset_px,
            "lane_width": lane_width,
            "drivable_ratio": drivable_ratio,
            "objects": objects_list
        }

        # publish to context engine 
        pub.send_json(msg)

        # Print the message for verification
        # print(msg)

        # save log to file
        log_file.write(f"{msg}")
        log_file.flush()

    fps = idx / (time.time() - t0)
    print(f"✓ {idx+1} frames · {fps:.1f} FPS")
    log_file.close()

if __name__ == "__main__":
    main()
