#Perception.py

# This runs in real time, produces symbolic facts from each frame such as object detections + ByteTrack IDs + segmentation masks. This is also used for visualizing the lane and drivable area segmentation masks and detection annotations on videos

import argparse
import os
import cv2
import time
import json
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import zmq
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


def main():
    time.sleep(1)  # Wait for subscriber to wait
    # write output
    log_file = open("log_perception.txt", 'w')

    #  JSON publishing stub
    pub = setup_pub()
    # if using default arguments
    sample_version = "two"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=f"/datasets/videos/unsafe_distance/dashcam_video_{sample_version}.mp4", help="video path or camera index")
    parser.add_argument("--yolo_weight", type=str, default="/workspace/models/yolo_models/best.pt", help="YOLO .pt checkpoint")
    parser.add_argument("--seg_weight", type=str, default="/workspace/models/TwinLiteNet_models/large.pth", help="Segmentation model .pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hyp", type=str, default="/workspace/scripts/TwinLiteNetPlus_scripts/hyperparameters/twinlitev2_hyper.yaml")
    parser.add_argument("--ego_speed_source", type=str, default=f"/datasets/videos/unsafe_distance/ego_speed/ego_speed_{sample_version}.txt", help="The file containing the ego speed for each frame")
    parser.add_argument("--thw", type=str, default=f"/datasets/videos/unsafe_distance/thw/thw_{sample_version}.txt" , help="The file containing the fround truth time head way for each frame")
    parser.add_argument("--config", default="large")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)
    info   = sv.VideoInfo.from_video_path(args.source)
    frames = sv.get_video_frames_generator(args.source)


    # ego frame speed
    ego_speeds = pd.read_csv(args.ego_speed_source)
    # thw
    thw = pd.read_csv(args.thw)

    # Assert that the lengths are equal. If not, raise an error with the given message.
    assert len(ego_speeds) == info.total_frames and len(thw) == info.total_frames, "Frame count must be the same as the number of ego speed and thw entries."

    t0 = time.time()
    for (idx, frame), ego_speed, thw in zip(enumerate(frames), ego_speeds['ego_speed_mps'], thw['thw']):
        seen_frame = time.time()
        objects_list = []
    
        # Create and send/print JSON message
        msg = {
            "t": idx,
            "seen_time": seen_frame,
            "ego_speed_mps": ego_speed,
            "objects": objects_list,
            "thw": thw
        }

        # publish to context engine
        pub.send_json(msg)

        # Print the message for verification
        # print(msg)

        # Throttle to target frame duration for ~30 FPS
        elapsed = time.time() - seen_frame
        frame_duration = 0.03333333 # 1 / 30  # 0.0333 seconds per frame
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)

        fps = idx / (time.time() - t0)

    print(f"✓ {idx+1} frames · {fps:.1f} FPS")
    log_file.close()

if __name__ == "__main__":
    main()
