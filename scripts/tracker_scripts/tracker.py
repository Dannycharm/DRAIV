"""
track_and_annotate.py
 Supervision v0.24.0
---------------------
Run:
    python track_and_annotate.py \
        --source demo.mp4 \
        --weights /scratch/dannycharm-alt-REU/DRAIV/models/yolo_models/best.pt \
        --out out.mp4 \
        --device cuda:0            # 0 = first CUDA GPU, -1 = CPU
"""

import argparse, time, os, cv2, torch, supervision as sv
from ultralytics import YOLO

# ----------------------------- CLI ---------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video path or camera index")
ap.add_argument("--weights", required=True, help=".pt checkpoint")
ap.add_argument("--out", default="demo_tracked.avi",
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
    det = sv.Detections.from_ultralytics(model(frame, device=device)[0])
    det = tracker.update_with_detections(det)
    labels = [f"{model.names[c]} #{tid}"
              for c, tid in zip(det.class_id, det.tracker_id)]
    frame = box_anno.annotate(frame.copy(), det)
    frame = lbl_anno.annotate(frame, det, labels)
    sink.write_frame(frame)

sink.__exit__(None, None, None)
fps = idx / (time.time() - t0)
print(f"✓ Saved {args.out} · {idx} frames · {fps:.1f} FPS")
