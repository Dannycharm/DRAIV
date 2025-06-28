#!/usr/bin/env python3
"""
Convert BDD100K detection JSON → YOLO TXT (multiprocessing version)

Example:
  python bdd100k_json_to_yolo.py \
      --json     /datasets/bdd100k/labels/det_20/det_train.json \
      --img-root /datasets/bdd100k/images/100k/train \
      --txt-root /datasets/bdd100k/labels/100k/train \
      --workers 4
"""

import argparse, json, pathlib, functools, multiprocessing as mp
import tqdm

# ----------------------------------------------------------------------
# 1. Category → class-index map (edit if you prune classes)
# ----------------------------------------------------------------------
CLS2ID = {
    "car": 0,
    "traffic sign": 1,
    "traffic light": 2,
    "pedestrian": 3,
    "truck": 4,
    "bus": 5,
    "bicycle": 6,
    "rider": 7,
    "motorcycle": 8,
    "other vehicle": 9,
    "other person": 10,
    "train": 11,
    "trailer": 12,
}

# ----------------------------------------------------------------------
# 2. Per-frame worker function  (runs in each process)
# ----------------------------------------------------------------------
def _one_frame(frame, img_root: pathlib.Path, txt_root: pathlib.Path) -> int:
    """Convert one BDD frame dict → YOLO txt file. Return 1 if written."""
    img_name = frame["name"].replace(".jpg", "")
    img_path = img_root / f"{img_name}.jpg"
    txt_file = txt_root / f"{img_name}.txt"

    # Skip if missing image or label already done (idempotent/resumable)
    if not img_path.exists() or txt_file.exists():
        return 0

    # Default resolution = 1280×720 if key absent
    w, h = frame.get("attributes", {}).get("resolution", [1280, 720])

    lines = []
    for obj in frame.get("labels", []):
        cls = obj["category"]
        if cls not in CLS2ID:
            continue
        box = obj["box2d"]
        xc = (box["x1"] + box["x2"]) / 2 / w
        yc = (box["y1"] + box["y2"]) / 2 / h
        bw = (box["x2"] - box["x1"]) / w
        bh = (box["y2"] - box["y1"]) / h
        lines.append(f"{CLS2ID[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        txt_file.write_text("\n".join(lines))
        return 1
    return 0

# ----------------------------------------------------------------------
# 3. Main conversion routine
# ----------------------------------------------------------------------
def convert(args):
    txt_root = pathlib.Path(args.txt_root)
    txt_root.mkdir(parents=True, exist_ok=True)

    with open(args.json) as f:
        items = json.load(f)

    fn = functools.partial(
        _one_frame,
        img_root=pathlib.Path(args.img_root),
        txt_root=txt_root,
    )

    with mp.Pool(args.workers) as pool:
        total_written = sum(
            tqdm.tqdm(pool.imap_unordered(fn, items), total=len(items))
        )

    print(f"✅ Done. Wrote {total_written:,} label files to {txt_root}")

# ----------------------------------------------------------------------
# 4. CLI entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",     required=True, help="det_train.json or det_val.json")
    ap.add_argument("--img-root", required=True, help="directory with *.jpg images")
    ap.add_argument("--txt-root", required=True, help="output directory for YOLO txt")
    ap.add_argument("-w", "--workers", type=int, default=4,
                    help="CPU processes to use (default 4)")
    convert(ap.parse_args())

