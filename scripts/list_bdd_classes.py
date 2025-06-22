import json, pathlib, collections, argparse

def main(label_root):
    counter = collections.Counter()
    json_files = sorted(pathlib.Path(label_root).rglob("*.json"))
    assert json_files, f"No JSON files found in {label_root}"
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        for frame in data:
            for obj in frame.get("labels", []):
                counter[obj["category"]] += 1

    print("\n=== BDD100K detection classes ===")
    for i, (cls, n) in enumerate(counter.most_common()):
        print(f"{i:2d}  {cls:<20}  boxes={n:,}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Path to labels/det_20/train or .../val directory")
    args = p.parse_args()
    main(args.root)
''' 
Output I got:
=== BDD100K detection classes ===
 0  car                   boxes=803,540
 1  traffic sign          boxes=272,994
 2  traffic light         boxes=214,755
 3  pedestrian            boxes=105,584
 4  truck                 boxes=32,135
 5  bus                   boxes=13,637
 6  bicycle               boxes=8,163
 7  rider                 boxes=5,218
 8  motorcycle            boxes=3,483
 9  other vehicle         boxes=889
10  other person          boxes=211
11  train                 boxes=143
12  trailer               boxes=73 '''
