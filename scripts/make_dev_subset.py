import argparse, pathlib, random, shutil

def main(src, dst, n):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = list(src.glob("*.jpg"))
    sample = random.sample(imgs, n)

    for f in sample:
        shutil.copy2(f, dst / f.name)

    print(f"Copied {n} images from {src} → {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="source image dir")
    p.add_argument("--dst", required=True, help="destination dir")
    p.add_argument("-n", type=int, default=1000, help="#images")
    args = p.parse_args()
    main(args.src, args.dst, args.n)

