import argparse, glob, os
from src.perf_viz import make_montage

def main():
    ap = argparse.ArgumentParser(description="Combine perf PNGs into a single montage.")
    ap.add_argument("folder", help="Folder with PNGs (e.g., perf/plots_train)")
    ap.add_argument("--cols", type=int, default=2, help="Number of columns")
    ap.add_argument("--out",  default="perf/montage.png", help="Output image path")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.folder, "*.png")))
    make_montage(imgs, args.out, cols=args.cols)

if __name__ == "__main__":
    main()
