"""Quick frame extractor — pulls exact frames out of a rendered mp4 for verification."""
import argparse
import cv2
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frames", nargs="+", type=int, required=True,
                   help="Exact frame numbers to extract")
    p.add_argument("--outdir", default="outputs/frames_v15_1")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")

    src = Path(args.video).stem
    for fno in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            print(f"skip frame {fno} (read failed)")
            continue
        out_path = Path(args.outdir) / f"{src}_frame{fno:05d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"saved {out_path}  shape={frame.shape}")
    cap.release()


if __name__ == "__main__":
    main()
