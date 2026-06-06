"""
Interactive 4-point homography picker.

Click the four corners of a real-world rectangle whose dimensions you have
measured (e.g. the brick crosswalk strip). The script saves the four image
points plus the corresponding world coordinates into the chosen zones YAML
under a `homography:` block. The rest of the pipeline reads that block to
project detections onto a true bird's-eye-view ground plane.

Click order (looking at the image):
    1. NEAR-LEFT   — the corner closest to the camera, on the left
    2. NEAR-RIGHT  — closest to camera, on the right
    3. FAR-RIGHT   — farther from camera, on the right
    4. FAR-LEFT    — farther from camera, on the left

World axes:
    +X = "across" (peds walk this; --width)
    +Y = "along traffic" (depth from near to far; --depth)

Usage (the values 7.15 x 3.0 are the user's measured crosswalk):
    python scripts/homography_picker.py \
        --video data/IMG_5757.MOV --frame 60 \
        --config configs/zones_img5757.yaml \
        --width 7.15 --depth 3.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

WINDOW = ("Homography Picker  |  click 4 corners (near-L, near-R, far-R, far-L)  "
          "|  Enter=save  R=redo  Q=quit")
DOT_COLOR  = (0, 255, 80)
LINE_COLOR = (0, 220, 255)
LABELS = ["1 near-L", "2 near-R", "3 far-R", "4 far-L"]


def load_frame(video_path: str, frame_n: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_n, max(total - 1, 0)))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_n} from {video_path}")
    return frame


def render(base: np.ndarray, points: list) -> np.ndarray:
    img = base.copy()
    for i, pt in enumerate(points):
        cv2.circle(img, pt, 7, DOT_COLOR, -1)
        cv2.circle(img, pt, 7, (0, 0, 0), 1)
        cv2.putText(img, LABELS[i], (pt[0] + 9, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, DOT_COLOR, 2)
    if len(points) >= 2:
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], LINE_COLOR, 2)
    if len(points) == 4:
        cv2.line(img, points[3], points[0], LINE_COLOR, 2)
        poly = np.array(points, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], LINE_COLOR)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    next_label = LABELS[len(points)] if len(points) < 4 else "DONE — Enter to save"
    bar = f"  {len(points)}/4 corners  |  next: {next_label}  |  R=redo  Q=quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, bar, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def pick_corners(frame: np.ndarray) -> list | None:
    points: list = []

    def on_click(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, on_click)

    while True:
        cv2.imshow(WINDOW, render(frame, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord(" ")):
            if len(points) == 4:
                break
            print(f"Need 4 points — currently have {len(points)}.")
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return points


def save_homography(config_path: str, image_pts: list,
                    width_m: float, depth_m: float) -> None:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = cfg or {}

    world_pts = [
        [0.0,      0.0],
        [float(width_m), 0.0],
        [float(width_m), float(depth_m)],
        [0.0,      float(depth_m)],
    ]
    cfg["homography"] = {
        "image_points": [[int(p[0]), int(p[1])] for p in image_pts],
        "world_points": world_pts,
        "width_m": float(width_m),
        "depth_m": float(depth_m),
    }
    path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))

    print(f"\nSaved homography to {config_path}:")
    for label, img, wld in zip(LABELS, image_pts, world_pts):
        print(f"  {label:<10}  image=({img[0]:>5},{img[1]:>5})  "
              f"world=({wld[0]:.2f}, {wld[1]:.2f}) m")


def main():
    p = argparse.ArgumentParser(description="Interactive 4-point homography picker")
    p.add_argument("--video",  default="data/IMG_5757.MOV")
    p.add_argument("--frame",  type=int, default=60)
    p.add_argument("--config", default="configs/zones_img5757.yaml")
    p.add_argument("--width",  type=float, required=True,
                   help='Real-world width in metres ("across" — peds walk this)')
    p.add_argument("--depth",  type=float, required=True,
                   help='Real-world depth in metres (along traffic direction)')
    args = p.parse_args()

    print(f"Loading frame {args.frame} from {args.video} ...")
    frame = load_frame(args.video, args.frame)

    print("Click the 4 corners of your measured rectangle in this order:")
    for lab in LABELS:
        print(f"  {lab}")
    print(f"World rectangle: {args.width:.2f} m wide x {args.depth:.2f} m deep")
    print("  Enter/Space — save    R — redo    Q/Esc — quit without saving")

    points = pick_corners(frame)
    if points is None:
        print("Cancelled — config unchanged.")
        sys.exit(0)
    save_homography(args.config, points, args.width, args.depth)


if __name__ == "__main__":
    main()
