"""
Measure a known real-world distance in pixels, then compute meters_per_pixel.

Usage:
    python scripts/pixel_ruler.py --video data/crosswalk.mp4 --real-meters 7.25

Click the two endpoints of the known distance (e.g., both edges of the crosswalk).
Press Enter to confirm and print the result.
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path

WINDOW = "Pixel Ruler  |  click 2 points  |  Enter=confirm  R=redo  Q=quit"


def load_frame(video_path: str, frame_n: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_n}")
    return frame


def render(base: np.ndarray, points: list) -> np.ndarray:
    img = base.copy()
    for pt in points:
        cv2.circle(img, pt, 7, (0, 255, 80), -1)
        cv2.circle(img, pt, 7, (0, 0, 0), 1)
    if len(points) == 2:
        cv2.line(img, points[0], points[1], (0, 220, 255), 2)
        px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        mid = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
        cv2.putText(img, f"{px_dist:.0f} px", (mid[0] + 8, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    label = f"  {len(points)}/2 points  |  Enter=confirm  R=redo  Q=quit"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return img


def measure(frame: np.ndarray) -> tuple | None:
    points = []

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, on_click)

    while True:
        cv2.imshow(WINDOW, render(frame, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord(" ")):
            if len(points) == 2:
                break
            print("Click exactly 2 points first.")
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return tuple(points)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",        default="data/crosswalk.mp4")
    p.add_argument("--frame",        type=int, default=600)
    p.add_argument("--real-meters",  type=float, required=True,
                   help="Real-world distance between the two points (metres)")
    p.add_argument("--config",       default="configs/zones.yaml")
    args = p.parse_args()

    frame = load_frame(args.video, args.frame)
    result = measure(frame)
    if result is None:
        print("Cancelled.")
        return

    p1, p2 = result
    px_dist = float(np.linalg.norm(np.array(p1) - np.array(p2)))
    mpp = args.real_meters / px_dist

    print(f"\nPixel distance : {px_dist:.1f} px")
    print(f"Real distance  : {args.real_meters} m")
    print(f"meters_per_pixel = {mpp:.5f}")

    path = Path(args.config)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["risk"]["meters_per_pixel"] = round(mpp, 5)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved meters_per_pixel = {mpp:.5f} to {args.config}")


if __name__ == "__main__":
    main()
