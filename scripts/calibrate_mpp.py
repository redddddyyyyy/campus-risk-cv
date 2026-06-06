"""
Estimates meters_per_pixel from a video frame using a detected car's bounding box.

A standard sedan is ~1.8m wide. We find the widest car bbox in a sample frame,
measure its pixel width, and compute mpp = CAR_WIDTH_M / bbox_px_width.

Usage:
    python scripts/calibrate_mpp.py --video data/crosswalk.mp4 --frame 600
"""

import argparse
import cv2
from ultralytics import YOLO

CAR_WIDTH_M = 1.8  # typical sedan width in metres

VEHICLE_CLASSES = {"car", "bus", "truck"}


def estimate_mpp(video_path: str, frame_idx: int, model_path: str = "yolo12n.pt") -> float:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    model = YOLO(model_path)
    results = model(frame, verbose=False, conf=0.4)
    boxes = results[0].boxes

    best_width_px = None
    best_class = None
    for cls_id, xyxy in zip(boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy()):
        class_name = model.names[int(cls_id)]
        if class_name not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = xyxy
        width_px = x2 - x1
        if best_width_px is None or width_px > best_width_px:
            best_width_px = width_px
            best_class = class_name

    if best_width_px is None:
        raise RuntimeError("No vehicles detected in this frame. Try a different --frame index.")

    mpp = CAR_WIDTH_M / best_width_px
    print(f"Detected {best_class} with bbox width = {best_width_px:.0f} px")
    print(f"Assuming car width = {CAR_WIDTH_M} m")
    print(f"\n  meters_per_pixel = {mpp:.4f}")
    print(f"\nUpdate configs/zones.yaml:  meters_per_pixel: {mpp:.4f}")
    return mpp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",  default="data/crosswalk.mp4")
    p.add_argument("--frame",  type=int, default=600)
    p.add_argument("--model",  default="yolo12n.pt")
    args = p.parse_args()
    estimate_mpp(args.video, args.frame, args.model)
