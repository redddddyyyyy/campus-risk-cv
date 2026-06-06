import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

import cv2

from src.detector_tracker import DetectorTracker
from src.risk_scoring import RiskScorer
from src.utils import load_zone_config
from src.visualization import Visualizer

# Static-pedestrian filter constants. A "ped" detection is considered signage
# (not a real human) and dropped if it has been observed for a long time without
# moving and its bbox is small. Both gates required so we never accidentally
# drop a real ped standing still in the crosswalk for a long beat.
STATIC_PED_FRAMES        = 30    # ~1 second @ 30 fps of stationary track
STATIC_PED_DISPLACE_PX   = 5.0   # bbox-bottom drift over the window
STATIC_PED_MAX_BBOX_H_PX = 150   # taller than this is almost certainly a real ped
SIGN_PERSON_MAX_BBOX_H_PX = 170
SIGN_YELLOW_FRACTION = 0.18


def _yellow_fraction(frame, bbox) -> float:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 60, 80), (40, 255, 255))
    return float(mask.mean() / 255.0)


def _filter_static_pedestrians(detections: List[dict],
                               track_history: Dict[int, deque],
                               frame=None) -> List[dict]:
    """Drop ped detections that look like signage: long-tracked, motionless,
    and small. Catches yield-to-pedestrian signs and similar painted figures
    that YOLO classifies as `person`.
    """
    out: List[dict] = []
    for det in detections:
        if det.get("is_pedestrian") and not det.get("is_tree"):
            history = track_history.get(det["id"])
            x1, y1, x2, y2 = det["bbox"]
            bbox_h = y2 - y1
            if (frame is not None
                    and bbox_h < SIGN_PERSON_MAX_BBOX_H_PX
                    and _yellow_fraction(frame, det["bbox"]) >= SIGN_YELLOW_FRACTION):
                continue
            if (history is not None
                    and len(history) >= STATIC_PED_FRAMES
                    and bbox_h < STATIC_PED_MAX_BBOX_H_PX):
                window = list(history)[-STATIC_PED_FRAMES:]
                xs = [p[0] for p in window]
                ys = [p[1] for p in window]
                spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
                if spread < STATIC_PED_DISPLACE_PX:
                    continue
        out.append(det)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Campus Pedestrian-Vehicle Risk Analyzer")
    p.add_argument("--video",  required=True,               help="Input video path")
    p.add_argument("--config", default="configs/zones_img5757.yaml", help="Zone config YAML")
    p.add_argument("--output", default=None,                help="Annotated output video path")
    p.add_argument("--show",   action="store_true",          help="Display live preview window")
    p.add_argument("--csv",    default=None,                help="Events CSV output path")
    return p.parse_args(argv)


def write_events_csv(events: List[dict], csv_path: str) -> None:
    if not events:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)


def run(args) -> int:
    cfg = load_zone_config(args.config)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = DetectorTracker()
    min_speed_px_per_frame = cfg["min_vehicle_speed_mph"] / (fps * cfg["meters_per_pixel"] * 2.237)
    scorer   = RiskScorer(
        cfg["polygon"], cfg["caution_distance_px"], cfg["danger_distance_px"],
        min_vehicle_speed_px_per_frame=min_speed_px_per_frame,
        ground_plane=cfg["ground_plane"],
        caution_distance_m=cfg["caution_distance_m"],
        danger_distance_m=cfg["danger_distance_m"],
        min_vehicle_speed_m_per_s=cfg["min_vehicle_speed_m_per_s"],
        zone_hysteresis_frames=3,
    )
    viz      = Visualizer(
        cfg["polygon"], fps=fps, meters_per_pixel=cfg["meters_per_pixel"],
        ground_plane=cfg["ground_plane"],
        min_vehicle_display_mph=cfg["min_vehicle_speed_mph"],
    )
    if cfg["ground_plane"]:
        print(f"  Homography: ON  (caution={cfg['caution_distance_m']} m, "
              f"danger={cfg['danger_distance_m']} m, "
              f"min_speed={cfg['min_vehicle_speed_mph']} mph)")
    else:
        print(f"  Homography: OFF — using global mpp={cfg['meters_per_pixel']}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    track_history = defaultdict(lambda: deque(maxlen=30))
    all_events: List[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_and_track(frame)
        # Suppress sign-printed "pedestrians" (yield-to-ped signs etc.) before
        # they reach the scorer — track_history at this point still excludes
        # the current frame, so static signs that have been tracked for a while
        # already meet the filter criteria.
        detections = _filter_static_pedestrians(detections, track_history, frame)
        events     = scorer.score_frame(detections, track_history, frame_idx, fps)
        all_events.extend(events)

        for det in detections:
            if det.get("is_tree"):
                continue  # untracked context-only detections
            track_history[det["id"]].append(det["bottom_center"])

        # Display reflects CURRENT-FRAME risk state, not the cooldown-gated event log.
        # Without this split the colored bbox flashes briefly per cooldown fire and
        # is green between flashes — which makes a real close pass look safe and
        # post-pass cooldown fires light up after the danger has actually ended.
        active_risks = scorer.compute_active_risks(detections, track_history, frame_idx, fps)
        annotated = viz.draw_frame(frame, detections, track_history, active_risks)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Campus Risk Analyzer", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  frame {frame_idx}  events so far: {len(all_events)}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    csv_path = args.csv or (
        str(Path(args.output).with_suffix(".csv")) if args.output else None
    )
    if csv_path:
        write_events_csv(all_events, csv_path)
        print(f"Events saved → {csv_path}")

    print(f"Done. {frame_idx} frames processed — {len(all_events)} risk events logged.")
    return len(all_events)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
