from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.homography import GroundPlane
from src.utils import compute_velocity

PERSON_SAFE    = (0, 220, 0)      # green BGR — no risk
PERSON_CAUTION = (0, 140, 255)    # orange BGR — proximity
PERSON_DANGER  = (0, 0, 220)      # red BGR — TTC warning
VEHICLE_COLOR  = (220, 100, 0)    # blue BGR
TREE_COLOR     = (40, 130, 40)    # dark green BGR — context only, no risk
BANNER_TTC     = (0, 0, 220)      # red BGR
BANNER_PROX    = (0, 165, 200)    # orange BGR

PED_DISPLAY_WINDOW_FRAMES = 5      # peds move slow — short window keeps mph reactive
VEH_DISPLAY_WINDOW_FRAMES = 10     # vehicles wobble more — longer window suppresses jitter
PED_SPEED_NOISE_FLOOR_MPH = 0.3    # silence sub-walking-pace jitter
PED_HEIGHT_M              = 1.7    # assumed average pedestrian height for C-method
PED_MAX_DISPLAY_MPH       = 10.0   # clamp absurd mph from tiny/cropped ped bboxes
MPS_TO_MPH                = 2.237
GROUP_BBOX_EXPAND_FRAC    = 0.20   # expand each ped bbox 20% before testing overlap
GROUP_MIN_MEMBERS         = 2      # below this we draw individuals
RISK_PRIORITY = {None: 0, "PROXIMITY": 1, "TTC_WARNING": 2}


class Visualizer:
    def __init__(
        self,
        zone_polygon: np.ndarray,
        fps: float = 30.0,
        meters_per_pixel: float = 0.02,
        ground_plane: Optional[GroundPlane] = None,
        min_vehicle_display_mph: float = 1.0,
    ):
        self.zone_polygon = zone_polygon
        self._fps = fps
        self._mpp = meters_per_pixel
        self._gp = ground_plane
        self._veh_floor_mph = min_vehicle_display_mph

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        track_history: Dict[int, deque],
        events: List[Dict],
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_detections(out, detections, track_history, events)
        self._draw_banner(out, events)
        return out

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        track_history: Dict[int, deque],
        events: List[Dict],
    ) -> None:
        ped_risk: Dict[int, str] = {}
        for event in events:
            pid = event["person_id"]
            current = ped_risk.get(pid)
            if RISK_PRIORITY[event["risk_label"]] > RISK_PRIORITY[current]:
                ped_risk[pid] = event["risk_label"]

        peds  = [d for d in detections if d["is_pedestrian"] and not d.get("is_tree")]
        other = [d for d in detections if not (d["is_pedestrian"] and not d.get("is_tree"))]

        # Pedestrians: cluster overlapping bboxes into groups so a row of walkers
        # shows as one box instead of N stacked boxes. Risky peds keep individual
        # boxes so the warning stays attached to the right person.
        clusters = _cluster_pedestrians(peds, ped_risk)
        for cluster in clusters:
            if len(cluster) >= GROUP_MIN_MEMBERS:
                self._draw_group(frame, [peds[i] for i in cluster], ped_risk)
            else:
                self._draw_pedestrian(frame, peds[cluster[0]], track_history, ped_risk)

        for det in other:
            self._draw_other(frame, det, track_history)

    def _draw_pedestrian(self, frame, det, track_history, ped_risk):
        risk = ped_risk.get(det["id"])
        if risk == "TTC_WARNING":
            color, label = PERSON_DANGER, "DANGER"
        elif risk == "PROXIMITY":
            color, label = PERSON_CAUTION, "CAUTION"
        else:
            color, label = PERSON_SAFE, f"{det['class_name']} #{det['id']}"

        x1, y1, x2, y2 = det["bbox"]
        thickness = 3 if risk else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if risk else 1)

        history = track_history.get(det["id"], deque())
        speed_mph = self._ped_speed_mph(history, y1, y2)
        if speed_mph < PED_SPEED_NOISE_FLOOR_MPH:
            speed_mph = 0.0
        cv2.putText(frame, f"{speed_mph:.1f} mph", (x1, y2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def _draw_group(self, frame, members, ped_risk):
        worst = None
        for m in members:
            r = ped_risk.get(m["id"])
            if RISK_PRIORITY[r] > RISK_PRIORITY[worst]:
                worst = r
        if worst == "TTC_WARNING":
            color, prefix = PERSON_DANGER, "DANGER"
        elif worst == "PROXIMITY":
            color, prefix = PERSON_CAUTION, "CAUTION"
        else:
            color, prefix = PERSON_SAFE, "group"

        x1 = min(m["bbox"][0] for m in members)
        y1 = min(m["bbox"][1] for m in members)
        x2 = max(m["bbox"][2] for m in members)
        y2 = max(m["bbox"][3] for m in members)
        thickness = 3 if worst else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"{prefix} group of {len(members)}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if worst else 1)
        # Speed intentionally omitted on groups — average is meaningless.

    def _draw_other(self, frame, det, track_history):
        if det.get("is_tree"):
            color, label = TREE_COLOR, "tree"
            thickness = 1
        else:
            color = VEHICLE_COLOR
            label = f"{det['class_name']} #{det['id']}"
            thickness = 2

        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if det.get("is_tree"):
            return

        history = track_history.get(det["id"], deque())
        speed_mph = self._vehicle_speed_mph(history)
        if speed_mph < self._veh_floor_mph:
            speed_mph = 0.0
        cv2.putText(frame, f"{speed_mph:.1f} mph", (x1, y2 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def _ped_speed_mph(self, history: deque, y1: int, y2: int) -> float:
        bbox_h_px = max(y2 - y1, 1)
        local_mpp = PED_HEIGHT_M / bbox_h_px
        pixel_speed = compute_velocity(list(history), n_frames=PED_DISPLAY_WINDOW_FRAMES)
        return min(pixel_speed * self._fps * local_mpp * MPS_TO_MPH, PED_MAX_DISPLAY_MPH)

    def _vehicle_speed_mph(self, history: deque) -> float:
        if self._gp and len(history) >= 2:
            bev_history = [self._gp.image_to_world(p) for p in history]
            speed_m_per_frame = compute_velocity(bev_history, n_frames=VEH_DISPLAY_WINDOW_FRAMES)
            return speed_m_per_frame * self._fps * MPS_TO_MPH
        pixel_speed = compute_velocity(list(history), n_frames=VEH_DISPLAY_WINDOW_FRAMES)
        return pixel_speed * self._fps * self._mpp * MPS_TO_MPH

    def _draw_banner(self, frame: np.ndarray, events: List[Dict]) -> None:
        if not events:
            return
        worst = next((e for e in events if e["risk_label"] == "TTC_WARNING"), events[0])
        color = BANNER_TTC if worst["risk_label"] == "TTC_WARNING" else BANNER_PROX
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 30), color, -1)
        if worst["risk_label"] == "TTC_WARNING" and worst.get("ttc_sec") is not None:
            text = f"TTC WARNING  {worst['ttc_sec']:.1f}s  |  ped {worst['person_id']} <-> veh {worst['vehicle_id']}"
        else:
            dist_label = ""
            if worst.get("distance_m") is not None:
                dist_label = f"  {worst['distance_m']:.1f} m"
            text = f"PROXIMITY{dist_label}  |  ped {worst['person_id']} <-> veh {worst['vehicle_id']}"
        cv2.putText(frame, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _cluster_pedestrians(peds: List[Dict], ped_risk: Dict[int, str]) -> List[List[int]]:
    """Group nearby ped detections into clusters via union-find on expanded-bbox overlap.

    Risky peds (any non-None entry in ped_risk) stay as singletons so the
    warning bbox stays attached to the actual flagged individual.
    """
    n = len(peds)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    expanded = [_expand_bbox(p["bbox"], GROUP_BBOX_EXPAND_FRAC) for p in peds]
    risky = [ped_risk.get(p["id"]) is not None for p in peds]

    for i in range(n):
        if risky[i]:
            continue  # don't merge risky peds into groups
        for j in range(i + 1, n):
            if risky[j]:
                continue
            if _bboxes_overlap(expanded[i], expanded[j]):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _expand_bbox(bbox: Tuple[int, int, int, int], frac: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    dw = (x2 - x1) * frac
    dh = (y2 - y1) * frac
    return (x1 - dw, y1 - dh, x2 + dw, y2 + dh)


def _bboxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
