import math
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np
from src.homography import GroundPlane
from src.utils import (
    compute_velocity,
    euclidean_distance,
    point_in_polygon,
    velocity_vector,
)

BBOX_PROXIMITY_EXPAND_FRAC = 0.10  # expand each bbox by 10% before testing overlap
VISUAL_PROXIMITY_GAP_PX = 100.0  # near non-overlap gap for "car directly in front" cases
PED_LEAVING_THRESHOLD_PX_PER_FRAME = 0.5  # below this we treat ped as approaching/static; above and away → skip


def _expand_bbox(bbox, frac: float):
    x1, y1, x2, y2 = bbox
    dw = (x2 - x1) * frac
    dh = (y2 - y1) * frac
    return (x1 - dw, y1 - dh, x2 + dw, y2 + dh)


def _bboxes_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_gap_px(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def _bboxes_in_visual_conflict(ped_bbox, veh_bbox) -> bool:
    """Near image-space conflict for car-front cases missed by BEV extrapolation."""
    px1, py1, px2, py2 = ped_bbox
    vx1, vy1, vx2, vy2 = veh_bbox
    veh_h = max(vy2 - vy1, 1)
    veh_w = max(vx2 - vx1, 1)
    gap_limit = max(VISUAL_PROXIMITY_GAP_PX, 0.12 * veh_w)

    if _bbox_gap_px(ped_bbox, veh_bbox) > gap_limit:
        return False

    overlap_y = min(py2, vy2) - max(py1, vy1)
    if overlap_y < 0.12 * min(max(py2 - py1, 1), veh_h):
        return False

    ped_bottom = py2
    return (vy1 - 0.35 * veh_h) <= ped_bottom <= (vy2 + 0.65 * veh_h)


def _ped_leaving_vehicle(
    ped_history,
    ped_anchor: Tuple[float, float],
    veh_anchor: Tuple[float, float],
    leaving_threshold: float,
) -> bool:
    """True if the ped is moving AWAY from the vehicle fast enough to count
    as 'past' it.

    Approach component = projection of ped's velocity onto the (ped→veh)
    direction. Positive = approaching, zero = stationary or lateral motion,
    negative = leaving. Returns True only when the negative component
    exceeds `leaving_threshold` so a ped walking laterally past a parked car
    (approach ≈ 0) still fires.
    """
    if len(ped_history) < 2:
        return False
    vx, vy = velocity_vector(list(ped_history))
    rx = veh_anchor[0] - ped_anchor[0]
    ry = veh_anchor[1] - ped_anchor[1]
    mag_sq = rx * rx + ry * ry
    if mag_sq == 0:
        return False
    approach = (vx * rx + vy * ry) / math.sqrt(mag_sq)
    return approach < -leaving_threshold


def _ped_bbox_in_zone(bbox, polygon: np.ndarray) -> bool:
    """Robust in-zone test for a pedestrian bbox.

    Tests three points along the bbox bottom edge (left, center, right).
    The ped is considered in-zone if at least two lie inside the polygon.
    This preserves edge-jitter tolerance without letting a single sidewalk
    corner make the whole bbox eligible for risk scoring.
    """
    x1, _, x2, y2 = bbox
    samples = [(x1, y2), ((x1 + x2) / 2.0, y2), (x2, y2)]
    return sum(point_in_polygon(p, polygon) for p in samples) >= 2


class RiskScorer:
    """Pedestrian-vehicle risk scorer.

    Two outputs:

    - `score_frame(...)` returns COOLDOWN-GATED event dicts suitable for the
      CSV/event log. Each (ped_id, vehicle_id) pair fires at most once per
      `cooldown_frames`, so the log doesn't drown in repeats during a sustained
      close pass.
    - `compute_active_risks(...)` returns the SAME-SHAPED dicts but without the
      cooldown gate, so the visualization can keep the colored bbox lit for as
      long as the geometric danger actually persists.

    In-zone hysteresis: a ped must be in zone for `zone_hysteresis_frames`
    consecutive frames before it can fire — but the streak resets to 0 the
    moment the ped is out of zone for one frame. Asymmetric on purpose: ~100 ms
    delay on entry (invisible), instant green on exit (kills sidewalk-flicker).

    Modes (selected via constructor args):

    1. **Pixel mode** (legacy): all distances/speeds in image pixels, gated
       against `caution_distance_px`, `danger_distance_px`, and
       `min_vehicle_speed_px_per_frame`.
    2. **Metric mode** (when a `GroundPlane` is supplied): every detection's
       bottom_center is projected to BEV metres. The distance check, the
       perpendicular-path check, and the speed gate all run in true metric
       space — eliminates the perspective error a single global mpp can't model.
    """

    def __init__(
        self,
        zone_polygon: np.ndarray,
        caution_distance_px: float,
        danger_distance_px: float,
        cooldown_frames: int = 15,
        min_vehicle_speed_px_per_frame: float = 0.0,
        ground_plane: Optional[GroundPlane] = None,
        caution_distance_m: Optional[float] = None,
        danger_distance_m:  Optional[float] = None,
        min_vehicle_speed_m_per_s: float = 0.0,
        zone_hysteresis_frames: int = 1,
    ):
        self.zone_polygon = zone_polygon
        self.caution_distance_px = caution_distance_px
        self.danger_distance_px  = danger_distance_px
        self.cooldown_frames = cooldown_frames
        self.min_vehicle_speed_px_per_frame = min_vehicle_speed_px_per_frame
        self.ground_plane = ground_plane
        self.caution_distance_m = caution_distance_m
        self.danger_distance_m  = danger_distance_m
        self.min_vehicle_speed_m_per_s = min_vehicle_speed_m_per_s
        self.zone_hysteresis_frames = max(1, zone_hysteresis_frames)
        self._last_fired: Dict[Tuple[int, int], int] = {}
        self._zone_streak: Dict[int, int] = {}
        self._last_zone_update_frame: Optional[int] = None

    def update_zone_state(self, detections: List[Dict], frame_idx: Optional[int] = None) -> None:
        """Update per-ped consecutive-in-zone counter. Idempotent per frame_idx."""
        seen = set()
        for det in detections:
            if not (det.get("is_pedestrian") and not det.get("is_tree")):
                continue
            pid = det["id"]
            seen.add(pid)
            if _ped_bbox_in_zone(det["bbox"], self.zone_polygon):
                self._zone_streak[pid] = self._zone_streak.get(pid, 0) + 1
            else:
                self._zone_streak[pid] = 0
        for pid in list(self._zone_streak.keys()):
            if pid not in seen:
                self._zone_streak.pop(pid, None)
        if frame_idx is not None:
            self._last_zone_update_frame = frame_idx

    def _ensure_zone_state(self, detections: List[Dict], frame_idx: int) -> None:
        if self._last_zone_update_frame != frame_idx:
            self.update_zone_state(detections, frame_idx)

    def score_frame(
        self,
        detections: List[Dict],
        track_history: Dict[int, deque],
        frame_idx: int,
        fps: float,
    ) -> List[Dict]:
        """Cooldown-gated events for the CSV log."""
        self._ensure_zone_state(detections, frame_idx)
        events = []
        for risk in self._iter_risks(detections, track_history, fps):
            pair = (risk["person_id"], risk["vehicle_id"])
            last = self._last_fired.get(pair, -self.cooldown_frames)
            if frame_idx - last < self.cooldown_frames:
                continue
            self._last_fired[pair] = frame_idx
            risk["frame"] = frame_idx
            risk["timestamp_sec"] = round(frame_idx / fps, 2)
            events.append(risk)
        return events

    def compute_active_risks(
        self,
        detections: List[Dict],
        track_history: Dict[int, deque],
        frame_idx: int,
        fps: float,
    ) -> List[Dict]:
        """Per-frame active risks, without cooldown — for the visualization."""
        self._ensure_zone_state(detections, frame_idx)
        out = []
        for risk in self._iter_risks(detections, track_history, fps):
            risk["frame"] = frame_idx
            risk["timestamp_sec"] = round(frame_idx / fps, 2)
            out.append(risk)
        return out

    def _iter_risks(
        self,
        detections: List[Dict],
        track_history: Dict[int, deque],
        fps: float,
    ) -> Iterator[Dict]:
        peds = [d for d in detections if d["is_pedestrian"]]
        vehs = [d for d in detections if d["is_vehicle"]]
        gp = self.ground_plane

        for ped in peds:
            # Hysteresis on top of the robust in-zone check. Ped must have been
            # in zone for `zone_hysteresis_frames` consecutive frames before
            # firing — but the streak is reset to 0 by `update_zone_state` the
            # moment the ped is out of zone for one frame, so exits are instant.
            if self._zone_streak.get(ped["id"], 0) < self.zone_hysteresis_frames:
                continue

            ped_anchor = ped.get("bottom_center", ped["center"])
            ped_bev = gp.image_to_world(ped_anchor) if gp else None
            ped_expanded = _expand_bbox(ped["bbox"], BBOX_PROXIMITY_EXPAND_FRAC)

            for veh in vehs:
                veh_anchor = veh.get("bottom_center", veh["center"])
                veh_expanded = _expand_bbox(veh["bbox"], BBOX_PROXIMITY_EXPAND_FRAC)

                # Pixel-space proximity shortcut. If the ped's 10%-expanded bbox
                # overlaps the vehicle's, the pair is visually touching on
                # screen — this IS the in-front-of-the-car moment, the most
                # dangerous one in the whole pass. Fire TTC_WARNING (red)
                # regardless of BEV — bypasses the perspective error you get
                # when a parked car sits outside the homography quad.
                if _bboxes_overlap(ped_expanded, veh_expanded):
                    dist_px_overlap = euclidean_distance(ped_anchor, veh_anchor)
                    overlap_event = {
                        "person_id": ped["id"],
                        "vehicle_id": veh["id"],
                        "distance_px": round(dist_px_overlap, 1),
                        "risk_label": "TTC_WARNING",
                    }
                    if gp is not None:
                        overlap_event["distance_m"] = 0.0
                    yield overlap_event
                    continue

                visual_conflict = _bboxes_in_visual_conflict(ped["bbox"], veh["bbox"])

                if gp:
                    veh_bev = gp.image_to_world(veh_anchor)
                    dist_m  = euclidean_distance(ped_bev, veh_bev)
                    dist_px = euclidean_distance(ped_anchor, veh_anchor)
                    if dist_m > self.caution_distance_m and not visual_conflict:
                        continue
                else:
                    dist_px = euclidean_distance(ped_anchor, veh_anchor)
                    dist_m  = None
                    if dist_px > self.caution_distance_px and not visual_conflict:
                        continue

                veh_hist = track_history.get(veh["id"], deque())

                ped_hist = track_history.get(ped["id"], deque())

                if gp:
                    veh_hist_bev = [gp.image_to_world(p) for p in veh_hist]
                    veh_speed_m_per_frame = compute_velocity(veh_hist_bev)
                    is_moving = veh_speed_m_per_frame * fps >= self.min_vehicle_speed_m_per_s
                    if is_moving:
                        if not self._ped_in_vehicle_path(
                            veh_hist_bev, veh_bev, ped_bev, self.danger_distance_m,
                        ):
                            if not visual_conflict:
                                continue
                            label = "PROXIMITY"
                        else:
                            label = "TTC_WARNING" if dist_m <= self.danger_distance_m else "PROXIMITY"
                    else:
                        if dist_m > self.danger_distance_m and not visual_conflict:
                            continue
                        # Stopped vehicle: with no vehicle direction to use, fall
                        # back to ped direction. If the ped is clearly walking
                        # away from this vehicle, the danger has passed — skip
                        # so the box flips green instead of lingering orange.
                        if _ped_leaving_vehicle(
                            ped_hist, ped_anchor, veh_anchor,
                            PED_LEAVING_THRESHOLD_PX_PER_FRAME,
                        ):
                            continue
                        label = "PROXIMITY"
                else:
                    veh_speed = compute_velocity(list(veh_hist))
                    is_moving = veh_speed > self.min_vehicle_speed_px_per_frame
                    if is_moving:
                        if not self._ped_in_vehicle_path(
                            list(veh_hist), veh_anchor, ped_anchor, self.danger_distance_px,
                        ):
                            if not visual_conflict:
                                continue
                            label = "PROXIMITY"
                        else:
                            label = "TTC_WARNING" if dist_px <= self.danger_distance_px else "PROXIMITY"
                    else:
                        if dist_px > self.danger_distance_px and not visual_conflict:
                            continue
                        if _ped_leaving_vehicle(
                            ped_hist, ped_anchor, veh_anchor,
                            PED_LEAVING_THRESHOLD_PX_PER_FRAME,
                        ):
                            continue
                        label = "PROXIMITY"

                event = {
                    "person_id": ped["id"],
                    "vehicle_id": veh["id"],
                    "distance_px": round(dist_px, 1),
                    "risk_label": label,
                }
                if dist_m is not None:
                    event["distance_m"] = round(dist_m, 2)
                yield event

    def _ped_in_vehicle_path(
        self,
        veh_hist,
        veh_anchor: Tuple[float, float],
        ped_anchor: Tuple[float, float],
        threshold: float,
    ) -> bool:
        """True iff the ped is ahead of the vehicle AND within `threshold` of
        the line the vehicle is currently traveling along.

        Unit-agnostic: works in pixel space (threshold = danger_distance_px)
        or in BEV metres (threshold = danger_distance_m), as long as
        veh_hist, veh_anchor, ped_anchor are all in the same space.
        """
        vx, vy = velocity_vector(list(veh_hist))
        speed_sq = vx * vx + vy * vy
        if speed_sq == 0.0:
            return False
        rx = ped_anchor[0] - veh_anchor[0]
        ry = ped_anchor[1] - veh_anchor[1]
        longitudinal = rx * vx + ry * vy  # signed projection scaled by |v|
        if longitudinal <= 0:
            return False  # ped is behind the vehicle
        # perp_dist^2 = |r|^2 - (r·v)^2 / |v|^2
        r_sq = rx * rx + ry * ry
        perp_sq = max(r_sq - (longitudinal * longitudinal) / speed_sq, 0.0)
        return math.sqrt(perp_sq) <= threshold
