import numpy as np
import cv2
import yaml
from typing import Tuple, List

from src.homography import load_ground_plane

MPS_TO_MPH = 2.237

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def compute_velocity(points: List[Tuple[float, float]], n_frames: int = 15) -> float:
    """Speed (units/frame) over the last n_frames steps.

    Uses start-to-end displacement / interval rather than mean of per-frame
    deltas so detection jitter (which oscillates frame-to-frame) cancels.
    Unit-agnostic: pixels in / pixels out, metres in / metres out.
    """
    if len(points) < 2:
        return 0.0
    recent = list(points)[-n_frames:]
    span = len(recent) - 1
    if span <= 0:
        return 0.0
    return float(euclidean_distance(recent[0], recent[-1]) / span)


def velocity_vector(points: List[Tuple[float, float]], n_frames: int = 15) -> Tuple[float, float]:
    """Mean per-frame velocity vector (dx, dy) over the last n_frames steps."""
    if len(points) < 2:
        return (0.0, 0.0)
    recent = list(points)[-n_frames:]
    span = len(recent) - 1
    if span <= 0:
        return (0.0, 0.0)
    dx = (recent[-1][0] - recent[0][0]) / span
    dy = (recent[-1][1] - recent[0][1]) / span
    return (float(dx), float(dy))

def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def load_zone_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    mpp = cfg['risk'].get('meters_per_pixel', 0.0085)
    caution_m = cfg['risk']['caution_distance_m']
    danger_m  = cfg['risk']['danger_distance_m']
    min_mph   = cfg['risk'].get('min_vehicle_speed_mph', 0.0)
    return {
        'polygon': np.array(cfg['danger_zone'], dtype=np.int32),
        'caution_distance_m':  caution_m,
        'danger_distance_m':   danger_m,
        'caution_distance_px': caution_m / mpp,
        'danger_distance_px':  danger_m / mpp,
        'meters_per_pixel': mpp,
        'min_vehicle_speed_mph': min_mph,
        'min_vehicle_speed_m_per_s': min_mph / MPS_TO_MPH,
        'ground_plane': load_ground_plane(cfg),
    }
