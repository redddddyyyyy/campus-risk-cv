"""Ground-plane homography helper.

Wraps cv2.getPerspectiveTransform so the rest of the pipeline can talk in
real metres on a bird's-eye-view (BEV) plane instead of raw pixels. A
single global meters_per_pixel can't model perspective foreshortening, so
the same ped at the foreground vs. background reads as wildly different
speeds. Projecting through a homography fixes that.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class GroundPlane:
    """4-point image -> world (m) homography on the ground plane."""

    def __init__(self, image_points: Sequence[Sequence[float]],
                 world_points:  Sequence[Sequence[float]]):
        img = np.asarray(image_points, dtype=np.float32)
        wld = np.asarray(world_points, dtype=np.float32)
        if img.shape != (4, 2) or wld.shape != (4, 2):
            raise ValueError(
                f"image_points and world_points must each be 4x2; got "
                f"{img.shape} and {wld.shape}"
            )
        self.image_points = img
        self.world_points = wld
        self.H = cv2.getPerspectiveTransform(img, wld)

    def image_to_world(self, pt: Point) -> Point:
        x, y = float(pt[0]), float(pt[1])
        vec = self.H @ np.array([x, y, 1.0], dtype=np.float64)
        w = vec[2]
        if w == 0.0:
            return (float("inf"), float("inf"))
        return (float(vec[0] / w), float(vec[1] / w))

    def image_to_world_array(self, pts: Iterable[Point]) -> List[Point]:
        return [self.image_to_world(p) for p in pts]

    def distance_meters(self, p1: Point, p2: Point) -> float:
        x1, y1 = self.image_to_world(p1)
        x2, y2 = self.image_to_world(p2)
        return float(np.hypot(x2 - x1, y2 - y1))


def load_ground_plane(cfg: dict) -> GroundPlane | None:
    """Build a GroundPlane from a parsed YAML config, or None if no homography block."""
    block = cfg.get("homography")
    if not block:
        return None
    return GroundPlane(block["image_points"], block["world_points"])
