import numpy as np
import pytest
from src.utils import euclidean_distance, compute_velocity, point_in_polygon

def test_euclidean_distance_zero():
    assert euclidean_distance((0, 0), (0, 0)) == 0.0

def test_euclidean_distance_known():
    # 3-4-5 right triangle
    assert abs(euclidean_distance((0, 0), (3, 4)) - 5.0) < 1e-6

def test_compute_velocity_empty():
    assert compute_velocity([]) == 0.0

def test_compute_velocity_stationary():
    pts = [(100, 100), (100, 100), (100, 100)]
    assert compute_velocity(pts) == 0.0

def test_compute_velocity_moving():
    # Moving 10px per frame over 2 steps → avg speed = 10.0
    pts = [(0, 0), (10, 0), (20, 0)]
    assert abs(compute_velocity(pts) - 10.0) < 1e-6

def test_point_in_polygon_inside():
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    assert point_in_polygon((50, 50), polygon) == True

def test_point_in_polygon_outside():
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    assert point_in_polygon((200, 200), polygon) == False
