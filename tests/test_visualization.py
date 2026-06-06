from collections import deque
import numpy as np
import pytest
from src.visualization import Visualizer

# Zone at center of a 300x300 frame (rows/cols 80-220)
ZONE = np.array([[80, 80], [220, 80], [220, 220], [80, 220]], dtype=np.int32)
# Zone far from banner area (rows 0-30) and from top-left
ZONE_BOTTOM = np.array([[200, 200], [280, 200], [280, 280], [200, 280]], dtype=np.int32)

FPS = 30.0


def black(h=300, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


def white(h=300, w=300):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def ped_det(track_id=1, cx=150, cy=150, x1=100, y1=100, x2=200, y2=200):
    return {"id": track_id, "class_name": "person", "is_pedestrian": True,
            "is_vehicle": False, "center": (cx, cy), "bbox": (x1, y1, x2, y2)}


def veh_det(track_id=2, cx=50, cy=50, x1=20, y1=20, x2=80, y2=80):
    return {"id": track_id, "class_name": "car", "is_pedestrian": False,
            "is_vehicle": True, "center": (cx, cy), "bbox": (x1, y1, x2, y2)}


def ttc_event():
    return {"risk_label": "TTC_WARNING", "person_id": 1, "vehicle_id": 2,
            "distance_px": 80.0, "ttc_sec": 0.5, "frame": 10, "timestamp_sec": 0.33}


def prox_event():
    return {"risk_label": "PROXIMITY", "person_id": 1, "vehicle_id": 2,
            "distance_px": 120.0, "ttc_sec": None, "frame": 10, "timestamp_sec": 0.33}


# --- shape and mutation ---

def test_draw_frame_returns_ndarray_same_shape():
    v = Visualizer(ZONE)
    frame = black()
    out = v.draw_frame(frame.copy(), [], {}, [])
    assert isinstance(out, np.ndarray)
    assert out.shape == frame.shape


def test_draw_frame_does_not_mutate_input():
    v = Visualizer(ZONE)
    frame = black()
    original = frame.copy()
    v.draw_frame(frame, [], {}, [])
    assert np.array_equal(frame, original)


def test_draw_frame_empty_inputs_no_crash():
    v = Visualizer(ZONE)
    out = v.draw_frame(black(), [], {}, [])
    assert out is not None


# --- zone overlay ---

def test_zone_not_drawn():
    v = Visualizer(ZONE)
    frame = white()
    out = v.draw_frame(frame.copy(), [], {}, [])
    # Zone overlay removed — center pixel stays pure white
    assert np.array_equal(out[150, 150], [255, 255, 255])


def test_zone_leaves_pixels_outside_zone_unchanged():
    # Use small zone, check a pixel well outside it
    small_zone = np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.int32)
    v = Visualizer(small_zone)
    frame = white()
    out = v.draw_frame(frame.copy(), [], {}, [])
    # Pixel far from zone (top-left corner, row=2, col=2) should still be white
    assert np.array_equal(out[2, 2], [255, 255, 255])


# --- bounding boxes ---

def test_pedestrian_bbox_draws_green_border():
    v = Visualizer(ZONE_BOTTOM)
    frame = black()
    dets = [ped_det(cx=150, cy=150, x1=50, y1=50, x2=100, y2=100)]
    out = v.draw_frame(frame.copy(), dets, {}, [])
    # Green channel dominant at top-left corner of bbox
    assert out[50, 50, 1] > 100  # green (BGR index 1)


def test_vehicle_bbox_draws_non_green_border():
    v = Visualizer(ZONE_BOTTOM)
    frame = black()
    dets = [veh_det(cx=50, cy=50, x1=10, y1=10, x2=80, y2=80)]
    out = v.draw_frame(frame.copy(), dets, {}, [])
    # Vehicle color has blue dominance (BGR index 0), not green
    px = out[10, 10]
    assert px[0] > px[1]  # blue > green for vehicle


def test_pedestrian_and_vehicle_have_distinct_colors():
    v = Visualizer(ZONE_BOTTOM)
    frame = black()
    dets = [
        ped_det(cx=150, cy=50, x1=130, y1=30, x2=170, y2=70),
        veh_det(cx=50, cy=150, x1=30, y1=130, x2=70, y2=170),
    ]
    out = v.draw_frame(frame.copy(), dets, {}, [])
    ped_px = out[30, 130]   # top-left of ped bbox
    veh_px = out[130, 30]   # top-left of veh bbox
    assert not np.array_equal(ped_px, veh_px)


# --- trails ---

def test_trails_not_drawn(  ):
    v = Visualizer(ZONE_BOTTOM)
    frame = black()
    history = {1: deque([(50, 50), (60, 50), (70, 50)], maxlen=30)}
    out = v.draw_frame(frame.copy(), [], history, [])
    # Trails are disabled; the trail region should remain black
    trail_region = out[45:55, 45:75]
    assert trail_region.max() == 0


# --- event banner ---

def test_banner_drawn_on_ttc_warning():
    v = Visualizer(ZONE_BOTTOM)
    frame = white()
    out = v.draw_frame(frame.copy(), [], {}, [ttc_event()])
    # Banner covers top rows; red channel should be high at row=5
    assert out[5, 150, 2] > 150  # red (BGR index 2) for TTC_WARNING


def test_banner_drawn_on_proximity_event():
    v = Visualizer(ZONE_BOTTOM)
    frame = white()
    out = v.draw_frame(frame.copy(), [], {}, [prox_event()])
    # Banner should exist (top row not pure white)
    assert not np.array_equal(out[5, 150], [255, 255, 255])


def test_no_banner_when_no_events():
    v = Visualizer(ZONE_BOTTOM)
    frame = white()
    out = v.draw_frame(frame.copy(), [], {}, [])
    # No events → no banner → top-left pixel unaffected by zone or banner
    assert np.array_equal(out[5, 5], [255, 255, 255])


def test_ttc_warning_banner_is_redder_than_proximity_banner():
    v = Visualizer(ZONE_BOTTOM)
    frame = white()
    out_ttc = v.draw_frame(frame.copy(), [], {}, [ttc_event()])
    out_prox = v.draw_frame(frame.copy(), [], {}, [prox_event()])
    # TTC uses a redder color than PROXIMITY
    assert out_ttc[5, 150, 2] >= out_prox[5, 150, 2]


# --- risk zone overlay ---

# Zone kept away from the test pixel at (row=200, col=200)
ZONE_CORNER = np.array([[320, 320], [399, 320], [399, 399], [320, 399]], dtype=np.int32)


def _risk_dets():
    """Ped at left (bbox x=60-100), vehicle at right (bbox x=300-340), y=180-220."""
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (80, 200), "bbox": (60, 180, 100, 220)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (320, 200), "bbox": (300, 180, 340, 220)}
    return [ped, veh]


def test_pedestrian_bbox_red_on_ttc_warning():
    v = Visualizer(ZONE_CORNER)
    frame = black(400, 400)
    event = {"risk_label": "TTC_WARNING", "person_id": 1, "vehicle_id": 2,
             "distance_px": 240.0, "ttc_sec": 0.5}
    out = v.draw_frame(frame.copy(), _risk_dets(), {}, [event])
    # Ped bbox top edge (row=180, col=60) should be red (DANGER color)
    assert out[180, 60, 2] > 150  # red channel dominant


def test_pedestrian_bbox_unchanged_when_no_events():
    v = Visualizer(ZONE_CORNER)
    frame = black(400, 400)
    out = v.draw_frame(frame.copy(), _risk_dets(), {}, [])
    # No event → ped box is green, not red
    assert out[180, 60, 2] < 50   # red channel low
    assert out[180, 60, 1] > 100  # green channel high


def test_pedestrian_bbox_orange_on_proximity():
    v = Visualizer(ZONE_CORNER)
    frame = black(400, 400)
    event = {"risk_label": "PROXIMITY", "person_id": 1, "vehicle_id": 2,
             "distance_px": 240.0, "ttc_sec": None}
    out = v.draw_frame(frame.copy(), _risk_dets(), {}, [event])
    # Ped bbox top edge should be orange (red + green channels both present)
    px = out[180, 60]
    assert px[2] > 100  # orange has red component (BGR index 2)
