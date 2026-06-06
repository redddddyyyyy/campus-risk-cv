from collections import deque
from pathlib import Path
import numpy as np
import pytest
from src.main import (
    SIGN_PERSON_MAX_BBOX_H_PX,
    STATIC_PED_DISPLACE_PX,
    STATIC_PED_FRAMES,
    STATIC_PED_MAX_BBOX_H_PX,
    _filter_static_pedestrians,
    parse_args,
    write_events_csv,
)

EVENT = {
    "frame": 10, "timestamp_sec": 0.33, "person_id": 1, "vehicle_id": 2,
    "distance_px": 100.0, "ttc_sec": 0.5, "risk_label": "TTC_WARNING",
}


# --- parse_args ---

def test_parse_args_video_stored():
    args = parse_args(["--video", "data/crosswalk.mp4"])
    assert args.video == "data/crosswalk.mp4"


def test_parse_args_default_config():
    args = parse_args(["--video", "v.mp4"])
    assert args.config == "configs/zones_img5757.yaml"


def test_parse_args_default_output_none():
    args = parse_args(["--video", "v.mp4"])
    assert args.output is None


def test_parse_args_default_show_false():
    args = parse_args(["--video", "v.mp4"])
    assert args.show is False


def test_parse_args_custom_config():
    args = parse_args(["--video", "v.mp4", "--config", "custom.yaml"])
    assert args.config == "custom.yaml"


def test_parse_args_show_flag():
    args = parse_args(["--video", "v.mp4", "--show"])
    assert args.show is True


def test_parse_args_output_and_csv():
    args = parse_args(["--video", "v.mp4", "--output", "out.mp4", "--csv", "ev.csv"])
    assert args.output == "out.mp4"
    assert args.csv == "ev.csv"


# --- write_events_csv ---

def test_write_events_csv_creates_file(tmp_path):
    path = str(tmp_path / "events.csv")
    write_events_csv([EVENT], path)
    assert Path(path).exists()


def test_write_events_csv_header_matches_event_keys(tmp_path):
    path = str(tmp_path / "events.csv")
    write_events_csv([EVENT], path)
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header == list(EVENT.keys())


def test_write_events_csv_writes_all_rows(tmp_path):
    path = str(tmp_path / "events.csv")
    events = [EVENT, {**EVENT, "frame": 20, "timestamp_sec": 0.67}]
    write_events_csv(events, path)
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3  # 1 header + 2 data rows


def test_write_events_csv_empty_list_creates_no_file(tmp_path):
    path = str(tmp_path / "events.csv")
    write_events_csv([], path)
    assert not Path(path).exists()


def test_write_events_csv_values_correct(tmp_path):
    path = str(tmp_path / "events.csv")
    write_events_csv([EVENT], path)
    with open(path) as f:
        lines = f.readlines()
    # Second line is the data row
    assert "TTC_WARNING" in lines[1]
    assert "0.5" in lines[1]


# --- _filter_static_pedestrians ---

def _ped(track_id, x, y, h=300):
    """A foreground-sized ped detection, 50 px wide and `h` px tall, at (x,y)."""
    return {"id": track_id, "class_name": "person", "is_pedestrian": True,
            "is_vehicle": False, "center": (x, y), "bottom_center": (x, y),
            "bbox": (x - 25, y - h, x + 25, y)}


def _veh(track_id, x, y):
    return {"id": track_id, "class_name": "car", "is_pedestrian": False,
            "is_vehicle": True, "center": (x, y), "bottom_center": (x, y),
            "bbox": (x - 30, y - 30, x + 30, y + 30)}


def test_filter_drops_static_small_pedestrian():
    """A ped detection that has been stationary for the static-window
    threshold AND has a small bbox is treated as signage and dropped."""
    sign = _ped(99, 600, 500, h=80)  # small bbox, signage-sized
    history = {99: deque([(600, 500)] * STATIC_PED_FRAMES, maxlen=STATIC_PED_FRAMES)}
    out = _filter_static_pedestrians([sign], history)
    assert out == []


def test_filter_keeps_static_but_tall_pedestrian():
    """Tall bbox = real foreground person standing still. Don't drop."""
    person = _ped(99, 600, 500, h=300)
    history = {99: deque([(600, 500)] * STATIC_PED_FRAMES, maxlen=STATIC_PED_FRAMES)}
    out = _filter_static_pedestrians([person], history)
    assert len(out) == 1


def test_filter_keeps_moving_pedestrian():
    """Small bbox but visibly moving over the window — still a real ped."""
    walker = _ped(99, 600, 500, h=80)
    moving_history = deque(maxlen=STATIC_PED_FRAMES)
    for i in range(STATIC_PED_FRAMES):
        moving_history.append((600 + i, 500))  # 1 px / frame for 30 frames = 30 px > threshold
    out = _filter_static_pedestrians([walker], {99: moving_history})
    assert len(out) == 1


def test_filter_keeps_pedestrian_with_short_history():
    """A new track with < STATIC_PED_FRAMES of history is never dropped — we
    haven't seen it long enough to know if it's stationary signage."""
    new_ped = _ped(99, 600, 500, h=80)
    short_history = deque([(600, 500)] * 5, maxlen=STATIC_PED_FRAMES)
    out = _filter_static_pedestrians([new_ped], {99: short_history})
    assert len(out) == 1


def test_filter_drops_yellow_crosswalk_sign_person():
    sign = _ped(99, 600, 500, h=SIGN_PERSON_MAX_BBOX_H_PX - 10)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    x1, y1, x2, y2 = sign["bbox"]
    frame[y1:y2, x1:x2] = (0, 255, 255)
    out = _filter_static_pedestrians([sign], {}, frame)
    assert out == []


def test_filter_passes_vehicles_through_unchanged():
    """The static-ped filter must not affect vehicle detections."""
    car = _veh(99, 600, 500)
    out = _filter_static_pedestrians([car], {})
    assert out == [car]
