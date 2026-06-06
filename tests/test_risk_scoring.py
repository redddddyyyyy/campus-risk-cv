from collections import deque
import numpy as np
import pytest
from src.homography import GroundPlane
from src.risk_scoring import RiskScorer

ZONE = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32)
CAUTION_PX = 150
DANGER_PX  = 90
FPS = 30.0


def scorer(min_speed=0.0):
    return RiskScorer(zone_polygon=ZONE, caution_distance_px=CAUTION_PX,
                      danger_distance_px=DANGER_PX,
                      min_vehicle_speed_px_per_frame=min_speed)


def det(track_id, class_name, cx, cy):
    is_ped = class_name == "person"
    return {
        "id": track_id,
        "class_name": class_name,
        "is_pedestrian": is_ped,
        "is_vehicle": not is_ped,
        "center": (cx, cy),
        "bottom_center": (cx, cy),
        "bbox": (cx - 10, cy - 10, cx + 10, cy + 10),
    }


def hist(*positions):
    d = deque(maxlen=30)
    for p in positions:
        d.append(p)
    return d


def approaching_history(start_x, end_x, y, n=15):
    """Vehicle history walking from start_x toward end_x along constant y."""
    return hist(*[(int(start_x + (end_x - start_x) * i / (n - 1)), y) for i in range(n)])


# --- score_frame ---

def test_no_event_when_far_apart():
    s = scorer()
    dets = [det(1, "person", 100, 500), det(2, "car", 400, 500)]
    assert s.score_frame(dets, {}, frame_idx=0, fps=FPS) == []


def test_no_event_when_close_but_no_history():
    """No vehicle history → can't confirm motion → skip (not a flag).

    v12 contract: only moving vehicles heading toward ped fire events.
    Car at 330 keeps euclidean distance=130 within caution_px=150 but
    bbox gap=110 px > VISUAL_PROXIMITY_GAP_PX=100, so visual-conflict
    shortcut does not fire.
    """
    s = scorer()
    dets = [det(1, "person", 200, 500), det(2, "car", 330, 500)]
    assert s.score_frame(dets, {}, frame_idx=0, fps=FPS) == []


def test_stopped_vehicle_in_caution_range_does_not_fire():
    """Stopped car in caution range but >danger_distance from ped → NO event.

    v13 design: queued cars at lights don't flag sidewalk peds at moderate distance.
    Distance 130 px > danger_px=90 → stopped vehicle ignored.
    Car at 380 keeps bbox gap=110 px > VISUAL_PROXIMITY_GAP_PX=100, so
    visual-conflict shortcut does not fire.
    """
    s = scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 380, 500)]
    history = {
        1: hist((250, 500), (250, 500), (250, 500)),
        2: hist((380, 500), (380, 500), (380, 500)),
    }
    assert s.score_frame(dets, history, frame_idx=10, fps=FPS) == []


def test_stopped_vehicle_within_danger_radius_fires_proximity():
    """Stopped vehicle within danger_distance of an in-zone ped → PROXIMITY.

    v15.1 rule: the in-zone check on the ped is the sidewalk guard, not the
    speed gate. A stopped car right next to an in-zone ped is the textbook
    "creeping into the crosswalk" scenario and must flag. Sidewalk peds are
    excluded by the zone polygon, not by the vehicle speed gate.

    Distance 50 px ≤ danger_px=90 → PROXIMITY (never TTC for stopped vehicles).
    """
    s = scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 300, 500)]
    history = {
        1: hist((250, 500), (250, 500), (250, 500)),
        2: hist((300, 500), (300, 500), (300, 500)),
    }
    events = s.score_frame(dets, history, frame_idx=10, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "PROXIMITY"


def test_danger_event_when_vehicle_approaches_close():
    """Vehicle moving toward ped at danger distance → TTC_WARNING."""
    s = scorer()
    dets_close = [det(1, "person", 250, 500), det(2, "car", 295, 500)]
    history = {2: approaching_history(start_x=400, end_x=295, y=500, n=15)}
    events = s.score_frame(dets_close, history, frame_idx=14, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "TTC_WARNING"


def test_no_event_ped_vs_ped():
    s = scorer()
    dets = [det(1, "person", 200, 500), det(2, "person", 250, 500)]
    assert s.score_frame(dets, {}, frame_idx=0, fps=FPS) == []


def test_no_event_veh_vs_veh():
    s = scorer()
    dets = [det(1, "car", 200, 500), det(2, "bus", 250, 500)]
    assert s.score_frame(dets, {}, frame_idx=0, fps=FPS) == []


def test_event_timestamp_correct():
    s = scorer()
    dets = [det(1, "person", 200, 500), det(2, "car", 300, 500)]
    history = {2: approaching_history(start_x=400, end_x=300, y=500, n=15)}
    events = s.score_frame(dets, history, frame_idx=90, fps=FPS)
    assert len(events) == 1
    assert abs(events[0]["timestamp_sec"] - 3.0) < 0.01
    assert events[0]["frame"] == 90


def test_multiple_pairs_all_reported():
    s = scorer()
    dets = [
        det(1, "person", 250, 500),
        det(2, "car",    300, 500),
        det(3, "truck",  310, 500),
    ]
    history = {
        2: approaching_history(start_x=400, end_x=300, y=500, n=15),
        3: approaching_history(start_x=410, end_x=310, y=500, n=15),
    }
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 2  # ped vs car, ped vs truck


# --- direction check (the v10 fix) ---

def test_skip_when_vehicle_moving_away_from_ped():
    """Vehicle moving AWAY from ped → no event, even if close.

    Car at 380 keeps bbox gap=110 px > VISUAL_PROXIMITY_GAP_PX=100, so
    visual-conflict shortcut does not fire; path-direction check then
    correctly skips because vehicle moves away from ped.
    """
    s = scorer(min_speed=0.5)
    dets = [det(1, "person", 250, 500), det(2, "car", 380, 500)]
    # Vehicle moving from x=320 to x=430 — away from ped at x=250
    history = {2: approaching_history(start_x=320, end_x=430, y=500, n=15)}
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert events == []


def test_skip_when_ped_to_the_side_of_vehicle_path():
    """Vehicle moving past ped (ped in different lane / on perpendicular crosswalk)
    → no event, even if 2D euclidean distance is within caution_distance.

    Captures the camera-angle false positives where a ped on the sidewalk and a
    vehicle in the road look 'close' in pixel space but the vehicle's actual
    travel path passes wide of the ped (perpendicular distance > danger_px).
    """
    s = scorer(min_speed=0.5)
    # Vehicle moving along y=500 in +x direction
    # Ped is offset 100 px in y (different depth in image / different lane)
    dets = [det(1, "person", 250, 600), det(2, "car", 200, 500)]
    history = {2: approaching_history(start_x=100, end_x=200, y=500, n=15)}
    # Euclidean dist from ped to veh: sqrt(50^2+100^2)=~112 px → within caution_px=150
    # But perpendicular dist from ped to vehicle's path: 100 px > danger_px=90
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert events == [], "ped to the side of vehicle's path must not fire"


def test_walking_ped_past_distant_stopped_queue_no_event():
    """Walking ped near (but not adjacent to) stopped traffic → NO events.

    Cars far enough away (dist > danger_px=90) that the prof-compromise tight
    radius doesn't trigger. Confirms we still avoid v11's sidewalk noise.
    """
    s = scorer(min_speed=0.5)
    dets = [
        det(1, "person", 200, 500),
        det(2, "car",    340, 500),  # dist 140 > danger_px=90
        det(3, "car",    400, 500),  # dist 200 > danger_px=90
    ]
    history = {
        1: hist(*[(200 - 3 * i, 500) for i in reversed(range(15))]),
        2: hist(*[(340, 500)] * 15),
        3: hist(*[(400, 500)] * 15),
    }
    assert s.score_frame(dets, history, frame_idx=14, fps=FPS) == []


def test_slow_vehicle_beyond_danger_radius_skipped():
    """Slow vehicle BEYOND danger_distance_px → no event.

    v15.1 close-stopped rule only fires within the danger radius. Beyond it,
    slow/creeping vehicles still skip to keep stoplight queues quiet.
    """
    s = scorer(min_speed=2.0)
    # 145 px > danger_px=90, still ≤ caution_px=150
    dets = [det(1, "person", 250, 500), det(2, "car", 395, 500)]
    history = {2: hist(*[(395 + (14 - i), 500) for i in range(15)])}
    assert s.score_frame(dets, history, frame_idx=14, fps=FPS) == []


# --- zone filtering ---

def test_no_event_when_ped_outside_zone():
    small_zone = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.int32)
    s = RiskScorer(zone_polygon=small_zone, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX)
    dets = [det(1, "person", 500, 500), det(2, "car", 550, 500)]
    assert s.score_frame(dets, {}, frame_idx=0, fps=FPS) == []


def test_event_fires_when_ped_inside_zone():
    small_zone = np.array([[0, 0], [700, 0], [700, 700], [0, 700]], dtype=np.int32)
    s = RiskScorer(zone_polygon=small_zone, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX)
    dets = [det(1, "person", 350, 350), det(2, "car", 400, 350)]
    history = {2: approaching_history(start_x=500, end_x=400, y=350, n=15)}
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 1


def test_vehicle_outside_zone_still_triggers_if_ped_inside():
    small_zone = np.array([[0, 0], [700, 0], [700, 700], [0, 700]], dtype=np.int32)
    s = RiskScorer(zone_polygon=small_zone, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX)
    dets = [det(1, "person", 350, 350), det(2, "car", 400, 350)]
    history = {2: approaching_history(start_x=500, end_x=400, y=350, n=15)}
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 1


# --- cooldown ---

def _moving_history():
    """Helper: return a track_history dict where vehicles 2 and 3 approach (200,500)."""
    return {
        2: approaching_history(start_x=400, end_x=300, y=500, n=15),
        3: approaching_history(start_x=410, end_x=310, y=500, n=15),
    }


def test_cooldown_suppresses_same_pair_on_next_frame():
    s = RiskScorer(zone_polygon=ZONE, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX, cooldown_frames=15)
    dets = [det(1, "person", 200, 500), det(2, "car", 300, 500)]
    history = _moving_history()
    events1 = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events1) == 1
    events2 = s.score_frame(dets, history, frame_idx=15, fps=FPS)
    assert events2 == []


def test_cooldown_allows_fire_after_cooldown_period():
    s = RiskScorer(zone_polygon=ZONE, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX, cooldown_frames=15)
    dets = [det(1, "person", 200, 500), det(2, "car", 300, 500)]
    history = _moving_history()
    s.score_frame(dets, history, frame_idx=14, fps=FPS)
    events = s.score_frame(dets, history, frame_idx=29, fps=FPS)
    assert len(events) == 1


def test_cooldown_independent_per_pair():
    s = RiskScorer(zone_polygon=ZONE, caution_distance_px=CAUTION_PX,
                   danger_distance_px=DANGER_PX, cooldown_frames=15)
    dets = [det(1, "person", 200, 500), det(2, "car", 300, 500), det(3, "bus", 310, 500)]
    history = _moving_history()
    events1 = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events1) == 2
    events2 = s.score_frame(dets, history, frame_idx=15, fps=FPS)
    assert events2 == []


# --- metric mode (homography enabled) ---

def _identity_ground_plane(scale_m_per_px=0.01):
    """Build a GroundPlane where image coords map directly to metres at a fixed scale.

    With scale=0.01, 100 px = 1 m → easy to reason about distances.
    """
    img_pts = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    s = scale_m_per_px * 1000  # 10 m at scale=0.01
    wld_pts = [[0, 0], [s, 0], [s, s], [0, s]]
    return GroundPlane(img_pts, wld_pts)


def metric_scorer(caution_m=1.5, danger_m=0.9, min_mph=0.5):
    return RiskScorer(
        zone_polygon=ZONE,
        caution_distance_px=CAUTION_PX,         # ignored in metric mode
        danger_distance_px=DANGER_PX,           # ignored in metric mode
        ground_plane=_identity_ground_plane(),
        caution_distance_m=caution_m,
        danger_distance_m=danger_m,
        min_vehicle_speed_m_per_s=min_mph / 2.237,
    )


def test_metric_mode_event_carries_distance_m_field():
    """Metric mode must emit distance_m alongside distance_px."""
    s = metric_scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 295, 500)]
    history = {2: approaching_history(start_x=400, end_x=295, y=500, n=15)}
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 1
    assert "distance_m" in events[0]
    # 45 px @ 0.01 m/px = 0.45 m → within danger_m=0.9 → TTC_WARNING
    assert events[0]["distance_m"] == pytest.approx(0.45, abs=0.01)
    assert events[0]["risk_label"] == "TTC_WARNING"


def test_metric_mode_proximity_when_outside_danger_inside_caution():
    """Distance 1.2 m → outside danger_m=0.9 but inside caution_m=1.5 → PROXIMITY."""
    s = metric_scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 370, 500)]  # 120 px = 1.2 m
    history = {2: approaching_history(start_x=470, end_x=370, y=500, n=15)}
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "PROXIMITY"
    assert events[0]["distance_m"] == pytest.approx(1.2, abs=0.02)


def test_metric_mode_skip_outside_caution_radius():
    """Distance 2.0 m > caution_m=1.5 → no event."""
    s = metric_scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 450, 500)]  # 200 px = 2.0 m
    history = {2: approaching_history(start_x=550, end_x=450, y=500, n=15)}
    assert s.score_frame(dets, history, frame_idx=14, fps=FPS) == []


def test_metric_mode_perpendicular_path_check_in_metres():
    """Vehicle moving along y=500, ped offset 1 m perpendicular → no event
    when danger_m=0.9 (perp dist > threshold)."""
    s = metric_scorer(caution_m=2.0, danger_m=0.9)
    dets = [det(1, "person", 250, 600), det(2, "car", 200, 500)]  # 100 px y-offset = 1.0 m
    history = {2: approaching_history(start_x=100, end_x=200, y=500, n=15)}
    assert s.score_frame(dets, history, frame_idx=14, fps=FPS) == []


def test_metric_mode_slow_vehicle_far_from_ped_skipped():
    """Slow vehicle BEYOND danger_distance_m → no event.

    The close-stopped rule only fires within the danger radius. Outside it,
    slow/stopped vehicles still skip — same v12 sidewalk-noise protection.
    """
    s = metric_scorer(min_mph=2.0)  # ~0.89 m/s
    # 200 px @ 0.01 m/px = 2.0 m > danger_m=0.9 → outside danger radius
    dets = [det(1, "person", 250, 500), det(2, "car", 450, 500)]
    history = {2: hist(*[(450, 500)] * 15)}  # truly stopped
    assert s.score_frame(dets, history, frame_idx=14, fps=FPS) == []


def test_bbox_overlap_fires_proximity_in_zone():
    """A ped whose bbox overlaps a vehicle's bbox AND who is in zone fires
    PROXIMITY immediately, regardless of vehicle speed or BEV distance.

    This catches close passes where the vehicle is parked outside the homography
    quad — the BEV extrapolation can put it 'far' even when on screen the ped
    is touching the car.
    """
    s = scorer()
    # Bboxes overlap: ped (240..260, 490..510), car (255..275, 490..510)
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (250, 500), "bottom_center": (250, 500),
           "bbox": (240, 490, 260, 510)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (265, 500), "bottom_center": (265, 500),
           "bbox": (255, 490, 275, 510)}
    # Truly stopped — would normally be skipped by the speed gate, except for
    # the close-stopped rule. We're testing the bbox-overlap branch comes first.
    history = {2: hist(*[(265, 500)] * 15)}
    events = s.score_frame([ped, veh], history, frame_idx=10, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "TTC_WARNING"


def test_bbox_overlap_does_not_fire_when_ped_outside_zone():
    """Bbox-overlap rule still requires the ped's bottom_center inside the zone."""
    small_zone = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.int32)
    s = RiskScorer(small_zone, CAUTION_PX, DANGER_PX,
                   min_vehicle_speed_px_per_frame=0.0)
    # Bboxes overlap but ped's bottom_center is well outside the small zone.
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (500, 500), "bottom_center": (500, 500),
           "bbox": (490, 490, 510, 510)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (505, 500), "bottom_center": (505, 500),
           "bbox": (495, 490, 515, 510)}
    assert s.score_frame([ped, veh], {}, frame_idx=10, fps=FPS) == []


def test_bbox_overlap_fires_in_metric_mode_regardless_of_bev_distance():
    """Even if BEV says the pair is 'far' (because the vehicle is outside the
    calibrated quad and gets extrapolated), bbox overlap still fires PROXIMITY."""
    # Identity-ish ground plane scaled so 1 px == 1 m → distance in BEV equals
    # pixel distance. Then place the bboxes far apart in pixel-anchor space but
    # arrange them to overlap visually via wide bboxes.
    img_pts = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    wld_pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    gp = GroundPlane(img_pts, wld_pts)
    s = RiskScorer(
        zone_polygon=ZONE,
        caution_distance_px=CAUTION_PX,
        danger_distance_px=DANGER_PX,
        ground_plane=gp,
        caution_distance_m=0.5,        # very tight BEV gate so anchors > 0.5 m would skip
        danger_distance_m=0.3,
        min_vehicle_speed_m_per_s=0.0,
    )
    # Anchors 100 px apart → 1.0 m in BEV → exceeds caution_m=0.5 → would be filtered.
    # But wide bboxes overlap heavily.
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (250, 500), "bottom_center": (250, 500),
           "bbox": (200, 400, 350, 600)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (350, 500), "bottom_center": (350, 500),
           "bbox": (300, 400, 450, 600)}
    history = {2: hist(*[(350, 500)] * 15)}
    events = s.score_frame([ped, veh], history, frame_idx=10, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "TTC_WARNING"
    assert events[0]["distance_m"] == 0.0


def test_robust_in_zone_check_passes_when_only_corner_inside():
    """Bbox bottom-LEFT lands inside zone but bottom-CENTER and bottom-RIGHT
    don't — should still count as in-zone. Mirrors YOLO bbox jitter at the
    polygon edge that flickers single-point tests.
    """
    # Zone covers x in [200..400], y in [400..600]. Ped bbox bottom edge runs
    # from x=180 to x=260 at y=550 — left corner inside, center on boundary,
    # right outside.
    zone = np.array([[200, 400], [400, 400], [400, 600], [200, 600]], dtype=np.int32)
    s = RiskScorer(zone, CAUTION_PX, DANGER_PX,
                   min_vehicle_speed_px_per_frame=0.0,
                   zone_hysteresis_frames=1)
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (220, 525), "bottom_center": (220, 550),
           "bbox": (180, 500, 260, 550)}  # left=180 outside, x_center=220 inside
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (300, 550), "bottom_center": (300, 550),
           "bbox": (270, 530, 330, 570)}
    history = {2: approaching_history(start_x=400, end_x=300, y=550, n=15)}
    events = s.score_frame([ped, veh], history, frame_idx=14, fps=FPS)
    assert len(events) == 1


def test_zone_hysteresis_delays_entry_resets_on_exit():
    """zone_hysteresis_frames=3 → ped must be in zone 3 consecutive frames
    before firing; one out-of-zone frame resets the streak.
    """
    s = RiskScorer(ZONE, CAUTION_PX, DANGER_PX,
                   min_vehicle_speed_px_per_frame=0.0,
                   zone_hysteresis_frames=3)
    in_dets  = [det(1, "person", 250, 500), det(2, "car", 295, 500)]
    out_dets = [det(1, "person", 1500, 1500), det(2, "car", 295, 500)]  # ped outside ZONE
    history = {2: approaching_history(start_x=400, end_x=295, y=500, n=15)}

    # Frame 14: streak goes 0 → 1 → no fire
    assert s.score_frame(in_dets, history, frame_idx=14, fps=FPS) == []
    # Frame 15: streak 1 → 2 → no fire
    assert s.score_frame(in_dets, history, frame_idx=15, fps=FPS) == []
    # Frame 16: streak 2 → 3 → fires
    assert len(s.score_frame(in_dets, history, frame_idx=16, fps=FPS)) == 1
    # Frame 17: ped briefly out of zone — streak resets to 0
    assert s.score_frame(out_dets, history, frame_idx=17, fps=FPS) == []
    # Frame 18: back in zone, streak 0 → 1 → no fire (entry hysteresis again)
    assert s.score_frame(in_dets, history, frame_idx=18, fps=FPS) == []


def test_compute_active_risks_ignores_cooldown():
    """compute_active_risks must report the pair every frame the geometric danger
    holds, even within the cooldown window — that's how the visualization stays
    continuously colored during a close pass instead of strobing.
    """
    s = scorer()
    dets = [det(1, "person", 250, 500), det(2, "car", 295, 500)]
    history = {2: approaching_history(start_x=400, end_x=295, y=500, n=15)}

    # First frame fires a logged event.
    fired = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(fired) == 1
    # Cooldown silences the next logged frame...
    assert s.score_frame(dets, history, frame_idx=15, fps=FPS) == []
    # ...but active risk for that same frame still reports the danger.
    active = s.compute_active_risks(dets, history, frame_idx=15, fps=FPS)
    assert len(active) == 1
    assert active[0]["risk_label"] in ("PROXIMITY", "TTC_WARNING")


def test_compute_active_risks_clears_when_ped_leaves_zone():
    """Active risks vanish the instant the ped's bottom_center is no longer in zone."""
    s = scorer()
    history = {2: approaching_history(start_x=400, end_x=295, y=500, n=15)}

    # Ped in zone → active.
    in_zone = [det(1, "person", 250, 500), det(2, "car", 295, 500)]
    assert len(s.compute_active_risks(in_zone, history, frame_idx=14, fps=FPS)) == 1

    # Ped same speed/distance to vehicle but now outside zone → no active risk.
    small_zone = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.int32)
    s2 = RiskScorer(small_zone, CAUTION_PX, DANGER_PX,
                    min_vehicle_speed_px_per_frame=0.0)
    out_zone = [det(1, "person", 500, 500), det(2, "car", 545, 500)]
    history2 = {2: approaching_history(start_x=650, end_x=545, y=500, n=15)}
    assert s2.compute_active_risks(out_zone, history2, frame_idx=14, fps=FPS) == []


def test_close_stopped_skipped_when_ped_walking_away():
    """Stopped vehicle within danger radius BUT ped is walking away → skip.

    No more lingering orange after the ped has cleared the close pass.
    """
    s = scorer()
    # Bbox-disjoint pair so we hit the close-stopped branch (not bbox-overlap)
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (200, 500), "bottom_center": (200, 500),
           "bbox": (190, 480, 210, 520)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (260, 500), "bottom_center": (260, 500),
           "bbox": (240, 480, 280, 520)}
    # Ped track history: walking from x=260 toward x=200 → moving AWAY from
    # the vehicle (which is at x=260). Strong leaving signal.
    leaving = hist(*[(260 - 4 * i, 500) for i in range(15)])
    truly_stopped_veh = hist(*[(260, 500)] * 15)
    events = s.score_frame(
        [ped, veh],
        {1: leaving, 2: truly_stopped_veh},
        frame_idx=14, fps=FPS,
    )
    assert events == []


def test_close_stopped_fires_when_ped_walking_toward_vehicle():
    """Mirror of the leaving test: ped approaching the stopped vehicle still fires."""
    s = scorer()
    ped = {"id": 1, "class_name": "person", "is_pedestrian": True, "is_vehicle": False,
           "center": (200, 500), "bottom_center": (200, 500),
           "bbox": (190, 480, 210, 520)}
    veh = {"id": 2, "class_name": "car", "is_pedestrian": False, "is_vehicle": True,
           "center": (260, 500), "bottom_center": (260, 500),
           "bbox": (240, 480, 280, 520)}
    approaching = hist(*[(140 + 4 * i, 500) for i in range(15)])  # toward x=200
    events = s.score_frame(
        [ped, veh],
        {1: approaching, 2: hist(*[(260, 500)] * 15)},
        frame_idx=14, fps=FPS,
    )
    assert len(events) == 1
    assert events[0]["risk_label"] == "PROXIMITY"


def test_metric_mode_stopped_vehicle_within_danger_radius_fires_proximity():
    """Stopped vehicle within danger_distance_m of an in-zone ped → PROXIMITY.

    Mirrors the pixel-mode close-stopped rule — never TTC because the vehicle
    isn't approaching, but always fire so the user sees a creeping car next
    to a crosswalk ped.
    """
    s = metric_scorer(min_mph=1.0, danger_m=0.9)
    # 50 px @ 0.01 m/px = 0.5 m ≤ danger_m=0.9 → within danger radius
    dets = [det(1, "person", 250, 500), det(2, "car", 300, 500)]
    history = {2: hist(*[(300, 500)] * 15)}  # truly stopped
    events = s.score_frame(dets, history, frame_idx=14, fps=FPS)
    assert len(events) == 1
    assert events[0]["risk_label"] == "PROXIMITY"
    assert events[0]["distance_m"] == pytest.approx(0.5, abs=0.01)
