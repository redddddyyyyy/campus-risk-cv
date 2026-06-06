import numpy as np
import pytest

from src.homography import GroundPlane, load_ground_plane


def test_identity_image_equals_world():
    """When image_points == world_points, projection is the identity."""
    pts = [[0, 0], [10, 0], [10, 5], [0, 5]]
    gp = GroundPlane(pts, pts)
    for p in pts:
        x, y = gp.image_to_world(tuple(p))
        assert x == pytest.approx(p[0], abs=1e-6)
        assert y == pytest.approx(p[1], abs=1e-6)


def test_simple_rectangle_corners_map_correctly():
    """A rectangle in image-space with width 100px x height 50px maps to a
    7.15m x 3.0m world rectangle. The four corners must land on the four
    world corners exactly."""
    image_points = [[0, 100], [100, 100], [100, 50], [0, 50]]
    world_points = [[0, 0], [7.15, 0], [7.15, 3.0], [0, 3.0]]
    gp = GroundPlane(image_points, world_points)
    for img, wld in zip(image_points, world_points):
        x, y = gp.image_to_world(tuple(img))
        assert x == pytest.approx(wld[0], abs=1e-4)
        assert y == pytest.approx(wld[1], abs=1e-4)


def test_distance_meters_on_simple_rectangle():
    """Diagonal of the world rectangle is sqrt(7.15^2 + 3.0^2)."""
    image_points = [[0, 100], [100, 100], [100, 50], [0, 50]]
    world_points = [[0, 0], [7.15, 0], [7.15, 3.0], [0, 3.0]]
    gp = GroundPlane(image_points, world_points)
    diag = gp.distance_meters((0, 100), (100, 50))
    assert diag == pytest.approx(np.hypot(7.15, 3.0), abs=1e-4)
    width = gp.distance_meters((0, 100), (100, 100))
    assert width == pytest.approx(7.15, abs=1e-4)
    depth = gp.distance_meters((0, 100), (0, 50))
    assert depth == pytest.approx(3.0, abs=1e-4)


def test_perspective_distortion_compresses_far_objects():
    """A trapezoidal image rectangle (perspective view) maps to a true
    rectangle in world space. Equal world distances near and far map
    back to *unequal* image distances — and vice versa: equal image
    distances near and far span unequal world distances."""
    # Image: trapezoid (foreground wider than background)
    image_points = [[200, 800], [1200, 800], [900, 400], [500, 400]]
    # World: 7.15m x 3.0m rectangle (peds walk 7.15m across, 3m deep)
    world_points = [[0, 0], [7.15, 0], [7.15, 3.0], [0, 3.0]]
    gp = GroundPlane(image_points, world_points)

    # Same world distance (1m) should map to *more* pixels near the camera
    # than far. So a 100-pixel sweep in the foreground covers fewer metres
    # than a 100-pixel sweep in the background.
    near_world = gp.distance_meters((400, 800), (500, 800))   # 100px foreground
    far_world  = gp.distance_meters((600, 400), (700, 400))   # 100px background
    assert far_world > near_world, (
        f"Expected far 100px > near 100px in metres, got near={near_world:.3f} "
        f"far={far_world:.3f}"
    )


def test_load_ground_plane_returns_none_when_no_block():
    assert load_ground_plane({}) is None
    assert load_ground_plane({"danger_zone": [[0, 0]]}) is None


def test_load_ground_plane_builds_from_dict():
    cfg = {
        "homography": {
            "image_points": [[0, 100], [100, 100], [100, 50], [0, 50]],
            "world_points": [[0, 0], [7.15, 0], [7.15, 3.0], [0, 3.0]],
        }
    }
    gp = load_ground_plane(cfg)
    assert gp is not None
    assert gp.distance_meters((0, 100), (100, 100)) == pytest.approx(7.15, abs=1e-4)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        GroundPlane([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])  # 3 points
    with pytest.raises(ValueError):
        GroundPlane([[0, 0, 0]] * 4, [[0, 0]] * 4)  # 4x3 instead of 4x2
