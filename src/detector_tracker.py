from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional

PEDESTRIAN_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}  # bicycle dropped — bike racks were false-positiving
ALL_RELEVANT = PEDESTRIAN_CLASSES | VEHICLE_CLASSES

# Open Images V7 model classes used for context-only detection (no risk scoring)
TREE_CLASSES = {"Tree", "Palm tree"}


class DetectorTracker:
    def __init__(
        self,
        model_path: str = "yolo12n.pt",
        conf: float = 0.35,
        tree_model_path: Optional[str] = "yolov8n-oiv7.pt",
        tree_conf: float = 0.20,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tree_model = YOLO(tree_model_path) if tree_model_path else None
        self.tree_conf = tree_conf
        self._next_tree_id = -1

    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs YOLO + ByteTrack on one frame for peds/vehicles.
        Optionally runs a second OIv7 pass for tree detections (untracked, context-only).
        Returns a list of detection dicts:
          id, class_name, is_pedestrian, is_vehicle, is_tree, bbox, center, bottom_center
        """
        detections: List[Dict] = []

        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf)
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids     = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            xyxy    = boxes.xyxy.cpu().numpy()
            for track_id, cls_id, box in zip(ids, classes, xyxy):
                class_name = self.model.names[cls_id]
                if class_name not in ALL_RELEVANT:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                detections.append({
                    'id':           int(track_id),
                    'class_name':   class_name,
                    'is_pedestrian': class_name in PEDESTRIAN_CLASSES,
                    'is_vehicle':    class_name in VEHICLE_CLASSES,
                    'is_tree':       False,
                    'bbox':          (x1, y1, x2, y2),
                    'center':        (cx, cy),
                    'bottom_center': (cx, int(y2)),
                })

        if self.tree_model is not None:
            tree_results = self.tree_model(frame, verbose=False, conf=self.tree_conf)
            tboxes = tree_results[0].boxes
            if tboxes is not None and len(tboxes) > 0:
                t_classes = tboxes.cls.cpu().numpy().astype(int)
                t_xyxy    = tboxes.xyxy.cpu().numpy()
                for cls_id, box in zip(t_classes, t_xyxy):
                    class_name = self.tree_model.names[cls_id]
                    if class_name not in TREE_CLASSES:
                        continue
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    detections.append({
                        'id':           self._next_tree_id,
                        'class_name':   class_name.lower(),
                        'is_pedestrian': False,
                        'is_vehicle':    False,
                        'is_tree':       True,
                        'bbox':          (x1, y1, x2, y2),
                        'center':        (cx, cy),
                        'bottom_center': (cx, int(y2)),
                    })
                    self._next_tree_id -= 1

        return detections
