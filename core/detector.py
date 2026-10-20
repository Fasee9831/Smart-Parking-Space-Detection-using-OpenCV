"""
Detection Engine for Parking Space Occupancy.
Provides Edge/Texture Density analysis, Deep Learning vehicle detection, and Hybrid ensemble mode.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Torchvision detection support (lazy loaded)
_TORCH_AVAILABLE = False
_torch_model = None
_torch_device = None

def _init_torch_model():
    global _TORCH_AVAILABLE, _torch_model, _torch_device
    if _torch_model is not None:
        return True
    try:
        import torch
        import torchvision.models.detection as d
        
        _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load Faster R-CNN with MobileNetV3 backbone (lightweight and fast)
        _torch_model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        _torch_model.to(_torch_device)
        _torch_model.eval()
        _TORCH_AVAILABLE = True
        return True
    except Exception as e:
        print(f"[Detector] Torch model initialization fallback: {e}")
        _TORCH_AVAILABLE = False
        return False


class DetectionResult:
    """Holds detection status for a single slot."""

    def __init__(
        self,
        slot_id: str,
        is_occupied: bool,
        confidence: float,
        method: str,
        vehicle_type: Optional[str] = None,
        edge_ratio: float = 0.0,
        vehicle_bbox: Optional[List[int]] = None
    ):
        self.slot_id = slot_id
        self.is_occupied = is_occupied
        self.confidence = confidence
        self.method = method
        self.vehicle_type = vehicle_type
        self.edge_ratio = edge_ratio
        self.vehicle_bbox = vehicle_bbox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "is_occupied": self.is_occupied,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "vehicle_type": self.vehicle_type,
            "edge_ratio": round(self.edge_ratio, 3),
            "vehicle_bbox": self.vehicle_bbox
        }


class ParkingDetector:
    """Multi-model parking occupancy detector."""

    MODE_EDGE = "Edge / Texture Density"
    MODE_DL = "Deep Learning (MobileNetV3)"
    MODE_HYBRID = "Hybrid Ensemble"

    # COCO class IDs for vehicles: 3: car, 4: motorcycle, 6: bus, 8: truck
    COCO_VEHICLE_LABELS = {
        3: "car",
        4: "motorcycle",
        6: "bus",
        8: "truck"
    }

    def __init__(
        self,
        mode: str = MODE_HYBRID,
        edge_threshold: float = 0.08,
        dl_conf_threshold: float = 0.40,
        overlap_threshold: float = 0.25,
        canny_low: int = 50,
        canny_high: int = 150
    ):
        self.mode = mode
        self.edge_threshold = edge_threshold
        self.dl_conf_threshold = dl_conf_threshold
        self.overlap_threshold = overlap_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high

    def compute_edge_mask(self, image: np.ndarray) -> np.ndarray:
        """Preprocess frame into edge & contour representation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # Dilate slightly to connect vehicle contour fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return dilated

    def compute_slot_edge_density(self, slot_bbox: List[int], edge_mask: np.ndarray) -> Tuple[float, int, int]:
        """Calculates normalized edge ratio for a slot's region of interest."""
        h, w = edge_mask.shape[:2]
        x1 = max(0, min(slot_bbox[0], w - 1))
        y1 = max(0, min(slot_bbox[1], h - 1))
        x2 = max(x1 + 1, min(slot_bbox[2], w))
        y2 = max(y1 + 1, min(slot_bbox[3], h))

        roi = edge_mask[y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0, 0, 0

        edge_pixels = cv2.countNonZero(roi)
        ratio = edge_pixels / float(area)
        return ratio, edge_pixels, area

    def detect_vehicles_dl(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run deep learning vehicle detector on full frame."""
        if not _init_torch_model():
            return []

        import torch
        from torchvision.transforms import functional as F

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).unsqueeze(0).to(_torch_device)

        with torch.no_grad():
            predictions = _torch_model(tensor)[0]

        boxes = predictions["boxes"].cpu()
        labels = predictions["labels"].cpu()
        scores = predictions["scores"].cpu()

        vehicles = []
        for box, label, score in zip(boxes, labels, scores):
            lbl_id = label.item()
            sc = score.item()
            if lbl_id in self.COCO_VEHICLE_LABELS and sc >= self.dl_conf_threshold:
                b = [int(v) for v in box.tolist()]
                vehicles.append({
                    "type": self.COCO_VEHICLE_LABELS[lbl_id],
                    "confidence": sc,
                    "bbox": b  # [x1, y1, x2, y2]
                })

        return vehicles

    @staticmethod
    def calculate_box_overlap_ratio(slot_box: List[int], veh_box: List[int]) -> float:
        """Calculate intersection over slot area ratio."""
        sx1, sy1, sx2, sy2 = slot_box
        vx1, vy1, vx2, vy2 = veh_box

        ix1 = max(sx1, vx1)
        iy1 = max(sy1, vy1)
        ix2 = min(sx2, vx2)
        iy2 = min(sy2, vy2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        return intersection / float(slot_area)

    def process_frame(
        self,
        image: np.ndarray,
        slots: List[Any],
        mode_override: Optional[str] = None
    ) -> Tuple[List[DetectionResult], List[Dict[str, Any]], np.ndarray]:
        """
        Process a frame against all defined slots.
        Returns:
            - List of DetectionResult objects
            - List of detected vehicle bboxes from DL
            - Edge mask image for inspection
        """
        active_mode = mode_override or self.mode
        edge_mask = self.compute_edge_mask(image)
        detected_vehicles = []

        if active_mode in [self.MODE_DL, self.MODE_HYBRID]:
            detected_vehicles = self.detect_vehicles_dl(image)

        results = []

        for slot in slots:
            bbox = slot.bbox
            edge_ratio, _, _ = self.compute_slot_edge_density(bbox, edge_mask)

            is_occupied = False
            confidence = 0.0
            veh_type = None
            veh_box = None

            # Check overlap with any DL detected vehicle
            max_overlap = 0.0
            best_veh = None
            for veh in detected_vehicles:
                overlap = self.calculate_box_overlap_ratio(bbox, veh["bbox"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_veh = veh

            if active_mode == self.MODE_EDGE:
                is_occupied = edge_ratio >= self.edge_threshold
                confidence = min(1.0, edge_ratio / (self.edge_threshold * 2.0))
                veh_type = "vehicle" if is_occupied else None

            elif active_mode == self.MODE_DL:
                if best_veh and max_overlap >= self.overlap_threshold:
                    is_occupied = True
                    confidence = best_veh["confidence"]
                    veh_type = best_veh["type"]
                    veh_box = best_veh["bbox"]
                else:
                    is_occupied = False
                    confidence = 1.0 - (max_overlap / max(0.01, self.overlap_threshold))
                    veh_type = None

            else:  # MODE_HYBRID
                has_dl_match = best_veh and max_overlap >= self.overlap_threshold
                has_edge_match = edge_ratio >= self.edge_threshold

                if has_dl_match:
                    is_occupied = True
                    confidence = best_veh["confidence"]
                    veh_type = best_veh["type"]
                    veh_box = best_veh["bbox"]
                elif has_edge_match and edge_ratio > (self.edge_threshold * 1.5):
                    # High edge density even if DL missed it (e.g. partial view or dark vehicle)
                    is_occupied = True
                    confidence = min(0.95, edge_ratio / (self.edge_threshold * 2.0))
                    veh_type = "vehicle"
                else:
                    is_occupied = False
                    confidence = 1.0 - min(1.0, edge_ratio / self.edge_threshold)

            res = DetectionResult(
                slot_id=slot.id,
                is_occupied=is_occupied,
                confidence=confidence,
                method=active_mode,
                vehicle_type=veh_type,
                edge_ratio=edge_ratio,
                vehicle_bbox=veh_box
            )
            results.append(res)

        return results, detected_vehicles, edge_mask
