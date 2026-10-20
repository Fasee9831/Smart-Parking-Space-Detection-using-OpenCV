"""
Parking Slot Management and Serialization Module
Handles slot definitions, grid generation, persistence, and querying.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

SLOT_TYPES = ["Regular", "EV Charging", "Handicap", "VIP / Reserved"]

SLOT_COLORS = {
    "Regular": {"available": (46, 204, 113), "occupied": (231, 76, 60)},      # Green / Red
    "EV Charging": {"available": (52, 152, 219), "occupied": (231, 76, 60)},  # Blue / Red
    "Handicap": {"available": (241, 196, 15), "occupied": (231, 76, 60)},     # Yellow / Red
    "VIP / Reserved": {"available": (155, 89, 182), "occupied": (231, 76, 60)}# Purple / Red
}


class ParkingSlot:
    """Represents an individual parking slot in the facility."""

    def __init__(
        self,
        slot_id: str,
        bbox: List[int],
        slot_type: str = "Regular",
        polygon: Optional[List[List[int]]] = None
    ):
        self.id = slot_id
        self.slot_type = slot_type if slot_type in SLOT_TYPES else "Regular"
        self.bbox = [int(v) for v in bbox]  # [x1, y1, x2, y2]
        self.polygon = polygon  # Optional 4-point quadrilateral for angled spots
        
        # Runtime states
        self.is_occupied: bool = False
        self.confidence: float = 0.0
        self.vehicle_type: Optional[str] = None
        self.occupied_since: Optional[float] = None
        self.vacated_at: Optional[float] = None
        self.total_occupied_duration: float = 0.0  # seconds

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def contains_point(self, px: int, py: int) -> bool:
        """Check if point (px, py) falls inside slot boundaries."""
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def to_dict(self) -> Dict[str, Any]:
        """Serialize slot to dictionary."""
        d = {
            "id": self.id,
            "type": self.slot_type,
            "bbox": self.bbox
        }
        if self.polygon:
            d["polygon"] = self.polygon
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParkingSlot":
        return cls(
            slot_id=data["id"],
            bbox=data["bbox"],
            slot_type=data.get("type", "Regular"),
            polygon=data.get("polygon")
        )


class SlotManager:
    """Manages collections of parking slots, serialization, and grid generation."""

    def __init__(self, default_file: Optional[str] = None):
        self.slots: List[ParkingSlot] = []
        self.file_path = default_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "default_slots.json"
        )
        if os.path.exists(self.file_path):
            self.load_from_json(self.file_path)

    def add_slot(self, slot: ParkingSlot) -> None:
        """Add or update slot with same ID."""
        for idx, existing in enumerate(self.slots):
            if existing.id == slot.id:
                self.slots[idx] = slot
                return
        self.slots.append(slot)

    def remove_slot(self, slot_id: str) -> bool:
        """Remove a slot by its ID."""
        initial_len = len(self.slots)
        self.slots = [s for s in self.slots if s.id != slot_id]
        return len(self.slots) < initial_len

    def get_slot(self, slot_id: str) -> Optional[ParkingSlot]:
        """Retrieve slot by ID."""
        for s in self.slots:
            if s.id == slot_id:
                return s
        return None

    def find_slot_at_point(self, px: int, py: int) -> Optional[ParkingSlot]:
        """Find slot containing the given point."""
        for s in reversed(self.slots):  # check top-most first
            if s.contains_point(px, py):
                return s
        return None

    def clear(self) -> None:
        """Clear all slots."""
        self.slots.clear()

    def generate_grid(
        self,
        start_x: int,
        start_y: int,
        rows: int,
        cols: int,
        slot_w: int,
        slot_h: int,
        gap_x: int = 15,
        gap_y: int = 25,
        prefix: str = "S",
        slot_type: str = "Regular"
    ) -> List[ParkingSlot]:
        """Generate a regular matrix/grid of parking spaces."""
        new_slots = []
        count = 1
        for r in range(rows):
            for c in range(cols):
                x1 = start_x + c * (slot_w + gap_x)
                y1 = start_y + r * (slot_h + gap_y)
                x2 = x1 + slot_w
                y2 = y1 + slot_h
                slot_id = f"{prefix}-{count:02d}"
                slot = ParkingSlot(slot_id, [x1, y1, x2, y2], slot_type)
                new_slots.append(slot)
                count += 1
        self.slots.extend(new_slots)
        return new_slots

    def save_to_json(self, target_path: Optional[str] = None) -> str:
        """Save slots configuration to JSON."""
        dest = target_path or self.file_path
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        data = [s.to_dict() for s in self.slots]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return dest

    def load_from_json(self, source_path: Optional[str] = None) -> bool:
        """Load slots configuration from JSON."""
        path = source_path or self.file_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.slots = [ParkingSlot.from_dict(d) for d in data]
            return True
        except Exception as e:
            print(f"Error loading slots from {path}: {e}")
            return False

    def load_preset(self, preset_name: str) -> bool:
        """Load a predefined configuration preset."""
        preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        target = os.path.join(preset_dir, f"{preset_name}.json")
        return self.load_from_json(target)
