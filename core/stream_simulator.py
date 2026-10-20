"""
Parking Lot Video Stream & Traffic Simulator Module.
Supports video files, webcam streams, and dynamic live traffic simulation on static images.
"""

import time
import random
from typing import Optional, Tuple, Generator, List, Any
import cv2
import numpy as np


class StreamSimulator:
    """Simulates live camera streams or plays back video files/webcam."""

    def __init__(self, base_image_path: str = "parking_lot.png"):
        self.base_image_path = base_image_path
        self.base_image = cv2.imread(base_image_path)
        if self.base_image is None:
            # Fallback 1280x720 blank
            self.base_image = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Simulation states
        self.car_library: List[np.ndarray] = []
        self._extract_car_patches()
        self.active_simulated_slots: Dict[str, np.ndarray] = {}
        self.last_event_time = time.time()
        self.event_interval = 4.0  # seconds between car arrivals/departures

    def _extract_car_patches(self):
        """Extract a few car patches from the base image to use for synthetic traffic."""
        if self.base_image is None:
            return

        h, w = self.base_image.shape[:2]
        # Coordinates of parked cars in parking_lot.png
        known_patches = [
            (66, 456, 212, 559),
            (391, 408, 587, 502),
            (748, 449, 846, 560),
            (1167, 478, 1371, 561),
            (344, 660, 521, 766)
        ]

        for x1, y1, x2, y2 in known_patches:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            patch = self.base_image[y1:y2, x1:x2].copy()
            if patch.size > 0:
                self.car_library.append(patch)

    def generate_simulated_frame(self, slots: List[Any]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Generates a frame where cars arrive and depart randomly from vacant/occupied slots.
        Returns: (frame, event_message)
        """
        frame = self.base_image.copy()
        now = time.time()
        event_msg = None

        if now - self.last_event_time > self.event_interval and slots:
            self.last_event_time = now
            # Choose random action: 50% arrive, 50% depart
            if random.random() < 0.55 and len(self.active_simulated_slots) < len(slots):
                # Car Arrives in a vacant spot
                available_slots = [s for s in slots if s.id not in self.active_simulated_slots]
                if available_slots:
                    target_slot = random.choice(available_slots)
                    if self.car_library:
                        patch = random.choice(self.car_library)
                        target_w = target_slot.width
                        target_h = target_slot.height
                        resized_patch = cv2.resize(patch, (target_w, target_h))
                        self.active_simulated_slots[target_slot.id] = resized_patch
                        event_msg = f"Vehicle arrived at slot {target_slot.id}"
            else:
                # Car Departs
                if self.active_simulated_slots:
                    departing_id = random.choice(list(self.active_simulated_slots.keys()))
                    del self.active_simulated_slots[departing_id]
                    event_msg = f"Vehicle departed from slot {departing_id}"

        # Overlay simulated cars
        slot_map = {s.id: s for s in slots}
        for sid, patch in self.active_simulated_slots.items():
            if sid in slot_map:
                s = slot_map[sid]
                x1, y1, x2, y2 = s.bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
                x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))

                resized = cv2.resize(patch, (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = resized

        return frame, event_msg
