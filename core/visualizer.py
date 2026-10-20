"""
Computer Vision Visualizer Module.
Renders semi-transparent slot polygons, status pills, vehicle overlays, and HUD banners.
"""

from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class ParkingVisualizer:
    """Renders overlays, HUDs, and badges onto video/image frames."""

    # BGR Colors
    COLOR_VACANT_REGULAR = (46, 204, 113)     # Fresh Green
    COLOR_OCCUPIED = (50, 50, 230)            # Vivid Red
    COLOR_VACANT_EV = (235, 150, 52)          # Electric Cyan / Light Blue
    COLOR_VACANT_HANDICAP = (20, 200, 245)    # Amber Yellow
    COLOR_VACANT_VIP = (180, 80, 160)         # Royal Purple
    COLOR_SELECTED = (255, 255, 255)          # White border for selected
    COLOR_TEXT = (255, 255, 255)

    @classmethod
    def get_slot_color(cls, slot_type: str, is_occupied: bool) -> Tuple[int, int, int]:
        """Returns BGR color based on occupancy and slot classification."""
        if is_occupied:
            return cls.COLOR_OCCUPIED
        if slot_type == "EV Charging":
            return cls.COLOR_VACANT_EV
        elif slot_type == "Handicap":
            return cls.COLOR_VACANT_HANDICAP
        elif slot_type == "VIP / Reserved":
            return cls.COLOR_VACANT_VIP
        return cls.COLOR_VACANT_REGULAR

    @classmethod
    def draw_hud_banner(
        cls,
        image: np.ndarray,
        summary: Dict[str, Any],
        show_fps: Optional[float] = None
    ) -> np.ndarray:
        """Draws top HUD banner with facility KPI cards."""
        h, w = image.shape[:2]
        banner_h = 48
        overlay = image.copy()

        # Dark translucent banner
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 30), -1)
        cv2.line(overlay, (0, banner_h), (w, banner_h), (60, 60, 75), 2)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # Card elements
        total = summary.get("total_slots", 0)
        occ = summary.get("occupied_slots", 0)
        vac = summary.get("vacant_slots", 0)
        rate = summary.get("occupancy_rate", 0.0)
        revenue = summary.get("revenue", 0.0)

        # Draw metrics
        items = [
            ("SPACES", f"{total}", (200, 200, 200)),
            ("VACANT", f"{vac}", cls.COLOR_VACANT_REGULAR),
            ("OCCUPIED", f"{occ}", cls.COLOR_OCCUPIED),
            ("RATE", f"{rate:.0f}%", (245, 175, 40) if rate > 80 else (200, 200, 200)),
            ("REVENUE", f"${revenue:.2f}", (46, 204, 113))
        ]

        if show_fps is not None:
            items.append(("FPS", f"{show_fps:.1f}", (180, 180, 180)))

        spacing = w // (len(items) + 1)
        for idx, (label, val, color) in enumerate(items):
            cx = (idx + 1) * spacing
            text_label = f"{label}: "
            (lw, _), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(image, text_label, (cx - lw, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)
            cv2.putText(image, val, (cx + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return image

    @classmethod
    def render_overlay(
        cls,
        image: np.ndarray,
        slots: List[Any],
        detection_results: List[Any],
        detected_vehicles: Optional[List[Dict[str, Any]]] = None,
        selected_slot_id: Optional[str] = None,
        show_ids: bool = True,
        show_confidence: bool = True,
        show_dl_boxes: bool = False,
        show_hud: bool = True,
        summary: Optional[Dict[str, Any]] = None,
        fps: Optional[float] = None,
        alpha: float = 0.28
    ) -> np.ndarray:
        """
        Renders complete parking visualization onto the frame.
        """
        canvas = image.copy()
        fill_overlay = image.copy()

        res_map = {r.slot_id: r for r in detection_results}

        # 1. Fill translucent colored areas for each slot
        for slot in slots:
            res = res_map.get(slot.id)
            is_occ = res.is_occupied if res else False
            color = cls.get_slot_color(slot.slot_type, is_occ)

            x1, y1, x2, y2 = slot.bbox
            cv2.rectangle(fill_overlay, (x1, y1), (x2, y2), color, -1)

        # Blend fills
        cv2.addWeighted(fill_overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        # 2. Draw crisp borders and badges
        for slot in slots:
            res = res_map.get(slot.id)
            is_occ = res.is_occupied if res else False
            color = cls.get_slot_color(slot.slot_type, is_occ)

            x1, y1, x2, y2 = slot.bbox
            is_sel = (selected_slot_id == slot.id)

            # Border
            thickness = 3 if is_sel else 2
            border_col = cls.COLOR_SELECTED if is_sel else color
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, thickness)

            # Badge / Label text
            if show_ids:
                status_short = "OCC" if is_occ else "FREE"
                tag = f"{slot.id} [{status_short}]"
                if show_confidence and res:
                    tag += f" {int(res.confidence * 100)}%"

                font_scale = 0.42 if (x2 - x1) < 140 else 0.48
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                badge_y1 = max(4, y1 - th - 8)
                badge_y2 = badge_y1 + th + 6
                badge_x2 = min(canvas.shape[1] - 4, x1 + tw + 10)

                # Dark pill background
                cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), (20, 20, 25), -1)
                cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, 1)

                cv2.putText(
                    canvas,
                    tag,
                    (x1 + 5, badge_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    cls.COLOR_TEXT,
                    1,
                    cv2.LINE_AA
                )

        # 3. Optional DL Vehicle boxes
        if show_dl_boxes and detected_vehicles:
            for veh in detected_vehicles:
                vx1, vy1, vx2, vy2 = veh["bbox"]
                vlabel = f"{veh['type']} {int(veh['confidence'] * 100)}%"
                cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (255, 140, 0), 1)
                cv2.putText(
                    canvas,
                    vlabel,
                    (vx1, max(12, vy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 140, 0),
                    1,
                    cv2.LINE_AA
                )

        # 4. HUD Banner
        if show_hud and summary:
            canvas = cls.draw_hud_banner(canvas, summary, show_fps=fps)

        return canvas
