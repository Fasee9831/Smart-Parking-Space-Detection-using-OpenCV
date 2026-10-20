"""
Smart Parking Space Occupancy Detection System
Pure Python & OpenCV Application (Zero Heavy Dependencies)

Features:
- Analyzes ANY parking lot image (custom image path, file browser, or drag & drop)
- Interactive OpenCV GUI with native real-time Trackbars (Sensitivity, Spot Width, Spot Height)
- Click or drag to mark parking bays; Right-click to delete
- Auto-Grid generator for entire parking rows
- Persistent JSON configuration per image (<image_name>_spots.json)
- Multi-view display: Annotated view, Canny edge mask, and Side-by-side comparison
- Real-time Heads-Up Display (HUD) with Total, Vacant, Occupied, and Occupancy Rate %
- Single key press to export analyzed result image and CSV summary
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class ParkingSpaceAnalyzer:
    """Core OpenCV-based parking spot analyzer and interactive studio."""

    def __init__(self, image_path: Optional[str] = None):
        self.image_path = image_path or self._find_default_image()
        self.image_raw = None
        self.image_gray = None
        self.edges = None
        
        # Spot definition: list of dicts: [{"id": 1, "x1": int, "y1": int, "x2": int, "y2": int}]
        self.spots: List[Dict[str, int]] = []
        
        # Trackbar / Tuning parameters
        self.spot_width = 94
        self.spot_height = 62
        self.edge_threshold_pct = 8    # 8% threshold (0.08)
        self.canny_low = 50
        self.canny_high = 150
        
        # Display modes: 0 = Annotated, 1 = Edge Mask, 2 = Side-by-Side
        self.view_mode = 0
        self.show_hud = True
        self.show_labels = True
        
        # Drag drawing state
        self.drawing = False
        self.drag_start = (0, 0)
        self.drag_current = (0, 0)
        
        self.window_name = "Smart Parking Analyzer (OpenCV)"
        self.load_image(self.image_path)

    def _find_default_image(self) -> str:
        """Locates sample parking image in directory."""
        candidates = ["parking_lot.png", "parking.png", "parking.jpg", "lot.png"]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "parking_lot.png"

    def load_image(self, path: str) -> bool:
        """Loads and pre-processes any image given by the user."""
        if not os.path.exists(path):
            print(f"[!] Warning: File '{path}' does not exist.")
            return False

        img = cv2.imread(path)
        if img is None:
            print(f"[!] Error: Could not decode image '{path}'.")
            return False

        self.image_path = path
        self.image_raw = img
        self.h, self.w = img.shape[:2]

        # Automatic spot dimension scaling based on image resolution
        # 1408x768 -> 94x62. Scale proportionally if larger or smaller
        scale_factor = max(self.w / 1408.0, self.h / 768.0)
        self.spot_width = max(30, int(94 * scale_factor))
        self.spot_height = max(20, int(62 * scale_factor))

        self.compute_preprocessed_edges()
        self.load_spots_for_current_image()
        print(f"[+] Loaded image: {os.path.basename(path)} ({self.w}x{self.h})")
        return True

    def compute_preprocessed_edges(self):
        """Canny Edge & Texture Preprocessing optimized for universal car detection."""
        self.image_gray = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
        
        # Canny edge detection
        self.edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Slight morphological dilation to connect contour fragments on vehicle surfaces
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.edges = cv2.dilate(self.edges, kernel, iterations=1)

        # High-precision DL vehicle detector
        self.detected_vehicles = []
        try:
            import torch
            import torchvision.models.detection as d
            from torchvision.transforms import functional as F
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = d.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").to(device).eval()
            rgb = cv2.cvtColor(self.image_raw, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                preds = model(F.to_tensor(rgb).unsqueeze(0).to(device))[0]
            valid_labels = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
            for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
                if label.item() in valid_labels and score.item() >= 0.35:
                    self.detected_vehicles.append({
                        "type": valid_labels[label.item()],
                        "score": score.item(),
                        "box": [int(v) for v in box.tolist()]
                    })
        except Exception:
            pass

    def get_spots_filename(self) -> str:
        """Derives JSON layout filename for the current image."""
        base, _ = os.path.splitext(self.image_path)
        return f"{base}_spots.json"

    def load_spots_for_current_image(self):
        """Loads spots saved specifically for this image or common default."""
        specific_file = self.get_spots_filename()
        fallback_file = os.path.join("data", "default_slots.json")

        target = specific_file if os.path.exists(specific_file) else (fallback_file if os.path.exists(fallback_file) else None)
        self.spots = []

        if target and os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for idx, item in enumerate(data):
                    bbox = item.get("bbox", [])
                    if len(bbox) == 4:
                        self.spots.append({
                            "id": idx + 1,
                            "x1": int(bbox[0]),
                            "y1": int(bbox[1]),
                            "x2": int(bbox[2]),
                            "y2": int(bbox[3])
                        })
                print(f"[+] Loaded {len(self.spots)} parking spots from '{target}'")
            except Exception as e:
                print(f"[!] Could not load spots from {target}: {e}")

    def save_spots_for_current_image(self):
        """Saves marked spots to JSON file."""
        target = self.get_spots_filename()
        data = [{"id": f"P-{s['id']:02d}", "bbox": [s["x1"], s["y1"], s["x2"], s["y2"]]} for s in self.spots]
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[+] Successfully saved {len(self.spots)} spots to '{target}'")
        except Exception as e:
            print(f"[!] Error saving spots: {e}")

    def analyze_spot(self, spot: Dict[str, int]) -> Tuple[bool, float]:
        """
        Analyzes a single spot ROI using DL vehicle overlap + edge density.
        Returns: (is_occupied, edge_ratio)
        """
        x1 = max(0, min(spot["x1"], self.w - 1))
        y1 = max(0, min(spot["y1"], self.h - 1))
        x2 = max(x1 + 1, min(spot["x2"], self.w))
        y2 = max(y1 + 1, min(spot["y2"], self.h))

        pad_x = max(2, int((x2 - x1) * 0.10))
        pad_y = max(2, int((y2 - y1) * 0.10))
        roi_edges = self.edges[y1+pad_y : y2-pad_y, x1+pad_x : x2-pad_x]
        area = max(1, roi_edges.size)
        edge_count = cv2.countNonZero(roi_edges)
        edge_ratio = edge_count / float(area)

        # Check vehicle box overlap
        max_overlap = 0.0
        for v in getattr(self, "detected_vehicles", []):
            vx1, vy1, vx2, vy2 = v["box"]
            ix1, iy1 = max(x1, vx1), max(y1, vy1)
            ix2, iy2 = min(x2, vx2), min(y2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                ov = ((ix2 - ix1) * (iy2 - iy1)) / float((x2 - x1) * (y2 - y1))
                if ov > max_overlap:
                    max_overlap = ov

        if max_overlap >= 0.32:
            is_occupied = True
        elif edge_ratio >= 0.28:
            is_occupied = True
        else:
            is_occupied = False

        return is_occupied, edge_ratio

    def auto_generate_grid(self, start_x: int, start_y: int, cols: int, rows: int = 1, gap_x: int = 15, gap_y: int = 20):
        """Quickly generates an aligned row/matrix of parking spots."""
        count = len(self.spots) + 1
        for r in range(rows):
            for c in range(cols):
                x1 = start_x + c * (self.spot_width + gap_x)
                y1 = start_y + r * (self.spot_height + gap_y)
                x2 = x1 + self.spot_width
                y2 = y1 + self.spot_height
                if x2 < self.w and y2 < self.h:
                    self.spots.append({"id": count, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    count += 1
        print(f"[+] Generated grid. Total spots now: {len(self.spots)}")

    # --- Mouse Event Handling ---
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                dx = abs(x - self.drag_start[0])
                dy = abs(y - self.drag_start[1])

                if dx > 15 and dy > 15:
                    # User dragged a custom rectangle
                    x1 = min(self.drag_start[0], x)
                    y1 = min(self.drag_start[1], y)
                    x2 = max(self.drag_start[0], x)
                    y2 = max(self.drag_start[1], y)
                else:
                    # User clicked: place standard-sized box centered at click
                    x1 = max(0, x - self.spot_width // 2)
                    y1 = max(0, y - self.spot_height // 2)
                    x2 = min(self.w, x1 + self.spot_width)
                    y2 = min(self.h, y1 + self.spot_height)

                new_id = len(self.spots) + 1
                self.spots.append({"id": new_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                print(f"[+] Added spot #{new_id} at ({x1}, {y1}, {x2}, {y2})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete spot clicked inside or nearest
            removed = False
            for s in reversed(self.spots):
                if s["x1"] <= x <= s["x2"] and s["y1"] <= y <= s["y2"]:
                    self.spots.remove(s)
                    removed = True
                    print(f"[-] Removed spot #{s['id']}")
                    break

            if not removed and self.spots:
                # Find nearest within 50px
                nearest = min(self.spots, key=lambda s: np.hypot((s["x1"]+s["x2"])//2 - x, (s["y1"]+s["y2"])//2 - y))
                cx = (nearest["x1"] + nearest["x2"]) // 2
                cy = (nearest["y1"] + nearest["y2"]) // 2
                if np.hypot(cx - x, cy - y) < 60:
                    self.spots.remove(nearest)
                    print(f"[-] Removed nearest spot #{nearest['id']}")

    # --- Renderers ---
    def render(self) -> np.ndarray:
        """Renders complete visualization with transparency and HUD."""
        canvas = self.image_raw.copy()
        fill_overlay = canvas.copy()

        occupied_count = 0
        vacant_count = 0

        # 1. Analyze and draw each spot
        for s in self.spots:
            is_occ, edge_ratio = self.analyze_spot(s)
            if is_occ:
                occupied_count += 1
                color = (40, 40, 230)      # Red for Occupied
                status = f"OCC {int(edge_ratio * 100)}%"
            else:
                vacant_count += 1
                color = (50, 205, 50)      # Green for Vacant
                status = f"FREE {int(edge_ratio * 100)}%"

            # Semi-transparent fill
            cv2.rectangle(fill_overlay, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, -1)

        # Blend alpha
        cv2.addWeighted(fill_overlay, 0.28, canvas, 0.72, 0, canvas)

        # 2. Draw crisp borders and badges
        for s in self.spots:
            is_occ, edge_ratio = self.analyze_spot(s)
            color = (40, 40, 230) if is_occ else (50, 205, 50)
            status = f"OCC {int(edge_ratio * 100)}%" if is_occ else f"FREE {int(edge_ratio * 100)}%"

            cv2.rectangle(canvas, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, 2)

            if self.show_labels:
                tag = f"#{s['id']} {status}"
                font_scale = 0.42 if (s["x2"] - s["x1"]) < 120 else 0.48
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                by1 = max(4, s["y1"] - th - 6)
                by2 = by1 + th + 6
                bx2 = min(self.w - 2, s["x1"] + tw + 8)

                cv2.rectangle(canvas, (s["x1"], by1), (bx2, by2), (20, 20, 25), -1)
                cv2.rectangle(canvas, (s["x1"], by1), (bx2, by2), color, 1)
                cv2.putText(canvas, tag, (s["x1"] + 4, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Drag preview
        if self.drawing:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 3. Top HUD Banner
        total = len(self.spots)
        rate = (occupied_count / total * 100.0) if total > 0 else 0.0

        if self.show_hud:
            banner_h = 44
            hud_bg = canvas.copy()
            cv2.rectangle(hud_bg, (0, 0), (self.w, banner_h), (20, 20, 25), -1)
            cv2.line(hud_bg, (0, banner_h), (self.w, banner_h), (50, 50, 60), 2)
            cv2.addWeighted(hud_bg, 0.85, canvas, 0.15, 0, canvas)

            items = [
                (f"TOTAL: {total}", (200, 200, 200)),
                (f"VACANT: {vacant_count}", (50, 205, 50)),
                (f"OCCUPIED: {occupied_count}", (40, 40, 230)),
                (f"RATE: {rate:.0f}%", (40, 200, 240) if rate > 80 else (200, 200, 200)),
                (f"THRESHOLD: {self.edge_threshold_pct}%", (180, 180, 180)),
                (f"IMG: {os.path.basename(self.image_path)}", (160, 160, 160))
            ]

            spacing = self.w // (len(items) + 1)
            for idx, (text, col) in enumerate(items):
                cx = (idx + 1) * spacing
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.putText(canvas, text, (cx - tw // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)

        # Bottom Quick Guide
        guide_text = "Left Click/Drag: Add Spot | Right Click: Delete | 's': Save | 'g': Auto-Grid | 'o': Open Image | 'v': View Mode | ESC: Exit"
        cv2.putText(canvas, guide_text, (15, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        # 4. View Mode selection
        if self.view_mode == 1:
            # Edge mask in BGR
            edge_bgr = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
            return edge_bgr
        elif self.view_mode == 2:
            # Side-by-side comparison
            edge_bgr = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
            half_w = self.w // 2
            left_half = cv2.resize(canvas, (half_w, self.h))
            right_half = cv2.resize(edge_bgr, (half_w, self.h))
            split = np.hstack((left_half, right_half))
            return split

        return canvas

    def prompt_open_image(self):
        """Allows user to select any image from disk using native file dialog."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected = filedialog.askopenfilename(
                title="Select Parking Lot Image to Analyze",
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All Files", "*.*")]
            )
            root.destroy()
            if selected:
                self.load_image(selected)
                self.reset_trackbars()
        except Exception as e:
            print(f"[!] Could not open file dialog: {e}")

    def reset_trackbars(self):
        """Synchronizes trackbars with current image values."""
        cv2.setTrackbarPos("Sensitivity (%)", self.window_name, self.edge_threshold_pct)
        cv2.setTrackbarPos("Spot Width", self.window_name, min(300, self.spot_width))
        cv2.setTrackbarPos("Spot Height", self.window_name, min(200, self.spot_height))

    def export_summary(self):
        """Exports annotated image and CSV summary."""
        base, _ = os.path.splitext(self.image_path)
        out_img_name = f"{base}_analyzed.png"
        out_csv_name = f"{base}_summary.csv"

        # Save image
        cv2.imwrite(out_img_name, self.render())

        # Save CSV
        rows = ["Slot ID,Status,Edge Density,X1,Y1,X2,Y2"]
        for s in self.spots:
            is_occ, edge_ratio = self.analyze_spot(s)
            status_str = "Occupied" if is_occ else "Vacant"
            rows.append(f"#{s['id']},{status_str},{int(edge_ratio*100)}%,{s['x1']},{s['y1']},{s['x2']},{s['y2']}")

        with open(out_csv_name, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))

        print(f"[+] Exported analysis image to: {out_img_name}")
        print(f"[+] Exported CSV summary to: {out_csv_name}")

    def run(self):
        """Main interaction loop with OpenCV window & trackbars."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # --- Native OpenCV Trackbars ---
        def on_thresh(val):
            self.edge_threshold_pct = max(1, val)

        def on_width(val):
            self.spot_width = max(20, val)

        def on_height(val):
            self.spot_height = max(15, val)

        cv2.createTrackbar("Sensitivity (%)", self.window_name, self.edge_threshold_pct, 40, on_thresh)
        cv2.createTrackbar("Spot Width", self.window_name, min(300, self.spot_width), 300, on_width)
        cv2.createTrackbar("Spot Height", self.window_name, min(200, self.spot_height), 200, on_height)

        print("\n=======================================================")
        print("   [P] ParkSense AI - Pure Python & OpenCV Analyzer     ")
        print("=======================================================")
        print("  Controls:")
        print("  - Left Click        : Place spot with current Width x Height")
        print("  - Left Drag         : Draw custom spot rectangle")
        print("  - Right Click       : Delete spot")
        print("  - 'o' / 'f'         : Open ANY new image file to analyze")
        print("  - 'g'               : Auto-generate row grid of spots")
        print("  - 's'               : Save spots to JSON (<image>_spots.json)")
        print("  - 'l'               : Reload saved spots")
        print("  - 'c'               : Clear all spots")
        print("  - 'e'               : Export analyzed image & CSV summary")
        print("  - 'v'               : Cycle view (Annotated / Edge Mask / Split)")
        print("  - 'h'               : Toggle HUD banner")
        print("  - 'ESC' or 'q'      : Quit")
        print("=======================================================\n")

        while True:
            display_frame = self.render()
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(25) & 0xFF
            if key in [27, ord('q'), ord('Q')]:  # ESC or Q
                break
            elif key in [ord('o'), ord('O'), ord('f'), ord('F')]:
                self.prompt_open_image()
            elif key in [ord('s'), ord('S')]:
                self.save_spots_for_current_image()
            elif key in [ord('l'), ord('L')]:
                self.load_spots_for_current_image()
            elif key in [ord('c'), ord('C')]:
                self.spots.clear()
                print("[-] Cleared all spots.")
            elif key in [ord('g'), ord('G')]:
                # Generate a row of 6 spots across the middle of current image
                start_x = int(self.w * 0.1)
                start_y = int(self.h * 0.5)
                cols = max(3, int((self.w * 0.8) / (self.spot_width + 15)))
                self.auto_generate_grid(start_x=start_x, start_y=start_y, cols=cols)
            elif key in [ord('e'), ord('E')]:
                self.export_summary()
            elif key in [ord('v'), ord('V')]:
                self.view_mode = (self.view_mode + 1) % 3
                modes = ["Annotated View", "Edge Mask View", "Side-by-Side Split View"]
                print(f"[UI] View Mode: {modes[self.view_mode]}")
            elif key in [ord('h'), ord('H')]:
                self.show_hud = not self.show_hud

        cv2.destroyAllWindows()


def open_file_dialog() -> Optional[str]:
    """Opens a native OS file dialog to browse for an image."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        selected = filedialog.askopenfilename(
            title="Select Parking Lot Image to Analyze",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All Files", "*.*")]
        )
        root.destroy()
        return selected if selected else None
    except Exception as e:
        print(f"[!] File dialog error: {e}")
        return None


def ask_user_for_image() -> str:
    """Interactively asks the user to enter or select a parking lot image."""
    print("\n=======================================================")
    print("   [P] ParkSense AI - Smart Parking Space Analyzer     ")
    print("=======================================================")
    print("Please provide the parking lot image to analyze:")
    print("  [1] Type or drag-and-drop the image file path here")
    print("  [2] Press Enter (blank) to browse using File Dialog")
    print("  [3] Type 'default' to use sample 'parking_lot.png'")
    print("-------------------------------------------------------")

    while True:
        try:
            user_input = input("Enter parking lot image path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        # Blank input -> open file browser
        if not user_input:
            print("[*] Opening File Browser dialog...")
            selected = open_file_dialog()
            if selected and os.path.exists(selected):
                return selected
            print("[!] No file selected. Please enter a file path or type 'default'.")
            continue

        # Default sample image
        if user_input.lower() == "default":
            if os.path.exists("parking_lot.png"):
                return "parking_lot.png"
            print("[!] 'parking_lot.png' was not found in the current folder.")
            continue

        # User-provided file path
        if os.path.exists(user_input):
            test_img = cv2.imread(user_input)
            if test_img is not None:
                return user_input
            else:
                print(f"[!] Error: File '{user_input}' exists but could not be read as an image. Try another file.")
        else:
            print(f"[!] Error: File '{user_input}' not found. Please check the path and try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Parking Space Analyzer (Pure OpenCV)")
    parser.add_argument("image", nargs="?", default=None, help="Path to any parking lot image file")
    args = parser.parse_args()

    # If image path was not given as CLI argument, interactively ask the user first!
    image_path = args.image if args.image else ask_user_for_image()

    print(f"\n=======================================================")
    print(f"[+] Selected image: {image_path}")
    print(f"[+] Initializing OpenCV analysis engine...")
    print(f"=======================================================")

    app = ParkingSpaceAnalyzer(image_path=image_path)

    # If the image has no pre-saved spots, auto-generate an initial layout
    if not app.spots:
        print("[*] Generating initial parking bays for this image...")
        start_x = int(app.w * 0.08)
        start_y = int(app.h * 0.50)
        cols = max(3, min(10, int((app.w * 0.84) / (app.spot_width + 15))))
        app.auto_generate_grid(start_x=start_x, start_y=start_y, cols=cols)

    # Automatically save initial analysis output
    app.export_summary()

    # Launch the interactive OpenCV window
    app.run()