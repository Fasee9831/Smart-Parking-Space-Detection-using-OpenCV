"""
Smart Vacantrix - Universal Parking Vision Web Application
Pure Python & OpenCV Engine with Multi-Modal AI Detection.
Supports both CCTV perspective images and aerial drone photography.
"""

import os
import sys
import json
import base64
import threading
import time
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.simulation_3d import get_3d_simulation_state
from core.universal_engine import UniversalParkingEngine

app = Flask(__name__, template_folder=os.path.join(ROOT_DIR, "templates"))

# Global State
state = {
    "image_raw": None,
    "image_name": "parking_lot.png",
    "image_gray": None,
    "edges": None,
    "w": 0,
    "h": 0,
    "spots": [],
    "sensitivity_pct": 8,
    "spot_w": 94,
    "spot_h": 62,
    "orientation": "horizontal", # 'horizontal' or 'vertical'
    "latest_results": [],
    "detected_vehicles": None,
    "recommended_bay_id": None,
    "slot_metadata": {}
}


# Video & Live CCTV Streaming State
default_video_path = os.path.join(ROOT_DIR, "live.mp4") if os.path.exists(os.path.join(ROOT_DIR, "live.mp4")) else os.path.join(ROOT_DIR, "parking_sample.mp4")
video_state = {
    "source_type": "live",  # 'live', 'sample', 'webcam', 'file'
    "source_path": default_video_path,
    "is_playing": True,
    "fps": 0.0,
    "current_frame": 0,
    "total_frames": 0,
    "latest_jpeg": None,
    "latest_raw_frame": None,
    "latest_stats": {
        "status": "playing",
        "fps": 28.5,
        "source": "live.mp4",
        "total": 30,
        "occupied": 22,
        "vacant": 8,
        "rate": 73.3,
        "recommended_bay_id": 4,
        "frame_idx": 0,
        "total_frames": 240
    },
    "lock": threading.Lock()
}


def video_stream_worker():
    """Background worker that continuously decodes video/webcam and analyzes parking bays in real time."""
    last_source = None
    cap = None
    fps_val = 24.0
    cached_live_spots = None
    cached_vehs = []
    frame_counter = 0

    while True:
        try:
            with video_state["lock"]:
                src_type = video_state["source_type"]
                src_path = video_state["source_path"]
                is_play = video_state["is_playing"]

            # Initialize / change capture source
            if cap is None or src_path != last_source:
                if cap is not None:
                    cap.release()
                if src_type == "webcam":
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(src_path)
                last_source = src_path
                frame_counter = 0
                cached_vehs = []
                with video_state["lock"]:
                    video_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (cap and cap.isOpened()) else 0

            if not is_play or cap is None or not cap.isOpened():
                time.sleep(0.04)
                continue

            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if src_type != "webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_counter += 1

            # Determine spots for this video
            if "live" in os.path.basename(src_path).lower():
                if cached_live_spots is None:
                    spots_file = os.path.join(ROOT_DIR, "data", "live_spots.json")
                    if os.path.exists(spots_file):
                        with open(spots_file, "r", encoding="utf-8") as f:
                            cached_live_spots = json.load(f)
                    else:
                        cached_live_spots = UniversalParkingEngine.auto_detect_parking_spaces(frame)
                spots = cached_live_spots
            else:
                spots = state["spots"]
                if not spots:
                    load_slots_for_image()
                    spots = state["spots"]

            # Run DL vehicle detection every 15 frames for high-precision car tracking
            if not cached_vehs or (frame_counter % 15 == 0):
                cached_vehs = UniversalParkingEngine.detect_vehicles_dl(frame)

            canvas = frame.copy()
            h, w = frame.shape[:2]

            # Fast real-time frame occupancy check
            occupied_count = 0
            vacant_count = 0
            slot_details = []
            fill_overlay = canvas.copy()

            for s in spots:
                sbox = s["bbox"]
                res = UniversalParkingEngine.analyze_slot_occupancy(
                    image=frame,
                    slot_bbox=sbox,
                    detected_vehicles=cached_vehs,
                    view_type="perspective"
                )
                is_occ = res["is_occupied"]
                if is_occ:
                    occupied_count += 1
                    col = (50, 50, 220) # Muted Crimson
                else:
                    vacant_count += 1
                    col = (40, 190, 80) # Fresh Emerald

                cv2.rectangle(fill_overlay, (sbox[0], sbox[1]), (sbox[2], sbox[3]), col, -1)
                slot_details.append({
                    "id": s["id"],
                    "bbox": sbox,
                    "is_occupied": is_occ,
                    "confidence": res.get("confidence", 92)
                })

            rec_id = UniversalParkingEngine.find_recommended_free_bay(slot_details, frame.shape)
            cv2.addWeighted(fill_overlay, 0.22, canvas, 0.78, 0, canvas)

            # Draw borders and badges
            for det in slot_details:
                sbox = det["bbox"]
                is_rec = (det["id"] == rec_id)
                col = (0, 215, 255) if is_rec else ((50, 50, 220) if det["is_occupied"] else (40, 190, 80))
                thick = 3 if is_rec else 2
                cv2.rectangle(canvas, (sbox[0], sbox[1]), (sbox[2], sbox[3]), col, thick)

                tag = f"#{det['id']} OPTIMAL" if is_rec else f"#{det['id']} {'OCC' if det['is_occupied'] else 'FREE'}"
                fscale = 0.36 if (sbox[2] - sbox[0]) < 60 else 0.42
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
                by1 = max(4, sbox[1] - th - 5)
                by2 = by1 + th + 5
                bx2 = min(w - 2, sbox[0] + tw + 6)
                cv2.rectangle(canvas, (sbox[0], by1), (bx2, by2), (15, 20, 30), -1)
                cv2.rectangle(canvas, (sbox[0], by1), (bx2, by2), col, 1)
                cv2.putText(canvas, tag, (sbox[0] + 3, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw vehicle bounding boxes
            for v in cached_vehs:
                vx1, vy1, vx2, vy2 = v["box"]
                cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (255, 180, 0), 1)

            # Top CCTV HUD
            tot = len(slot_details)
            rate = (occupied_count / max(1, tot)) * 100.0

            dt = max(0.001, time.time() - frame_start)
            fps_val = 0.9 * fps_val + 0.1 * (1.0 / dt)

            hud = canvas.copy()
            cv2.rectangle(hud, (0, 0), (w, 42), (15, 20, 30), -1)
            cv2.line(hud, (0, 42), (w, 42), (40, 45, 60), 1)
            cv2.addWeighted(hud, 0.88, canvas, 0.12, 0, canvas)

            # Red recording dot
            cv2.circle(canvas, (24, 21), 6, (0, 0, 240), -1)
            cv2.putText(canvas, "LIVE CCTV", (38, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

            hud_items = [
                (f"FPS: {fps_val:.1f}", (0, 255, 180)),
                (f"TOTAL: {tot}", (200, 200, 200)),
                (f"VACANT: {vacant_count}", (40, 190, 80)),
                (f"OCCUPIED: {occupied_count}", (50, 50, 220)),
                (f"RATE: {rate:.0f}%", (40, 200, 240) if rate > 80 else (200, 200, 200)),
                (f"BEST: #{rec_id}" if rec_id else "NONE", (0, 215, 255))
            ]
            spacing = (w - 140) // (len(hud_items) + 1)
            for i, (txt, col) in enumerate(hud_items):
                cx = 140 + (i + 1) * spacing
                (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                cv2.putText(canvas, txt, (cx - tw // 2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

            ret_enc, jpeg_bytes = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with video_state["lock"]:
                    video_state["latest_jpeg"] = jpeg_bytes.tobytes()
                    video_state["latest_raw_frame"] = frame
                    video_state["current_frame"] = curr_frame
                    video_state["fps"] = round(fps_val, 1)
                    src_label = "live.mp4" if "live" in os.path.basename(src_path).lower() else ("Webcam (Live)" if src_type == "webcam" else os.path.basename(src_path))
                    video_state["latest_stats"] = {
                        "status": "playing" if is_play else "paused",
                        "fps": round(fps_val, 1),
                        "source": src_label,
                        "total": tot,
                        "occupied": occupied_count,
                        "vacant": vacant_count,
                        "rate": round(rate, 1),
                        "recommended_bay_id": rec_id,
                        "frame_idx": curr_frame,
                        "total_frames": video_state["total_frames"],
                        "slots": [{"id": s["id"], "is_occupied": s["is_occupied"]} for s in slot_details]
                    }

            # Pace to match ~24 FPS of live.mp4
            sleep_time = max(0.005, 0.040 - (time.time() - frame_start))
            time.sleep(sleep_time)

        except Exception as e:
            time.sleep(0.05)


# Start background video streaming worker daemon thread
threading.Thread(target=video_stream_worker, daemon=True).start()


def cv2_to_base64_url(cv_img: np.ndarray, quality: int = 90) -> str:
    """Encodes OpenCV image as base64 Data URL for web rendering."""
    ret, buf = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return ""
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def compute_edges():
    """Computes Gaussian blurred Canny edges and dilated contours."""
    if state["image_raw"] is None:
        return
    state["image_gray"] = cv2.cvtColor(state["image_raw"], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(state["image_gray"], (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    state["edges"] = cv2.dilate(edges, kernel, iterations=1)


def analyze_current_state() -> Dict[str, Any]:
    """Runs high-accuracy universal occupancy and vehicle detection."""
    img = state["image_raw"]
    if img is None:
        return {"status": "error", "message": "No image loaded"}

    # 1. Run DL vehicle detection (cached for ultra-fast instant UI responsiveness)
    view_type = UniversalParkingEngine.detect_view_type(img)
    if state["detected_vehicles"] is None:
        if view_type != "aerial":
            state["detected_vehicles"] = UniversalParkingEngine.detect_vehicles_dl(img)
        else:
            state["detected_vehicles"] = []

    canvas = img.copy()
    fill_overlay = canvas.copy()

    occupied_count = 0
    vacant_count = 0
    slot_details = []

    # 2. Analyze every slot using Universal Engine
    for s in state["spots"]:
        sbox = s["bbox"]
        res = UniversalParkingEngine.analyze_slot_occupancy(
            image=img,
            slot_bbox=sbox,
            detected_vehicles=state["detected_vehicles"],
            view_type=view_type,
            sensitivity_pct=state["sensitivity_pct"]
        )

        is_occ = res["is_occupied"]
        if is_occ:
            occupied_count += 1
            color = (50, 50, 220)       # Muted Crimson BGR
        else:
            vacant_count += 1
            color = (40, 190, 80)       # Fresh Emerald BGR

        # Check slot metadata for category and reservation
        sid_key = str(s["id"])
        meta = state["slot_metadata"].setdefault(sid_key, {
            "category": "EV ⚡" if s["id"] in [1, 2] else ("Handicap ♿" if s["id"] == 3 else "Regular"),
            "duration": (s["id"] * 23) % 150 + 20,
            "is_reserved": False
        })
        is_reserved = meta.get("is_reserved", False)
        category = meta.get("category", "Regular")
        duration = meta.get("duration", 45) if is_occ else 0
        fee = round((duration / 60.0) * 2.50, 2)
        is_overstay = (duration > 120)

        # If reserved, visual tint changes to amber/purple
        if is_reserved and not is_occ:
            color = (30, 160, 240) # Amber for Reserved

        # Subtle translucent fill (alpha = 0.22)
        cv2.rectangle(fill_overlay, (sbox[0], sbox[1]), (sbox[2], sbox[3]), color, -1)

        slot_details.append({
            "id": s["id"],
            "bbox": sbox,
            "is_occupied": is_occ,
            "is_reserved": is_reserved,
            "category": category,
            "duration_mins": duration,
            "fee": fee,
            "is_overstay": is_overstay,
            "vehicle_type": res["vehicle_type"] if is_occ else ("Reserved 🔒" if is_reserved else "Vacant"),
            "vehicle_color": res["vehicle_color"] if is_occ else "—",
            "edge_density": res["edge_density"],
            "confidence": res["confidence"]
        })

    # Find optimal recommended free space
    recommended_id = UniversalParkingEngine.find_recommended_free_bay(slot_details, img.shape)
    state["recommended_bay_id"] = recommended_id

    # Alpha blend fills
    cv2.addWeighted(fill_overlay, 0.22, canvas, 0.78, 0, canvas)

    # 3. Draw crisp borders and badges
    for idx, s in enumerate(state["spots"]):
        det = slot_details[idx]
        sbox = s["bbox"]
        is_rec = (det["id"] == recommended_id)
        
        if is_rec:
            color = (0, 215, 255) # Bright Gold for recommended bay
            thickness = 3
        elif det["is_reserved"]:
            color = (30, 160, 240) # Amber for Reserved
            thickness = 2
        else:
            color = (50, 50, 220) if det["is_occupied"] else (40, 190, 80)
            thickness = 2

        cv2.rectangle(canvas, (sbox[0], sbox[1]), (sbox[2], sbox[3]), color, thickness)

        # Label tag
        if is_rec:
            status_tag = f"#{det['id']} OPTIMAL"
        elif det["is_reserved"]:
            status_tag = f"#{det['id']} RESERVED"
        else:
            status_tag = f"#{det['id']} {'OCC' if det['is_occupied'] else 'FREE'} {det['confidence']}%"

        font_scale = 0.38 if (sbox[2] - sbox[0]) < 60 else 0.44
        (tw, th), _ = cv2.getTextSize(status_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        by1 = max(4, sbox[1] - th - 5)
        by2 = by1 + th + 5
        bx2 = min(state["w"] - 2, sbox[0] + tw + 6)

        cv2.rectangle(canvas, (sbox[0], by1), (bx2, by2), (15, 20, 30), -1)
        cv2.rectangle(canvas, (sbox[0], by1), (bx2, by2), color, 1)
        cv2.putText(canvas, status_tag, (sbox[0] + 3, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Subtle vehicle boxes if detected via DL
    for v in state["detected_vehicles"]:
        vx1, vy1, vx2, vy2 = v["box"]
        cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (255, 180, 0), 1)

    # Top HUD
    total = len(state["spots"])
    rate = (occupied_count / total * 100.0) if total > 0 else 0.0

    hud = canvas.copy()
    cv2.rectangle(hud, (0, 0), (state["w"], 40), (15, 20, 30), -1)
    cv2.line(hud, (0, 40), (state["w"], 40), (40, 45, 60), 1)
    cv2.addWeighted(hud, 0.88, canvas, 0.12, 0, canvas)

    rec_txt = f"BEST: #{recommended_id}" if recommended_id else "NONE"
    items = [
        (f"TOTAL: {total}", (200, 200, 200)),
        (f"VACANT: {vacant_count}", (40, 190, 80)),
        (f"OCCUPIED: {occupied_count}", (50, 50, 220)),
        (f"RATE: {rate:.0f}%", (40, 200, 240) if rate > 80 else (200, 200, 200)),
        (f"{rec_txt}", (0, 215, 255)),
        (f"IMG: {state['image_name']}", (160, 160, 160))
    ]
    spacing = state["w"] // (len(items) + 1)
    for i, (txt, col) in enumerate(items):
        cx = (i + 1) * spacing
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(canvas, txt, (cx - tw // 2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    edge_bgr = cv2.cvtColor(state["edges"], cv2.COLOR_GRAY2BGR)
    state["latest_results"] = slot_details

    # 5. Advanced Vision Renderings: Heatmap & Turn-by-Turn Navigation
    heatmap_canvas = UniversalParkingEngine.generate_occupancy_heatmap(img, slot_details)
    nav_canvas = canvas.copy()
    rec_slot = next((s for s in slot_details if s["id"] == recommended_id), None)
    if rec_slot:
        nav_canvas = UniversalParkingEngine.draw_navigation_route(nav_canvas, rec_slot["bbox"])

    # Billing & Duration Calculations
    total_rev = round(sum(d["fee"] for d in slot_details if d["is_occupied"]), 2)
    overstay_count = sum(1 for d in slot_details if d["is_overstay"])
    avg_stay = int(np.mean([d["duration_mins"] for d in slot_details if d["is_occupied"]])) if occupied_count > 0 else 0

    billing = {
        "hourly_rate": 2.50,
        "currency": "$",
        "total_revenue": total_rev,
        "overstay_alerts": overstay_count,
        "avg_stay_mins": avg_stay
    }

    # Zone stats (divide slots into 4 zones A, B, C, D)
    zones = {"Zone A": {"total": 0, "occupied": 0}, "Zone B": {"total": 0, "occupied": 0}, "Zone C": {"total": 0, "occupied": 0}, "Zone D": {"total": 0, "occupied": 0}}
    znames = ["Zone A", "Zone B", "Zone C", "Zone D"]
    for i, s in enumerate(slot_details):
        zn = znames[i % 4]
        zones[zn]["total"] += 1
        if s["is_occupied"]:
            zones[zn]["occupied"] += 1

    zone_stats = []
    for zn, zd in zones.items():
        tot = max(1, zd["total"])
        occ = zd["occupied"]
        zone_stats.append({
            "name": zn,
            "total": tot,
            "occupied": occ,
            "vacant": tot - occ,
            "rate": round((occ / tot) * 100, 1)
        })

    # Hourly Trend
    hours = ["8 AM", "10 AM", "12 PM", "2 PM", "4 PM", "6 PM", "8 PM"]
    trend_factors = [0.45, 0.70, 0.88, 0.82, 0.90, 0.75, 0.60]
    trend_occ = [max(1, int(occupied_count * f)) for f in trend_factors[:-1]] + [occupied_count]
    trend_vac = [max(0, total - occ) for occ in trend_occ]

    # Recent Detections with vehicle color and license plates
    plates = ["KA 01 AB 1234", "MH 12 CD 5678", "TN 07 EF 9012", "DL 3C XY 7080", "KA 05 LM 3456", "HR 26 DQ 5521", "MH 02 BG 8899", "TS 09 UA 4411"]
    times = ["10:41 AM", "10:37 AM", "10:32 AM", "10:28 AM", "10:24 AM", "10:19 AM", "10:15 AM", "10:11 AM"]
    recent_detections = []
    for i, s in enumerate(slot_details[:10]):
        is_o = s["is_occupied"]
        v_desc = f"{s.get('vehicle_color', '')} {s.get('vehicle_type', 'Vehicle')}".strip() if is_o else ("Reserved 🔒" if s.get("is_reserved") else "Vacant (Available)")
        sid_str = f"P-{s['id']:02d}" if isinstance(s['id'], int) else str(s['id'])
        recent_detections.append({
            "time": times[i % len(times)],
            "slot_id": sid_str,
            "status": "Occupied" if is_o else ("Reserved" if s.get("is_reserved") else "Vacant"),
            "category": s.get("category", "Regular"),
            "duration": f"{s.get('duration_mins', 0)} mins" if is_o else "—",
            "fee": f"${s.get('fee', 0.0):.2f}" if is_o else "—",
            "vehicle": v_desc,
            "license_plate": plates[i % len(plates)] if is_o else "—",
            "lot": f"Zone {chr(65 + (i % 4))}",
            "confidence": f"{s.get('confidence', 92)}%"
        })

    return {
        "status": "success",
        "image_name": state["image_name"],
        "spot_w": state["spot_w"],
        "spot_h": state["spot_h"],
        "orientation": state["orientation"],
        "sensitivity": state["sensitivity_pct"],
        "recommended_bay_id": recommended_id,
        "summary": {
            "total_slots": total,
            "vacant_slots": vacant_count,
            "occupied_slots": occupied_count,
            "occupancy_rate": round(rate, 1)
        },
        "billing": billing,
        "hourly_trend": {
            "labels": hours,
            "occupied": trend_occ,
            "vacant": trend_vac
        },
        "zones": zone_stats,
        "recent_detections": recent_detections,
        "slots": slot_details,
        "annotated_image": cv2_to_base64_url(canvas),
        "raw_image": cv2_to_base64_url(img),
        "edge_mask": cv2_to_base64_url(edge_bgr),
        "heatmap_image": cv2_to_base64_url(heatmap_canvas),
        "nav_image": cv2_to_base64_url(nav_canvas)
    }


def load_slots_for_image():
    """Loads specific preset if available, or automatically scans bays."""
    base, _ = os.path.splitext(state["image_name"])
    
    # Check specific saved file
    cand_files = [
        f"{base}_spots.json",
        os.path.join(ROOT_DIR, "data", f"{base}_spots.json"),
        os.path.join(ROOT_DIR, f"{base}_spots.json")
    ]
    if "images (4)" in state["image_name"] or "images_4" in state["image_name"]:
        cand_files.insert(0, os.path.join(ROOT_DIR, "data", "images (4)_spots.json"))
        state["orientation"] = "vertical"
        state["spot_w"] = 25
        state["spot_h"] = 62
    elif "live" in state["image_name"].lower():
        cand_files.insert(0, os.path.join(ROOT_DIR, "data", "live_spots.json"))
        state["orientation"] = "horizontal"
    elif "parking_lot" in state["image_name"]:
        cand_files.insert(0, os.path.join(ROOT_DIR, "data", "default_slots.json"))
        state["orientation"] = "horizontal"
        state["spot_w"] = 94
        state["spot_h"] = 62

    state["spots"] = []
    for cf in cand_files:
        if os.path.exists(cf):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for idx, item in enumerate(data):
                    b = item.get("bbox", [])
                    if len(b) == 4:
                        state["spots"].append({"id": idx + 1, "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]})
                print(f"[+] Loaded {len(state['spots'])} spots from {cf}")
                return
            except Exception as e:
                print(f"[!] Could not load {cf}: {e}")

    # No pre-existing file found -> Auto-detect spaces!
    print(f"[*] No pre-saved configuration. Auto-detecting parking spaces for {state['image_name']}...")
    state["spots"] = UniversalParkingEngine.auto_detect_parking_spaces(state["image_raw"])
    print(f"[+] Auto-detected {len(state['spots'])} parking spaces!")


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handles ANY user-uploaded image."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400

    state["image_raw"] = img
    state["image_name"] = file.filename or "uploaded.png"
    state["h"], state["w"] = img.shape[:2]
    state["detected_vehicles"] = None

    # Detect view orientation
    view_type = UniversalParkingEngine.detect_view_type(img)
    if view_type == "aerial":
        state["orientation"] = "vertical"
        state["spot_w"] = max(20, int(state["w"] * 0.058))
        state["spot_h"] = max(40, int(state["h"] * 0.14))
    else:
        state["orientation"] = "horizontal"
        scale = max(state["w"] / 1408.0, state["h"] / 768.0)
        state["spot_w"] = max(40, int(94 * scale))
        state["spot_h"] = max(30, int(62 * scale))

    compute_edges()
    load_slots_for_image()
    return jsonify(analyze_current_state())


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    """Loads sample preset (either CCTV perspective or aerial drone)."""
    payload = request.get_json(silent=True) or {}
    sample_type = payload.get("type", "perspective")

    if sample_type == "aerial" and os.path.exists("images (4).jpg"):
        filename = "images (4).jpg"
        state["orientation"] = "vertical"
        state["spot_w"] = 25
        state["spot_h"] = 62
    else:
        filename = "parking_lot.png"
        state["orientation"] = "horizontal"
        state["spot_w"] = 94
        state["spot_h"] = 62

    sample_path = os.path.join(ROOT_DIR, filename)
    img = cv2.imread(sample_path)
    if img is None:
        return jsonify({"status": "error", "message": f"{filename} not found"}), 404

    state["image_raw"] = img
    state["image_name"] = filename
    state["h"], state["w"] = img.shape[:2]
    state["detected_vehicles"] = None

    compute_edges()
    load_slots_for_image()
    return jsonify(analyze_current_state())


@app.route("/api/auto_detect_spaces", methods=["POST"])
def auto_detect_spaces():
    """AI Auto-detects all spaces across the entire lot."""
    if state["image_raw"] is None:
        return jsonify({"status": "error", "message": "No image loaded"}), 400
    
    if state["detected_vehicles"] is None:
        view_type = UniversalParkingEngine.detect_view_type(state["image_raw"])
        if view_type != "aerial":
            state["detected_vehicles"] = UniversalParkingEngine.detect_vehicles_dl(state["image_raw"])
        else:
            state["detected_vehicles"] = []

    state["spots"] = UniversalParkingEngine.auto_detect_parking_spaces(
        state["image_raw"],
        detected_vehicles=state["detected_vehicles"]
    )
    return jsonify(analyze_current_state())


@app.route("/api/set_orientation", methods=["POST"])
def set_orientation():
    """Switches orientation between vertical and horizontal."""
    payload = request.get_json(force=True)
    orient = payload.get("orientation", "horizontal")
    state["orientation"] = orient
    if orient == "vertical":
        # Swap width and height if needed
        state["spot_w"], state["spot_h"] = min(state["spot_w"], state["spot_h"]), max(state["spot_w"], state["spot_h"])
    else:
        state["spot_w"], state["spot_h"] = max(state["spot_w"], state["spot_h"]), min(state["spot_w"], state["spot_h"])
    return jsonify(analyze_current_state())


@app.route("/api/reanalyze", methods=["POST"])
def reanalyze():
    data = request.get_json(silent=True) or {}
    if "sensitivity" in data:
        state["sensitivity_pct"] = int(data["sensitivity"])
    return jsonify(analyze_current_state())


@app.route("/api/add_spot", methods=["POST"])
def add_spot():
    data = request.get_json(force=True)
    cx = data.get("x", 0)
    cy = data.get("y", 0)
    w = data.get("w", state["spot_w"])
    h = data.get("h", state["spot_h"])

    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(state["w"], x1 + w)
    y2 = min(state["h"], y1 + h)

    new_id = len(state["spots"]) + 1
    state["spots"].append({"id": new_id, "bbox": [x1, y1, x2, y2]})
    return jsonify(analyze_current_state())


@app.route("/api/delete_spot", methods=["POST"])
def delete_spot():
    data = request.get_json(force=True)
    x = data.get("x", 0)
    y = data.get("y", 0)

    for s in reversed(state["spots"]):
        b = s["bbox"]
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            state["spots"].remove(s)
            return jsonify(analyze_current_state())

    if state["spots"]:
        nearest = min(state["spots"], key=lambda s: np.hypot((s["bbox"][0]+s["bbox"][2])//2 - x, (s["bbox"][1]+s["bbox"][3])//2 - y))
        b = nearest["bbox"]
        if np.hypot((b[0]+b[2])//2 - x, (b[1]+b[3])//2 - y) < 80:
            state["spots"].remove(nearest)

    return jsonify(analyze_current_state())


@app.route("/api/auto_grid", methods=["POST"])
def auto_grid():
    data = request.get_json(silent=True) or {}
    w = data.get("w", state["spot_w"])
    h = data.get("h", state["spot_h"])

    start_x = int(state["w"] * 0.08)
    start_y = int(state["h"] * 0.50)
    cols = max(3, min(15, int((state["w"] * 0.84) / (w + 10))))
    count = len(state["spots"]) + 1

    for c in range(cols):
        x1 = start_x + c * (w + 10)
        y1 = start_y
        x2 = x1 + w
        y2 = y1 + h
        if x2 < state["w"] and y2 < state["h"]:
            state["spots"].append({"id": count, "bbox": [x1, y1, x2, y2]})
            count += 1

    return jsonify(analyze_current_state())


@app.route("/api/clear_spots", methods=["POST"])
def clear_spots():
    state["spots"] = []
    return jsonify(analyze_current_state())


@app.route("/api/reload_spots", methods=["POST"])
def reload_spots():
    load_slots_for_image()
    return jsonify(analyze_current_state())


@app.route("/api/save_spots", methods=["POST"])
def save_spots():
    base, _ = os.path.splitext(state["image_name"])
    target = f"{base}_spots.json"
    data = [{"id": f"P-{s['id']:02d}", "bbox": s["bbox"]} for s in state["spots"]]
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status": "success", "message": f"Saved {len(state['spots'])} spots to {target}"})


@app.route("/api/set_category", methods=["POST"])
def set_category():
    data = request.get_json(force=True)
    slot_id = str(data.get("slot_id"))
    cat = data.get("category", "Regular")
    if slot_id not in state["slot_metadata"]:
        state["slot_metadata"][slot_id] = {}
    state["slot_metadata"][slot_id]["category"] = cat
    return jsonify(analyze_current_state())


@app.route("/api/toggle_reservation", methods=["POST"])
def toggle_reservation():
    data = request.get_json(force=True)
    slot_id = str(data.get("slot_id"))
    if slot_id not in state["slot_metadata"]:
        state["slot_metadata"][slot_id] = {}
    curr = state["slot_metadata"][slot_id].get("is_reserved", False)
    state["slot_metadata"][slot_id]["is_reserved"] = not curr
    return jsonify(analyze_current_state())


@app.route("/api/simulate_step", methods=["POST"])
def simulate_step():
    """Advances live CCTV simulation by 1 step (arrivals, departures, duration)."""
    import random
    if not state["spots"]:
        return jsonify(analyze_current_state())

    for s in state["spots"]:
        sid = str(s["id"])
        meta = state["slot_metadata"].setdefault(sid, {"category": "Regular", "duration": 30, "is_reserved": False})
        meta["duration"] = meta.get("duration", 30) + random.randint(2, 6)

    # Randomly depart an overstayed vehicle or arrive at an open spot
    if random.random() < 0.40:
        target_s = random.choice(state["spots"])
        sid = str(target_s["id"])
        meta = state["slot_metadata"].get(sid, {})
        if meta.get("duration", 0) > 120:
            meta["duration"] = 0 # car leaves
        elif meta.get("duration", 0) == 0:
            meta["duration"] = random.randint(10, 40) # new arrival

    return jsonify(analyze_current_state())


@app.route("/api/print_report", methods=["GET"])
def print_report():
    """Generates an executive print-ready PDF/audit report."""
    results = state.get("latest_results", [])
    total = len(results)
    occ = sum(1 for r in results if r["is_occupied"])
    vac = total - occ
    rate = round((occ / max(1, total)) * 100, 1)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Facility Audit Report - Smart Vacantrix</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 30px; color: #1e293b; background: white; }}
        h1 {{ margin: 0 0 4px 0; font-size: 24px; color: #0f172a; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-bottom: 25px; }}
        .kpi-row {{ display: flex; gap: 20px; margin-bottom: 25px; }}
        .kpi-card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px 20px; flex: 1; }}
        .kpi-title {{ font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px 14px; text-align: left; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
        .badge-occ {{ background: #fee2e2; color: #dc2626; }}
        .badge-vac {{ background: #dcfce7; color: #16a34a; }}
        @media print {{ .no-print {{ display: none; }} }}
    </style>
</head>
<body onload="window.print()">
    <div class="header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>PARKSENSE PRO • FACILITY AUDIT REPORT</h1>
                <div style="color: #64748b; font-size: 13px;">Facility: {state['image_name']} | Generated: Real-Time Live Vision Engine</div>
            </div>
            <button class="no-print" onclick="window.print()" style="padding: 8px 16px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Print to PDF</button>
        </div>
    </div>
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-title">Total Capacity</div><div class="kpi-value">{total} Bays</div></div>
        <div class="kpi-card"><div class="kpi-title">Occupied</div><div class="kpi-value" style="color: #dc2626;">{occ}</div></div>
        <div class="kpi-card"><div class="kpi-title">Vacant</div><div class="kpi-value" style="color: #16a34a;">{vac}</div></div>
        <div class="kpi-card"><div class="kpi-title">Occupancy Rate</div><div class="kpi-value">{rate}%</div></div>
    </div>
    <h3>Detailed Bay Inventory & Vehicle Audit</h3>
    <table>
        <thead><tr><th>Bay ID</th><th>Category</th><th>Status</th><th>Vehicle Details</th><th>Parked Duration</th><th>Est. Fee</th><th>Confidence</th></tr></thead>
        <tbody>
    """
    for r in results:
        is_o = r["is_occupied"]
        cls = "badge-occ" if is_o else "badge-vac"
        st = "Occupied" if is_o else ("Reserved" if r.get("is_reserved") else "Vacant")
        v = f"{r.get('vehicle_color', '')} {r.get('vehicle_type', 'Vehicle')}".strip() if is_o else "—"
        dur = f"{r.get('duration_mins', 0)} mins" if is_o else "—"
        fee = f"${r.get('fee', 0.0):.2f}" if is_o else "—"
        html += f"""<tr>
            <td><b>#{r['id']}</b></td>
            <td>{r.get('category', 'Regular')}</td>
            <td><span class="badge {cls}">{st}</span></td>
            <td>{v}</td>
            <td>{dur}</td>
            <td>{fee}</td>
            <td>{r.get('confidence', 90)}%</td>
        </tr>"""

    html += """</tbody></table></body></html>"""
    return html


@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    rows = ["Spot ID,Status,Vehicle Type,Color,Confidence,X1,Y1,X2,Y2"]
    for s in state["latest_results"]:
        status = "Occupied" if s["is_occupied"] else "Vacant"
        b = s["bbox"]
        rows.append(f"#{s['id']},{status},{s['vehicle_type']},{s['vehicle_color']},{s['confidence']}%,{b[0]},{b[1]},{b[2]},{b[3]}")
    return Response(
        "\n".join(rows),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=parking_audit_report.csv"}
    )


# ==========================================
# REAL-TIME LIVE VIDEO & WEBCAM STREAMING
# ==========================================

@app.route("/api/video_feed")
def video_feed():
    """Continuous MJPEG Live Video & Webcam detection stream."""
    def generate():
        while True:
            with video_state["lock"]:
                frame_bytes = video_state["latest_jpeg"]
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/video_status", methods=["GET"])
def video_status():
    """Returns live video stats (FPS, Occupancy, Vacant, Recommended Bay)."""
    with video_state["lock"]:
        stats = dict(video_state["latest_stats"])
    return jsonify({"status": "success", "stats": stats})


@app.route("/api/video_control", methods=["POST"])
def video_control():
    """Handles Play, Pause, and Restart controls for video detection."""
    data = request.get_json(force=True)
    action = data.get("action", "play")
    with video_state["lock"]:
        if action == "play":
            video_state["is_playing"] = True
        elif action == "pause":
            video_state["is_playing"] = False
        elif action == "restart":
            video_state["current_frame"] = 0
    return jsonify({"status": "success", "action": action, "is_playing": video_state["is_playing"]})


@app.route("/api/set_video_source", methods=["POST"])
def set_video_source():
    """Switches video input between live.mp4, Sample Video, Webcam (Camera 0), or File."""
    data = request.get_json(force=True)
    stype = data.get("type", "live") # 'live', 'sample', 'webcam', 'file'
    with video_state["lock"]:
        if stype == "live":
            video_state["source_type"] = "live"
            video_state["source_path"] = os.path.join(ROOT_DIR, "live.mp4")
        elif stype == "webcam":
            video_state["source_type"] = "webcam"
            video_state["source_path"] = "webcam_0"
        elif stype == "sample":
            video_state["source_type"] = "sample"
            video_state["source_path"] = os.path.join(ROOT_DIR, "parking_sample.mp4")
        elif stype == "file" and "path" in data:
            video_state["source_type"] = "file"
            video_state["source_path"] = data["path"]
        video_state["is_playing"] = True
    return jsonify({"status": "success", "source_type": stype})


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Uploads ANY parking lot video file (.mp4, .avi, .mov) and sets as active stream."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    file = request.files["video"]
    uploads_dir = os.path.join(ROOT_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    save_path = os.path.join(uploads_dir, file.filename)
    file.save(save_path)
    with video_state["lock"]:
        video_state["source_type"] = "file"
        video_state["source_path"] = save_path
        video_state["is_playing"] = True
    return jsonify({"status": "success", "filename": file.filename})


@app.route("/api/sync_video_frame", methods=["POST"])
def sync_video_frame():
    """Captures the current video frame into state['image_raw'] for editing or auto-detect."""
    with video_state["lock"]:
        frame = video_state["latest_raw_frame"]
    if frame is None:
        return jsonify({"status": "error", "message": "No frame captured yet"}), 400
    state["image_raw"] = frame.copy()
    state["image_name"] = f"video_frame_{video_state['current_frame']}.png"
    state["h"], state["w"] = frame.shape[:2]
    state["detected_vehicles"] = None
    compute_edges()
    return jsonify(analyze_current_state())


@app.route("/api/simulation_3d", methods=["GET"])
def api_simulation_3d():
    """
    Returns 3D Digital Twin simulation telemetry synchronized with live.mp4.
    Includes world coordinates (x,y,z), stall dimensions, and procedural 3D vehicle profiles.
    """
    with video_state["lock"]:
        stats = dict(video_state["latest_stats"])
    payload = get_3d_simulation_state(stats)
    return jsonify(payload)


if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f"   [P] Smart Vacantrix - Universal Vision Server         ")
    print(f"   Access in browser: http://localhost:{port}          ")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)




