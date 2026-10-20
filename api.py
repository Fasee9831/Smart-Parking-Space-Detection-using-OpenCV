"""
ParkSense AI - REST API Microservice.
Provides HTTP JSON endpoints for integration with mobile apps, gate barriers, and smart city dashboards.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from flask import Flask, jsonify, request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.slot_manager import SlotManager, ParkingSlot
from core.detector import ParkingDetector
from core.tracker import ParkingTracker
from core.analytics import ParkingAnalytics

app = Flask(__name__)

# State
slot_manager = SlotManager()
detector = ParkingDetector(mode=ParkingDetector.MODE_HYBRID)
tracker = ParkingTracker()
base_image = cv2.imread(os.path.join(ROOT_DIR, "parking_lot.png"))


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current parking facility KPI summary."""
    img = base_image.copy() if base_image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    results, _, _ = detector.process_frame(img, slot_manager.slots)
    tracker.update_slot_states(results, {s.id: s for s in slot_manager.slots})
    summary = ParkingAnalytics.compute_summary(slot_manager.slots, results, tracker)
    return jsonify({
        "status": "success",
        "timestamp": cv2.getTickCount(),
        "data": summary
    })


@app.route("/api/slots", methods=["GET"])
def get_slots():
    """Lists all configured parking slots and their real-time occupancy status."""
    img = base_image.copy() if base_image is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    results, _, _ = detector.process_frame(img, slot_manager.slots)
    res_map = {r.slot_id: r for r in results}

    slot_type_filter = request.args.get("type")
    status_filter = request.args.get("status")

    data = []
    for s in slot_manager.slots:
        if slot_type_filter and s.slot_type != slot_type_filter:
            continue
        res = res_map.get(s.id)
        is_occ = res.is_occupied if res else False

        if status_filter:
            if status_filter.lower() == "vacant" and is_occ:
                continue
            if status_filter.lower() == "occupied" and not is_occ:
                continue

        ticket = tracker.active_sessions.get(s.id)
        data.append({
            "id": s.id,
            "type": s.slot_type,
            "bbox": s.bbox,
            "is_occupied": is_occ,
            "confidence": round(res.confidence, 3) if res else 0.0,
            "edge_ratio": round(res.edge_ratio, 3) if res else 0.0,
            "vehicle_type": res.vehicle_type if (res and is_occ) else None,
            "duration": ticket.duration_formatted if ticket else None,
            "accrued_fee": ticket.billed_amount if ticket else 0.0
        })

    return jsonify({
        "status": "success",
        "count": len(data),
        "slots": data
    })


@app.route("/api/detect", methods=["POST"])
def detect_image():
    """Processes an uploaded image and returns occupancy detection results."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No 'image' file provided in form-data"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400

    results, dl_boxes, _ = detector.process_frame(img, slot_manager.slots)
    summary = ParkingAnalytics.compute_summary(slot_manager.slots, results, tracker)

    return jsonify({
        "status": "success",
        "summary": summary,
        "results": [r.to_dict() for r in results],
        "vehicles_detected": dl_boxes
    })


@app.route("/api/slots", methods=["POST"])
def add_or_update_slot():
    """Adds or updates a slot via JSON payload."""
    payload = request.get_json(force=True)
    if not payload or "id" not in payload or "bbox" not in payload:
        return jsonify({"status": "error", "message": "Invalid slot payload. Required: 'id', 'bbox'"}), 400

    slot = ParkingSlot(
        slot_id=payload["id"],
        bbox=payload["bbox"],
        slot_type=payload.get("type", "Regular")
    )
    slot_manager.add_slot(slot)
    slot_manager.save_to_json()

    return jsonify({
        "status": "success",
        "message": f"Slot {slot.id} updated successfully",
        "slot": slot.to_dict()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParkSense AI REST API Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    args = parser.parse_args()

    print(f"\n🚀 ParkSense AI REST API running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
