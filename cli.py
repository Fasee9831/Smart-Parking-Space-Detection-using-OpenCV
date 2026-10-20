"""
Smart Vacantrix - Command Line Interface (CLI).
Enables headless batch processing, automated report generation, and exports.
"""

import os
import sys
import argparse
import json
import cv2
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.slot_manager import SlotManager
from core.detector import ParkingDetector
from core.tracker import ParkingTracker
from core.analytics import ParkingAnalytics
from core.visualizer import ParkingVisualizer


def run_cli():
    parser = argparse.ArgumentParser(description="Smart Vacantrix Command Line Facility Inspector")
    parser.add_argument("--source", type=str, default="parking_lot.png", help="Path to input image or video")
    parser.add_argument("--slots", type=str, default=None, help="Path to custom parking slots JSON")
    parser.add_argument("--output", type=str, default="output_annotated.png", help="Path to save annotated image")
    parser.add_argument("--mode", type=str, default=ParkingDetector.MODE_HYBRID, help="Detection algorithm")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export slot report as CSV")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export report as JSON")
    args = parser.parse_args()

    print("\n=======================================================")
    print("        Smart Vacantrix - Batch Vision Inspector          ")
    print("=======================================================")

    slot_manager = SlotManager(default_file=args.slots)
    print(f"[*] Loaded {len(slot_manager.slots)} slot definitions.")

    detector = ParkingDetector(mode=args.mode)
    print(f"[*] Engine mode: {args.mode}")

    if not os.path.exists(args.source):
        print(f"[!] Error: Source file '{args.source}' does not exist.")
        sys.exit(1)

    img = cv2.imread(args.source)
    if img is None:
        print(f"[!] Error: Failed to read image from '{args.source}'")
        sys.exit(1)

    print(f"[*] Processing frame ({img.shape[1]}x{img.shape[0]})...")
    results, dl_boxes, _ = detector.process_frame(img, slot_manager.slots)

    tracker = ParkingTracker()
    tracker.update_slot_states(results, {s.id: s for s in slot_manager.slots})
    summary = ParkingAnalytics.compute_summary(slot_manager.slots, results, tracker)

    # Console Report
    print("\n---------------- FACILITY SUMMARY ----------------")
    print(f" Total Spaces     : {summary['total_slots']}")
    print(f" Vacant (Free)    : {summary['vacant_slots']}")
    print(f" Occupied         : {summary['occupied_slots']}")
    print(f" Occupancy Rate   : {summary['occupancy_rate']}%")
    print(f" Active Violations: {summary['active_violations']}")
    print(" Breakdown by Category:")
    for cname, cdata in summary["by_type"].items():
        print(f"   - {cname:15}: {cdata['vacant']} free / {cdata['total']} total")
    print("--------------------------------------------------\n")

    # Render & Save Output Image
    if args.output:
        rendered = ParkingVisualizer.render_overlay(
            image=img,
            slots=slot_manager.slots,
            detection_results=results,
            detected_vehicles=dl_boxes,
            summary=summary,
            show_ids=True,
            show_hud=True
        )
        cv2.imwrite(args.output, rendered)
        print(f"[+] Annotated image successfully written to: {args.output}")

    # Export CSV
    rows = ParkingAnalytics.get_slot_table_rows(slot_manager.slots, results, tracker)
    if args.export_csv:
        df = pd.DataFrame(rows)
        df.to_csv(args.export_csv, index=False)
        print(f"[+] Audit CSV report exported to: {args.export_csv}")

    # Export JSON
    if args.export_json:
        report = {
            "source": args.source,
            "summary": summary,
            "slots": rows
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[+] JSON report exported to: {args.export_json}")

    print("\n[OK] Inspection complete.\n")


if __name__ == "__main__":
    run_cli()
