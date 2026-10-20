"""
Parking Analytics & KPI Computation Module.
Generates metrics, slot breakdown, historical simulations, and exportable reports.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ParkingAnalytics:
    """Computes real-time and aggregate statistics for parking intelligence."""

    @staticmethod
    def compute_summary(
        slots: List[Any],
        detection_results: List[Any],
        tracker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Compute key summary metrics."""
        total = len(slots)
        occupied = sum(1 for r in detection_results if r.is_occupied)
        vacant = max(0, total - occupied)
        occ_rate = (occupied / total * 100.0) if total > 0 else 0.0

        # Breakdown by slot type
        by_type: Dict[str, Dict[str, int]] = {
            "Regular": {"total": 0, "occupied": 0, "vacant": 0},
            "EV Charging": {"total": 0, "occupied": 0, "vacant": 0},
            "Handicap": {"total": 0, "occupied": 0, "vacant": 0},
            "VIP / Reserved": {"total": 0, "occupied": 0, "vacant": 0}
        }

        res_map = {r.slot_id: r for r in detection_results}
        for s in slots:
            stype = s.slot_type if s.slot_type in by_type else "Regular"
            by_type[stype]["total"] += 1
            is_occ = res_map.get(s.id).is_occupied if s.id in res_map else False
            if is_occ:
                by_type[stype]["occupied"] += 1
            else:
                by_type[stype]["vacant"] += 1

        revenue = tracker.get_realtime_revenue() if tracker else 0.0
        turnover = len(tracker.completed_sessions) if tracker else 0
        violations = sum(len(t.violations) for t in tracker.active_sessions.values()) if tracker else 0

        return {
            "total_slots": total,
            "occupied_slots": occupied,
            "vacant_slots": vacant,
            "occupancy_rate": round(occ_rate, 1),
            "by_type": by_type,
            "revenue": round(revenue, 2),
            "turnover_sessions": turnover,
            "active_violations": violations
        }

    @staticmethod
    def generate_hourly_trend_data() -> Dict[str, List[Any]]:
        """Simulate realistic 24-hour occupancy profile for parking facility analytics."""
        hours = [f"{h:02d}:00" for h in range(24)]
        # Typical bell curve peaking during morning rush (09:00 - 11:00) and afternoon (14:00 - 17:00)
        base_curve = [
            5, 3, 2, 2, 4, 10, 28, 55, 78, 88, 85, 82, 
            80, 84, 89, 91, 86, 75, 60, 48, 35, 22, 14, 8
        ]
        # Add slight natural variation
        np.random.seed(42)
        varied = [min(100, max(0, int(v + np.random.normal(0, 2)))) for v in base_curve]
        return {
            "hours": hours,
            "occupancy_pct": varied
        }

    @staticmethod
    def get_slot_table_rows(slots: List[Any], detection_results: List[Any], tracker: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Generate formatted tabular data for all slots for display or CSV export."""
        res_map = {r.slot_id: r for r in detection_results}
        rows = []
        for s in slots:
            res = res_map.get(s.id)
            is_occ = res.is_occupied if res else False
            ticket = tracker.active_sessions.get(s.id) if tracker else None

            duration_str = ticket.duration_formatted if ticket else ("-" if not is_occ else "Ongoing")
            fee_str = f"${ticket.billed_amount:.2f}" if ticket else ("-" if not is_occ else "$0.00")
            veh_type = (res.vehicle_type or "vehicle") if (res and is_occ) else "-"
            edge_pct = f"{int(res.edge_ratio * 100)}%" if res else "-"

            rows.append({
                "Slot ID": s.id,
                "Type": s.slot_type,
                "Status": "Occupied" if is_occ else "Vacant",
                "Vehicle": veh_type,
                "Confidence": f"{int((res.confidence if res else 0.0) * 100)}%",
                "Edge Density": edge_pct,
                "Duration": duration_str,
                "Accrued Fee": fee_str,
                "Violations": ", ".join(ticket.violations) if (ticket and ticket.violations) else "None"
            })
        return rows
