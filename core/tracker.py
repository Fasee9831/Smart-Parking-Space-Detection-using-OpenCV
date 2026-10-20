"""
Parking Session Tracker, Billing & Violations Engine.
Tracks duration, real-time fee calculation, overstay alerts, and event logging.
"""

import time
import math
from typing import Dict, Any, List, Optional
from datetime import datetime


class ParkingTicket:
    """Represents a billing ticket for a parking session."""

    def __init__(
        self,
        slot_id: str,
        slot_type: str,
        entry_time: float,
        vehicle_type: Optional[str] = "car"
    ):
        self.ticket_id = f"TKT-{slot_id}-{int(entry_time)}"
        self.slot_id = slot_id
        self.slot_type = slot_type
        self.entry_time = entry_time
        self.exit_time: Optional[float] = None
        self.vehicle_type = vehicle_type
        self.billed_amount: float = 0.0
        self.is_active: bool = True
        self.violations: List[str] = []

    @property
    def duration_seconds(self) -> float:
        end = self.exit_time if self.exit_time is not None else time.time()
        return max(0.0, end - self.entry_time)

    @property
    def duration_formatted(self) -> str:
        secs = int(self.duration_seconds)
        mins, s = divmod(secs, 60)
        hrs, m = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}h {m}m {s}s"
        return f"{m}m {s}s"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "slot_id": self.slot_id,
            "slot_type": self.slot_type,
            "entry_time": datetime.fromtimestamp(self.entry_time).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": datetime.fromtimestamp(self.exit_time).strftime("%Y-%m-%d %H:%M:%S") if self.exit_time else "Active",
            "duration": self.duration_formatted,
            "duration_secs": round(self.duration_seconds, 1),
            "vehicle_type": self.vehicle_type or "Unknown",
            "billed_amount": round(self.billed_amount, 2),
            "status": "Active" if self.is_active else "Completed",
            "violations": ", ".join(self.violations) if self.violations else "None"
        }


class ParkingTracker:
    """Session tracker, revenue accountant, and security alert monitor."""

    def __init__(
        self,
        base_fee: float = 2.0,
        hourly_rate: float = 3.5,
        ev_surcharge_per_hr: float = 1.5,
        grace_period_mins: float = 10.0,
        max_stay_hours: float = 4.0,
        max_daily_cap: float = 35.0,
        currency_symbol: str = "$"
    ):
        self.base_fee = base_fee
        self.hourly_rate = hourly_rate
        self.ev_surcharge_per_hr = ev_surcharge_per_hr
        self.grace_period_mins = grace_period_mins
        self.max_stay_hours = max_stay_hours
        self.max_daily_cap = max_daily_cap
        self.currency_symbol = currency_symbol

        # State storage
        self.active_sessions: Dict[str, ParkingTicket] = {}
        self.completed_sessions: List[ParkingTicket] = []
        self.event_log: List[Dict[str, Any]] = []
        self.total_revenue_collected: float = 0.0

    def calculate_fee(self, duration_seconds: float, slot_type: str) -> float:
        """Calculate parking fee based on duration and slot category."""
        duration_mins = duration_seconds / 60.0
        if duration_mins <= self.grace_period_mins:
            return 0.0  # Free within grace period

        # Round up to next full 30-min block for fairness
        billable_hours = math.ceil(duration_mins / 30.0) * 0.5

        rate = self.hourly_rate
        if slot_type == "EV Charging":
            rate += self.ev_surcharge_per_hr
        elif slot_type == "VIP / Reserved":
            rate += 2.0

        fee = self.base_fee + (billable_hours * rate)
        return min(self.max_daily_cap, round(fee, 2))

    def update_slot_states(self, detection_results: List[Any], slot_map: Dict[str, Any], current_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Update states of all slots given fresh detection results.
        Detects transitions: Empty -> Occupied (Check-in), Occupied -> Empty (Check-out).
        Returns list of new events generated during this tick.
        """
        now = current_timestamp or time.time()
        new_events = []

        for res in detection_results:
            sid = res.slot_id
            slot = slot_map.get(sid)
            stype = slot.slot_type if slot else "Regular"
            is_occ = res.is_occupied

            if is_occ:
                if sid not in self.active_sessions:
                    # Vehicle Checked In
                    ticket = ParkingTicket(
                        slot_id=sid,
                        slot_type=stype,
                        entry_time=now,
                        vehicle_type=res.vehicle_type or "car"
                    )
                    self.active_sessions[sid] = ticket
                    event = {
                        "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                        "slot_id": sid,
                        "type": "ENTRY",
                        "description": f"Vehicle ({res.vehicle_type or 'car'}) arrived at {stype} slot {sid}",
                        "severity": "info"
                    }
                    self.event_log.append(event)
                    new_events.append(event)
                else:
                    # Ongoing session: check overstay violation
                    ticket = self.active_sessions[sid]
                    duration_hrs = (now - ticket.entry_time) / 3600.0
                    if duration_hrs > self.max_stay_hours and "Overstay" not in ticket.violations:
                        ticket.violations.append("Overstay")
                        event = {
                            "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                            "slot_id": sid,
                            "type": "VIOLATION",
                            "description": f"Overstay Alert: Slot {sid} parked for {ticket.duration_formatted} (Limit: {self.max_stay_hours}h)",
                            "severity": "warning"
                        }
                        self.event_log.append(event)
                        new_events.append(event)

                    # Update current bill estimate
                    ticket.billed_amount = self.calculate_fee(ticket.duration_seconds, stype)

            else:
                if sid in self.active_sessions:
                    # Vehicle Departed / Checked Out
                    ticket = self.active_sessions.pop(sid)
                    ticket.exit_time = now
                    ticket.is_active = False
                    ticket.billed_amount = self.calculate_fee(ticket.duration_seconds, stype)
                    self.completed_sessions.append(ticket)
                    self.total_revenue_collected += ticket.billed_amount

                    event = {
                        "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                        "slot_id": sid,
                        "type": "EXIT",
                        "description": f"Vehicle departed slot {sid} ({ticket.duration_formatted}). Billed: {self.currency_symbol}{ticket.billed_amount:.2f}",
                        "severity": "success"
                    }
                    self.event_log.append(event)
                    new_events.append(event)

        return new_events

    def get_realtime_revenue(self) -> float:
        """Calculates total revenue collected plus accrued active tickets."""
        active_accrued = sum(t.billed_amount for t in self.active_sessions.values())
        return self.total_revenue_collected + active_accrued

    def get_active_tickets_table(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.active_sessions.values()]

    def get_completed_tickets_table(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.completed_sessions]
