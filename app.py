"""
Smart Vacantrix - End-to-End Smart Parking Space Occupancy & Facility Management System
Built with Streamlit, OpenCV, and PyTorch.
"""

import os
import sys
import time
import json
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.slot_manager import SlotManager, ParkingSlot, SLOT_TYPES
from core.detector import ParkingDetector
from core.tracker import ParkingTracker
from core.analytics import ParkingAnalytics
from core.visualizer import ParkingVisualizer
from core.stream_simulator import StreamSimulator

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Vacantrix - Smart Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Header Container */
    .header-banner {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f3f4f6;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin: 0.2rem 0 0 0;
    }
    .status-badge-live {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.5px;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        font-weight: 600;
        border-top: 2px solid #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "slot_manager" not in st.session_state:
    st.session_state.slot_manager = SlotManager()

if "tracker" not in st.session_state:
    st.session_state.tracker = ParkingTracker()

if "stream_sim" not in st.session_state:
    st.session_state.stream_sim = StreamSimulator(
        os.path.join(ROOT_DIR, "parking_lot.png")
    )

if "selected_slot_id" not in st.session_state:
    st.session_state.selected_slot_id = None

if "simulation_tick" not in st.session_state:
    st.session_state.simulation_tick = 0

sm = st.session_state.slot_manager
tracker = st.session_state.tracker
sim = st.session_state.stream_sim


# --- Sidebar Navigation & Settings ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/parking.png", width=64)
    st.title("ParkSense Control")
    st.caption("AI-Powered Vision & Facility Platform")
    st.markdown("---")

    # 1. Media Source
    st.subheader("🎥 Video & Camera Feed")
    feed_mode = st.selectbox(
        "Input Stream Source",
        [
            "Sample Parking Lot (Image)",
            "Live Traffic Simulator (Video Stream)",
            "Upload Custom Image",
            "Upload Custom Video",
            "Webcam Stream (Direct)"
        ],
        index=0
    )

    uploaded_file = None
    if feed_mode in ["Upload Custom Image", "Upload Custom Video"]:
        uploaded_file = st.file_uploader(
            "Upload media file",
            type=["png", "jpg", "jpeg", "mp4", "avi"] if "Video" in feed_mode else ["png", "jpg", "jpeg"]
        )

    st.markdown("---")

    # 2. AI Model Settings
    st.subheader("🧠 Detection Engine")
    det_mode = st.selectbox(
        "Detection Algorithm",
        [
            ParkingDetector.MODE_HYBRID,
            ParkingDetector.MODE_EDGE,
            ParkingDetector.MODE_DL
        ],
        index=0,
        help="Hybrid combines neural vehicle bounding box detection with edge texture analysis."
    )

    edge_thresh = st.slider(
        "Edge Density Threshold",
        min_value=0.03,
        max_value=0.25,
        value=0.08,
        step=0.01,
        help="Higher values require more edges to consider a slot occupied."
    )

    dl_conf = st.slider(
        "Deep Learning Confidence",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum confidence for vehicle detection."
    )

    overlap_iou = st.slider(
        "Slot Overlap Ratio",
        min_value=0.10,
        max_value=0.60,
        value=0.25,
        step=0.05,
        help="Fraction of slot covered by detected vehicle to trigger occupancy."
    )

    st.markdown("---")

    # 3. Visual Overlays
    st.subheader("👁️ Display Options")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        show_ids = st.checkbox("Slot IDs", value=True)
        show_hud = st.checkbox("HUD Bar", value=True)
    with col_v2:
        show_conf = st.checkbox("Confidence %", value=True)
        show_dl = st.checkbox("Car Boxes", value=(det_mode != ParkingDetector.MODE_EDGE))

    overlay_alpha = st.slider("Overlay Transparency", 0.1, 0.7, 0.28, 0.05)

    st.markdown("---")

    # 4. Pricing & Rates
    st.subheader("💳 Parking Tariffs")
    currency = st.selectbox("Currency", ["$", "€", "£", "₹"], index=0)
    tracker.currency_symbol = currency

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        tracker.base_fee = st.number_input("Base Fee", value=tracker.base_fee, step=0.5)
        tracker.hourly_rate = st.number_input("Rate / Hr", value=tracker.hourly_rate, step=0.5)
    with col_r2:
        tracker.ev_surcharge_per_hr = st.number_input("EV Surcharge", value=tracker.ev_surcharge_per_hr, step=0.5)
        tracker.grace_period_mins = st.number_input("Grace Mins", value=tracker.grace_period_mins, step=5.0)

    tracker.max_stay_hours = st.slider("Max Stay Limit (Hrs)", 1.0, 12.0, 4.0, 0.5)

    st.markdown("---")
    st.caption("Smart Vacantrix Platform v2.0 • End-to-End Edition")


# --- Detector Instance ---
detector = ParkingDetector(
    mode=det_mode,
    edge_threshold=edge_thresh,
    dl_conf_threshold=dl_conf,
    overlap_threshold=overlap_iou
)


# --- Input Frame Resolution ---
current_frame = None
sim_event_msg = None

if feed_mode == "Sample Parking Lot (Image)":
    img_path = os.path.join(ROOT_DIR, "parking_lot.png")
    current_frame = cv2.imread(img_path)

elif feed_mode == "Live Traffic Simulator (Video Stream)":
    # Generate dynamic frame from simulator
    current_frame, sim_event_msg = sim.generate_simulated_frame(sm.slots)

elif feed_mode == "Upload Custom Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    current_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif feed_mode == "Webcam Stream (Direct)":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        current_frame = frame
    else:
        st.warning("Could not access local webcam. Reverting to sample lot.")
        current_frame = cv2.imread(os.path.join(ROOT_DIR, "parking_lot.png"))

if current_frame is None:
    # Fallback to sample lot
    current_frame = cv2.imread(os.path.join(ROOT_DIR, "parking_lot.png"))


# --- Process Detection ---
start_t = time.time()
detection_results, dl_boxes, edge_mask = detector.process_frame(current_frame, sm.slots)
proc_time_ms = (time.time() - start_t) * 1000.0
fps = 1000.0 / proc_time_ms if proc_time_ms > 0 else 30.0

# Update tracker states
slot_map = {s.id: s for s in sm.slots}
new_events = tracker.update_slot_states(detection_results, slot_map)

# Compute analytics
summary = ParkingAnalytics.compute_summary(sm.slots, detection_results, tracker)


# --- Top Header Banner ---
st.markdown(f"""
<div class="header-banner">
    <div>
        <h1 class="header-title">🅿️ Smart Vacantrix Facility Monitor</h1>
        <p class="header-subtitle">Real-time Vision Occupancy Detection, Automated Billing & Enforcement</p>
    </div>
    <div style="text-align: right;">
        <span class="status-badge-live">● SYSTEM ACTIVE</span>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 4px;">Lat: {proc_time_ms:.1f}ms • {fps:.1f} FPS • {det_mode}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Main KPI Cards Row ---
kpi_cols = st.columns(6)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Capacity</div>
        <div class="metric-val" style="color: #58a6ff;">{summary['total_slots']}</div>
        <div style="font-size: 0.75rem; color: #8b949e;">Configured Bays</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Vacant Spaces</div>
        <div class="metric-val" style="color: #2ea043;">{summary['vacant_slots']}</div>
        <div style="font-size: 0.75rem; color: #2ea043;">Available Now</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Occupied</div>
        <div class="metric-val" style="color: #f85149;">{summary['occupied_slots']}</div>
        <div style="font-size: 0.75rem; color: #f85149;">Cars Parked</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Occupancy Rate</div>
        <div class="metric-val" style="color: {'#f85149' if summary['occupancy_rate'] > 80 else '#e3b341'};">{summary['occupancy_rate']:.0f}%</div>
        <div style="font-size: 0.75rem; color: #8b949e;">Facility Load</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    ev_vacant = summary['by_type']['EV Charging']['vacant']
    hc_vacant = summary['by_type']['Handicap']['vacant']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Special Bays Free</div>
        <div class="metric-val" style="color: #38bdf8;">{ev_vacant + hc_vacant}</div>
        <div style="font-size: 0.75rem; color: #8b949e;">⚡ EV: {ev_vacant} | ♿ HC: {hc_vacant}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[5]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Est. Revenue</div>
        <div class="metric-val" style="color: #3fb950;">{currency}{summary['revenue']:.2f}</div>
        <div style="font-size: 0.75rem; color: #8b949e;">Active + Settled</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)


# --- Tabs Organization ---
tab_surv, tab_editor, tab_analytics, tab_billing, tab_api = st.tabs([
    "📊 Live Surveillance",
    "📐 Calibration Studio & Editor",
    "📈 Analytics & Heatmaps",
    "💳 Smart Billing & Violations",
    "🔌 REST API & Integration"
])


# =========================================================================
# TAB 1: LIVE SURVEILLANCE
# =========================================================================
with tab_surv:
    col_feed, col_side = st.columns([7, 3])

    with col_feed:
        # Render visual overlay
        rendered_frame = ParkingVisualizer.render_overlay(
            image=current_frame,
            slots=sm.slots,
            detection_results=detection_results,
            detected_vehicles=dl_boxes,
            selected_slot_id=st.session_state.selected_slot_id,
            show_ids=show_ids,
            show_confidence=show_conf,
            show_dl_boxes=show_dl,
            show_hud=show_hud,
            summary=summary,
            fps=fps,
            alpha=overlay_alpha
        )

        view_option = st.radio(
            "Display Feed View:",
            ["Annotated Overlay", "Raw Camera Feed", "Computer Vision Edge Mask"],
            horizontal=True
        )

        if view_option == "Annotated Overlay":
            rgb_display = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_display, use_container_width=True, caption=f"Smart Vacantrix Live Vision Overlay ({fps:.1f} FPS)")
        elif view_option == "Raw Camera Feed":
            rgb_display = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_display, use_container_width=True, caption="Raw Surveillance Stream")
        else:
            st.image(edge_mask, use_container_width=True, caption="Canny Edge & Dilated Texture Mask")

        # Simulator control row if in simulator mode
        if feed_mode == "Live Traffic Simulator (Video Stream)":
            st.markdown("---")
            sim_c1, sim_c2, sim_c3 = st.columns([2, 2, 6])
            with sim_c1:
                if st.button("🔄 Step / Simulate Traffic Tick", use_container_width=True):
                    st.session_state.simulation_tick += 1
                    st.rerun()
            with sim_c2:
                auto_refresh = st.checkbox("Auto-Tick Feed (Loop)", value=False)
                if auto_refresh:
                    time.sleep(1.5)
                    st.session_state.simulation_tick += 1
                    st.rerun()
            with sim_c3:
                if sim_event_msg:
                    st.info(f"🚦 Simulator Event: {sim_event_msg}")

    with col_side:
        st.subheader("⚡ Live Bay Availability")

        # Visual mini cards for each slot category
        for cat_name, cat_data in summary["by_type"].items():
            tot = cat_data["total"]
            occ = cat_data["occupied"]
            vac = cat_data["vacant"]
            pct = (occ / tot * 100) if tot > 0 else 0
            
            icon = "🚗" if cat_name == "Regular" else ("⚡" if "EV" in cat_name else ("♿" if "Handicap" in cat_name else "👑"))
            st.markdown(f"**{icon} {cat_name}** ({vac} of {tot} Free)")
            st.progress(pct / 100.0)

        st.markdown("---")
        st.subheader("📋 Recent Access Events")

        if tracker.event_log:
            recent_events = tracker.event_log[-8:]
            for ev in reversed(recent_events):
                badge_color = "#2ea043" if ev["type"] == "EXIT" else ("#f85149" if ev["type"] == "VIOLATION" else "#58a6ff")
                st.markdown(f"""
                <div style="background: #161b22; border-left: 3px solid {badge_color}; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px; font-size: 0.85rem;">
                    <span style="color: #8b949e; font-size: 0.75rem;">{ev['timestamp']}</span> • <b>{ev['slot_id']}</b>
                    <br/><span style="color: #c9d1d9;">{ev['description']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No events recorded yet. Arriving and departing cars will log here.")


# =========================================================================
# TAB 2: CALIBRATION STUDIO & EDITOR
# =========================================================================
with tab_editor:
    st.subheader("📐 Interactive Parking Space Annotator & Calibration")
    st.caption("Define, position, calibrate, and categorize parking bays with real-time feedback.")

    ed_c1, ed_c2 = st.columns([7, 3])

    with ed_c1:
        # Show image with selected slot highlighted
        annotated_preview = ParkingVisualizer.render_overlay(
            image=current_frame,
            slots=sm.slots,
            detection_results=detection_results,
            selected_slot_id=st.session_state.selected_slot_id,
            show_ids=True,
            show_confidence=True,
            show_hud=False,
            alpha=0.35
        )
        st.image(
            cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption="Calibration Canvas: Selected slot is highlighted with a white border"
        )

    with ed_c2:
        slot_action = st.radio(
            "Action Mode:",
            ["Edit Existing Slot", "Add New Slot", "Auto-Generate Grid", "Preset Manager"],
            horizontal=False
        )

        st.markdown("---")

        if slot_action == "Edit Existing Slot":
            slot_ids = [s.id for s in sm.slots]
            if slot_ids:
                curr_sel = st.selectbox("Select Slot to Modify:", slot_ids)
                st.session_state.selected_slot_id = curr_sel
                target_slot = sm.get_slot(curr_sel)

                if target_slot:
                    new_type = st.selectbox(
                        "Slot Category:",
                        SLOT_TYPES,
                        index=SLOT_TYPES.index(target_slot.slot_type) if target_slot.slot_type in SLOT_TYPES else 0
                    )
                    target_slot.slot_type = new_type

                    st.markdown("**Bounding Box Coordinates:**")
                    x1 = st.number_input("X1 (Left)", value=target_slot.x1, step=5)
                    y1 = st.number_input("Y1 (Top)", value=target_slot.y1, step=5)
                    x2 = st.number_input("X2 (Right)", value=target_slot.x2, step=5)
                    y2 = st.number_input("Y2 (Bottom)", value=target_slot.y2, step=5)

                    target_slot.bbox = [int(x1), int(y1), int(x2), int(y2)]

                    col_del, col_save = st.columns(2)
                    with col_del:
                        if st.button("🗑️ Delete Slot", use_container_width=True):
                            sm.remove_slot(curr_sel)
                            sm.save_to_json()
                            st.session_state.selected_slot_id = None
                            st.rerun()
                    with col_save:
                        if st.button("💾 Save Changes", use_container_width=True):
                            sm.save_to_json()
                            st.success(f"Slot {curr_sel} updated!")
                            st.rerun()
            else:
                st.warning("No slots configured. Use Add New Slot or Auto-Generate Grid.")

        elif slot_action == "Add New Slot":
            new_id = st.text_input("New Slot ID:", value=f"S-{len(sm.slots)+1:02d}")
            new_type = st.selectbox("Category:", SLOT_TYPES, index=0)
            st.markdown("**Position Coordinates:**")
            col_nx, col_ny = st.columns(2)
            with col_nx:
                nx1 = st.number_input("X1", value=200, step=10)
                nx2 = st.number_input("X2", value=294, step=10)
            with col_ny:
                ny1 = st.number_input("Y1", value=500, step=10)
                ny2 = st.number_input("Y2", value=562, step=10)

            if st.button("➕ Add Slot", use_container_width=True):
                new_s = ParkingSlot(new_id, [int(nx1), int(ny1), int(nx2), int(ny2)], new_type)
                sm.add_slot(new_s)
                sm.save_to_json()
                st.session_state.selected_slot_id = new_id
                st.success(f"Added slot {new_id}!")
                st.rerun()

        elif slot_action == "Auto-Generate Grid":
            st.markdown("**Matrix / Row Parameters:**")
            g_prefix = st.text_input("Row Prefix:", value="ROW")
            g_type = st.selectbox("Category for Grid:", SLOT_TYPES, index=0)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                g_rows = st.number_input("Rows", value=1, min_value=1, max_value=10)
                g_cols = st.number_input("Columns", value=6, min_value=1, max_value=20)
                g_w = st.number_input("Slot Width (px)", value=94, min_value=30)
                g_h = st.number_input("Slot Height (px)", value=62, min_value=30)
            with col_g2:
                g_sx = st.number_input("Start X", value=50, step=10)
                g_sy = st.number_input("Start Y", value=500, step=10)
                g_gx = st.number_input("Gap X", value=15, step=2)
                g_gy = st.number_input("Gap Y", value=20, step=2)

            if st.button("⚡ Generate Grid", use_container_width=True):
                sm.generate_grid(
                    start_x=int(g_sx),
                    start_y=int(g_sy),
                    rows=int(g_rows),
                    cols=int(g_cols),
                    slot_w=int(g_w),
                    slot_h=int(g_h),
                    gap_x=int(g_gx),
                    gap_y=int(g_gy),
                    prefix=g_prefix,
                    slot_type=g_type
                )
                sm.save_to_json()
                st.success("Grid generated successfully!")
                st.rerun()

        elif slot_action == "Preset Manager":
            if st.button("🔄 Reset to Default Layout", use_container_width=True):
                default_file = os.path.join(ROOT_DIR, "data", "default_slots.json")
                sm.load_from_json(default_file)
                st.success("Default layout loaded!")
                st.rerun()

            if st.button("🧹 Clear All Slots", use_container_width=True):
                sm.clear()
                sm.save_to_json()
                st.warning("All slots cleared.")
                st.rerun()

            # Download JSON
            json_str = json.dumps([s.to_dict() for s in sm.slots], indent=2)
            st.download_button(
                "📥 Download Slots JSON",
                data=json_str,
                file_name="parking_slots_config.json",
                mime="application/json",
                use_container_width=True
            )


# =========================================================================
# TAB 3: ANALYTICS & HEATMAPS
# =========================================================================
with tab_analytics:
    st.subheader("📈 Facility Occupancy Analytics & Insights")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        # Donut Chart for Current Status
        donut_data = pd.DataFrame({
            "Status": ["Vacant", "Occupied"],
            "Count": [summary["vacant_slots"], summary["occupied_slots"]]
        })
        fig_donut = px.pie(
            donut_data,
            values="Count",
            names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map={"Vacant": "#2ea043", "Occupied": "#f85149"},
            title="Real-Time Occupancy Distribution"
        )
        fig_donut.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#e6edf3",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_c2:
        # Category breakdown bar chart
        cat_records = []
        for cname, cdata in summary["by_type"].items():
            cat_records.append({"Category": cname, "Status": "Vacant", "Count": cdata["vacant"]})
            cat_records.append({"Category": cname, "Status": "Occupied", "Count": cdata["occupied"]})
        df_cat = pd.DataFrame(cat_records)

        fig_bar = px.bar(
            df_cat,
            x="Category",
            y="Count",
            color="Status",
            barmode="stack",
            color_discrete_map={"Vacant": "#2ea043", "Occupied": "#f85149"},
            title="Occupancy by Bay Category"
        )
        fig_bar.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#e6edf3",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Historical 24-hour Occupancy Curve
    trend_data = ParkingAnalytics.generate_hourly_trend_data()
    df_trend = pd.DataFrame(trend_data)
    fig_trend = px.line(
        df_trend,
        x="hours",
        y="occupancy_pct",
        title="24-Hour Historical Occupancy Profile (Peak Hours Analysis)",
        markers=True
    )
    fig_trend.update_traces(line_color="#58a6ff", line_width=3)
    fig_trend.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis_title="Time of Day",
        yaxis_title="Occupancy Rate (%)",
        yaxis_range=[0, 100],
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Detailed Slot Utilization Table
    st.subheader("📋 Comprehensive Slot Audit Table")
    slot_rows = ParkingAnalytics.get_slot_table_rows(sm.slots, detection_results, tracker)
    df_slots = pd.DataFrame(slot_rows)
    st.dataframe(df_slots, use_container_width=True)

    # CSV Export Button
    csv_bytes = df_slots.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Audit Log (CSV)",
        data=csv_bytes,
        file_name="parking_audit_report.csv",
        mime="text/csv"
    )


# =========================================================================
# TAB 4: SMART BILLING & VIOLATIONS
# =========================================================================
with tab_billing:
    st.subheader("💳 Parking Tariff & Enforcement Center")

    b_col1, b_col2 = st.columns([6, 4])

    with b_col1:
        st.markdown("### Active Parking Sessions")
        active_tickets = tracker.get_active_tickets_table()
        if active_tickets:
            df_active = pd.DataFrame(active_tickets)
            st.dataframe(df_active, use_container_width=True)
        else:
            st.info("No active billable sessions currently recorded.")

        st.markdown("### Settled Transactions")
        completed_tickets = tracker.get_completed_tickets_table()
        if completed_tickets:
            df_comp = pd.DataFrame(completed_tickets)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.caption("No completed transactions yet.")

    with b_col2:
        st.markdown("### Enforcement & Security Alerts")
        violations_found = []
        for t in tracker.active_sessions.values():
            if t.violations:
                violations_found.append(t)

        if violations_found:
            for v in violations_found:
                st.error(f"⚠️ **Violation at Slot {v.slot_id}**\n- Types: {', '.join(v.violations)}\n- Duration: {v.duration_formatted}")
        else:
            st.success("✅ Zero active parking violations detected. Facility running smoothly.")

        st.markdown("---")
        st.markdown("### Current Tariff Schedule")
        st.markdown(f"""
        - **Base Entry Fee:** {currency}{tracker.base_fee:.2f}
        - **Standard Rate:** {currency}{tracker.hourly_rate:.2f} / hour
        - **EV Surcharge:** +{currency}{tracker.ev_surcharge_per_hr:.2f} / hour
        - **Grace Period:** {tracker.grace_period_mins:.0f} minutes (Free)
        - **Max Overstay Limit:** {tracker.max_stay_hours:.1f} hours
        - **Daily Cap:** {currency}{tracker.max_daily_cap:.2f}
        """)


# =========================================================================
# TAB 5: REST API & INTEGRATION GUIDE
# =========================================================================
with tab_api:
    st.subheader("🔌 External API Integration")
    st.caption("Integrate Smart Vacantrix with digital street signs, barrier gates, mobile apps, and municipal parking networks.")

    st.markdown("""
    Smart Vacantrix provides a lightweight REST API server located at `api.py`.
    
    #### Start the REST API:
    ```bash
    python api.py --port 5000
    ```
    
    #### Available Endpoints:
    """)

    api_c1, api_c2 = st.columns(2)
    with api_c1:
        st.markdown("""
        **1. Get Facility Status Summary**
        - **Route:** `GET /api/status`
        - **Sample Response:**
        ```json
        {
          "total_slots": 24,
          "occupied_slots": 12,
          "vacant_slots": 12,
          "occupancy_rate": 50.0,
          "available_ev": 2,
          "available_handicap": 1,
          "revenue": 45.50
        }
        ```
        """)

    with api_c2:
        st.markdown("""
        **2. List All Parking Slots**
        - **Route:** `GET /api/slots`
        - **Query Params:** `?type=EV%20Charging&status=Vacant`
        - **Sample Response:**
        ```json
        [
          {
            "id": "A-04",
            "type": "EV Charging",
            "is_occupied": false,
            "confidence": 0.95
          }
        ]
        ```
        """)

    st.markdown("---")
    st.markdown("""
    **3. Run Inference on Uploaded Camera Image**
    - **Route:** `POST /api/detect`
    - **Body:** `multipart/form-data` with `image` file
    - **Returns:** Occupancy status and vehicle coordinates for all configured slots.
    """)
