"""
Campus Pedestrian-Vehicle Risk Analyzer — Streamlit Demo
Run: streamlit run app.py
"""
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.detector_tracker import DetectorTracker
from src.main import _filter_static_pedestrians
from src.risk_scoring import RiskScorer
from src.utils import load_zone_config
from src.visualization import Visualizer

st.set_page_config(
    page_title="Campus Risk Analyzer",
    page_icon="🚦",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 Campus Risk Analyzer")
    st.caption("CISC 442/642 · Rajeev Reddy")
    st.divider()

    st.subheader("Video Source")
    use_sample = st.checkbox("Use recorded crosswalk video", value=True)
    uploaded = st.file_uploader("…or upload your own", type=["mp4", "mov", "avi"],
                                disabled=use_sample)

    st.subheader("Thresholds")
    cooldown_fr = st.slider("Event cooldown (frames)", 5, 60, 15)

    st.subheader("Config")
    config_path = st.text_input("Zone config", value="configs/zones_img5757.yaml")
    st.caption("Caution / danger distances and mpp are loaded from this YAML.")

    st.divider()
    run_btn = st.button("▶  Analyze Video", type="primary", use_container_width=True)

# ── Helper ─────────────────────────────────────────────────────────────────────
def process_video(video_path: str, cfg: dict) -> tuple[list, str]:
    """Run the full pipeline. Returns (events, annotated_video_path)."""
    detector = DetectorTracker()
    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_speed_px_per_frame = cfg["min_vehicle_speed_mph"] / (fps * cfg["meters_per_pixel"] * 2.237)
    scorer   = RiskScorer(
        cfg["polygon"], cfg["caution_distance_px"], cfg["danger_distance_px"],
        cooldown_frames=cooldown_fr,
        min_vehicle_speed_px_per_frame=min_speed_px_per_frame,
        ground_plane=cfg["ground_plane"],
        caution_distance_m=cfg["caution_distance_m"],
        danger_distance_m=cfg["danger_distance_m"],
        min_vehicle_speed_m_per_s=cfg["min_vehicle_speed_m_per_s"],
        zone_hysteresis_frames=3,
    )
    viz      = Visualizer(
        cfg["polygon"], fps=fps, meters_per_pixel=cfg["meters_per_pixel"],
        ground_plane=cfg["ground_plane"],
        min_vehicle_display_mph=cfg["min_vehicle_speed_mph"],
    )
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_file.name
    tmp_file.close()
    # Try H.264 first (browser-compatible); fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    track_history = defaultdict(lambda: deque(maxlen=30))
    all_events: list = []
    bar = st.progress(0, text="Initializing…")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets   = detector.detect_and_track(frame)
        dets   = _filter_static_pedestrians(dets, track_history)
        events = scorer.score_frame(dets, track_history, frame_idx, fps)
        all_events.extend(events)

        for d in dets:
            if d.get("is_tree"):
                continue
            track_history[d["id"]].append(d["bottom_center"])

        active_risks = scorer.compute_active_risks(dets, track_history, frame_idx, fps)
        annotated = viz.draw_frame(frame, dets, track_history, active_risks)
        writer.write(annotated)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1)
            bar.progress(pct, text=f"Frame {frame_idx}/{total} — {len(all_events)} events logged")

        frame_idx += 1

    cap.release()
    writer.release()
    bar.progress(1.0, text=f"Done — {frame_idx} frames, {len(all_events)} events")
    return all_events, out_path


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## Campus Pedestrian-Vehicle Risk Analyzer")
st.caption("Detects pedestrian–vehicle proximity and time-to-collision events inside a calibrated crosswalk zone.")

if run_btn:
    # Resolve video path
    if use_sample:
        video_path = "data/crosswalk.mp4"
        if not Path(video_path).exists():
            st.error("Sample video not found at data/crosswalk.mp4 — please upload one.")
            st.stop()
    elif uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
    else:
        st.warning("Select 'Use recorded crosswalk video' or upload a file.")
        st.stop()

    try:
        cfg = load_zone_config(config_path)
    except FileNotFoundError:
        st.error(f"Config not found: {config_path}")
        st.stop()

    with st.spinner("Running YOLO detector + risk scorer…"):
        events, out_path = process_video(video_path, cfg)

    st.session_state.update({
        "events":   events,
        "out_path": out_path,
        "ran":      True,
    })

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.get("ran"):
    events   = st.session_state["events"]
    out_path = st.session_state["out_path"]
    df       = pd.DataFrame(events) if events else pd.DataFrame(
        columns=["frame","timestamp_sec","person_id","vehicle_id",
                 "distance_px","ttc_sec","risk_label"])

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events",    len(df))
    c2.metric("TTC Warnings",    int((df["risk_label"] == "TTC_WARNING").sum()) if not df.empty else 0)
    c3.metric("Proximity Events",int((df["risk_label"] == "PROXIMITY").sum())   if not df.empty else 0)
    min_ttc = df["ttc_sec"].min() if not df.empty and df["ttc_sec"].notna().any() else None
    c4.metric("Closest TTC", f"{min_ttc:.2f}s" if min_ttc is not None else "—")

    st.divider()

    tab_vid, tab_log, tab_chart = st.tabs(["📹 Annotated Video", "📋 Event Log", "📈 Timeline"])

    with tab_vid:
        if Path(out_path).exists():
            with open(out_path, "rb") as f:
                st.video(f.read(), format="video/mp4")
        else:
            st.info("Annotated video not available.")

    with tab_log:
        if df.empty:
            st.info("No risk events detected inside the danger zone.")
        else:
            label_filter = st.multiselect(
                "Filter by label", options=df["risk_label"].unique().tolist(),
                default=df["risk_label"].unique().tolist()
            )
            filtered = df[df["risk_label"].isin(label_filter)].sort_values("frame")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                filtered.to_csv(index=False).encode(),
                file_name="risk_events.csv",
                mime="text/csv",
            )

    with tab_chart:
        if df.empty:
            st.info("No events to chart.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Timeline scatter
            colors = df["risk_label"].map({"TTC_WARNING": "#e63946", "PROXIMITY": "#f4a261"})
            axes[0].scatter(df["timestamp_sec"], df["distance_px"],
                            c=colors, alpha=0.8, edgecolors="white", linewidths=0.4, s=60)
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Distance (px)")
            axes[0].set_title("Risk Events Over Time")
            from matplotlib.patches import Patch
            axes[0].legend(handles=[
                Patch(color="#e63946", label="TTC_WARNING"),
                Patch(color="#f4a261", label="PROXIMITY"),
            ])

            # Label breakdown bar
            counts = df["risk_label"].value_counts()
            bar_colors = ["#e63946" if l == "TTC_WARNING" else "#f4a261" for l in counts.index]
            axes[1].bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
            axes[1].set_title("Event Breakdown")
            axes[1].set_ylabel("Count")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.info("Configure settings in the sidebar and click **▶ Analyze Video** to begin.")
