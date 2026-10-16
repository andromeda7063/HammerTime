"""Artifact-backed HammerTime race replay and model diagnostics dashboard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
PROCESSED_DIR = Path(CONFIG["session"].get("processed_dir", "data/processed"))
if not PROCESSED_DIR.is_absolute():
    PROCESSED_DIR = PROJECT_DIR / PROCESSED_DIR
TELEMETRY_PATH = PROCESSED_DIR / "aligned_telemetry.parquet"
REPORT_PATH = PROCESSED_DIR / "evaluation_report.json"
CARDS_DIR = PROJECT_DIR / "broadcast" / "rendered"
VISUALS_DIR = PROJECT_DIR / "visuals"
CARBON = "#15151E"
CARD = "#1F1F2B"
RED = "#E10600"
WHITE = "#FFFFFF"

st.set_page_config(
    page_title="HammerTime · F1 Broadcast Insights",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"""<style>
    .stApp {{ background: {CARBON}; color: {WHITE}; }}
    h1, h2, h3 {{ color: {WHITE} !important; }}
    div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid #2D2D3D;
      border-radius: 8px; padding: 12px 18px; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading aligned race telemetry…")
def load_telemetry(path: str, modified: float) -> pd.DataFrame:
    del modified  # Included in the cache key so a newly written artifact is reloaded.
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading evaluation report…")
def load_report(path: str, modified: float) -> dict:
    del modified
    with open(path, "r", encoding="utf-8") as report_file:
        return json.load(report_file)


def artifact(path: Path, loader):
    if not path.is_file():
        return None
    return loader(str(path), path.stat().st_mtime)


def build_track_figure(frame: pd.DataFrame, selected_car: str, route: pd.DataFrame | None = None) -> go.Figure:
    """Draw positions and leader-to-follower edges from one aligned time slice."""
    fig = go.Figure()
    if frame.empty:
        return fig
    frame = frame.copy()
    frame = frame.dropna(subset=["Driver"])
    has_xy = {"X", "Y"}.issubset(frame.columns) and frame[["X", "Y"]].notna().to_numpy().all()
    if has_xy:
        frame = frame[(frame["X"] != 0) | (frame["Y"] != 0)]
    else:
        st.info("This telemetry artifact has no usable X/Y coordinates; the live track map is unavailable.")
        return fig

    if route is not None and {"X", "Y"}.issubset(route.columns):
        route = route.dropna(subset=["X", "Y"])
        if len(route) > 1:
            fig.add_trace(go.Scatter(
                x=route["X"], y=route["Y"], mode="lines",
                line={"color": "#555568", "width": 3}, hoverinfo="skip", showlegend=False,
            ))

    for _, row in frame.iterrows():
        leader = row.get("DriverAhead")
        gap = pd.to_numeric(row.get("DeltaT"), errors="coerce")
        if pd.notna(leader) and leader in set(frame["Driver"]) and pd.notna(gap) and 0 < gap <= 2.0:
            ahead = frame[frame["Driver"] == leader].iloc[0]
            color = RED if gap < 0.8 else "#FF9900"
            fig.add_trace(go.Scatter(
                x=[ahead["X"], row["X"]], y=[ahead["Y"], row["Y"]], mode="lines",
                line={"color": color, "width": max(1.5, float(np.exp(-(gap ** 2) / 2) * 6))},
                hovertemplate=f"{leader} → {row['Driver']} · gap {gap:.2f} s<extra></extra>",
                showlegend=False,
            ))

    colors = [RED if driver == selected_car else "#55C0B8" for driver in frame["Driver"]]
    fig.add_trace(go.Scatter(
        x=frame["X"], y=frame["Y"], mode="markers+text",
        marker={"size": 14, "color": colors, "line": {"color": WHITE, "width": 1}},
        text=frame["Driver"], textposition="top center",
        customdata=np.column_stack([
            frame.get("Speed", pd.Series(index=frame.index, dtype=float)),
            frame.get("DeltaT", pd.Series(index=frame.index, dtype=float)),
        ]),
        hovertemplate="%{text}<br>Speed %{customdata[0]:.0f} km/h<br>Gap %{customdata[1]:.2f} s<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        plot_bgcolor=CARD, paper_bgcolor=CARBON, font_color=WHITE, height=520,
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
    )
    return fig


def render_report_metrics(report: dict | None) -> None:
    if not report or not report.get("metrics"):
        st.info("Evaluation report is not available yet. Run the pipeline to publish metrics and physics results.")
        return
    metrics = report["metrics"]
    cols = st.columns(4)
    mae = metrics.get("mae_sec")
    r2 = metrics.get("r2_score")
    cols[0].metric("Test MAE", f"{mae:.3f} s" if mae is not None else "N/A")
    cols[1].metric("Test R²", f"{r2:.3f}" if r2 is not None else "N/A")
    tests = report.get("physics_tests", [])
    for index, title in enumerate(("Clean-air check", "DRS-train check")):
        result = tests[index] if index < len(tests) else None
        cols[index + 2].metric(title, ("PASS" if result.get("passed") else "FAIL") if result else "N/A")


def render_visual(path: Path, label: str) -> None:
    if path.is_file():
        st.image(str(path), caption=label, use_container_width=True)
    else:
        st.info(f"{label} has not been generated for this run.")


with st.sidebar:
    st.markdown("### Formula 1 · Broadcast Insights")
    st.title("HAMMERTIME ST-GNN")
    st.caption("Race telemetry and model evaluation")
    st.divider()
    artifact_override = st.text_input("Artifact directory", value=str(PROCESSED_DIR), help="Directory containing aligned_telemetry.parquet and evaluation_report.json")
    artifact_dir = Path(artifact_override).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_DIR / artifact_dir
    telemetry_path = artifact_dir / "aligned_telemetry.parquet"
    report_path = artifact_dir / "evaluation_report.json"
    telemetry = artifact(telemetry_path, load_telemetry)
    report = artifact(report_path, load_report)
    if telemetry is not None and "RaceId" in telemetry.columns:
        races = sorted(telemetry["RaceId"].dropna().astype(str).unique())
        selected_race = st.selectbox("Race", races) if races else None
        if selected_race is not None:
            telemetry = telemetry[telemetry["RaceId"].astype(str) == selected_race]
    else:
        selected_race = None
    if report and report.get("RaceId") is not None and selected_race is not None and str(report["RaceId"]) != selected_race:
        report = None
    if telemetry is not None and not telemetry.empty and "LapNumber" in telemetry:
        lap_values = sorted(pd.to_numeric(telemetry["LapNumber"], errors="coerce").dropna().astype(int).unique())
        selected_lap = st.select_slider("Race lap", options=lap_values, value=lap_values[min(len(lap_values) - 1, 30)])
        drivers = sorted(telemetry["Driver"].dropna().astype(str).unique())
    else:
        selected_lap, drivers = None, []
    selected_car = st.selectbox("Focus driver", drivers, index=0) if drivers else None
    st.caption(f"Telemetry: {'loaded' if telemetry is not None else 'not available'} · Evaluation: {'loaded' if report is not None else 'not available'}")

st.title("HammerTime · Race traffic insights")
tab_replay, tab_diagnostics, tab_studio = st.tabs(["Race replay", "Model diagnostics", "Broadcast cards"])

with tab_replay:
    if telemetry is None or telemetry.empty:
        st.info(f"Aligned telemetry is not available. Expected artifact: `{telemetry_path}`")
    else:
        lap_frame = telemetry[pd.to_numeric(telemetry["LapNumber"], errors="coerce") == selected_lap]
        time_value = pd.to_numeric(lap_frame["TimeSec"], errors="coerce").max()
        snapshot = lap_frame[pd.to_numeric(lap_frame["TimeSec"], errors="coerce") == time_value]
        route = lap_frame[lap_frame["Driver"].astype(str) == selected_car].sort_values("Distance") if "Distance" in lap_frame else None
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader(f"Track traffic · Lap {selected_lap}")
            st.plotly_chart(build_track_figure(snapshot, selected_car, route), use_container_width=True)
        with right:
            st.subheader("Traffic snapshot")
            if "DeltaT" in snapshot:
                board = snapshot.copy()
                board["Gap to car ahead (s)"] = pd.to_numeric(board["DeltaT"], errors="coerce")
                board["Wake"] = np.select(
                    [board["Gap to car ahead (s)"].between(0, 0.8, inclusive="left"), board["Gap to car ahead (s)"].between(0.8, 2.0, inclusive="both")],
                    ["Severe wake", "Wake"], default="Clear / unknown",
                )
                keep = [c for c in ["RaceId", "Driver", "DriverAhead", "Gap to car ahead (s)", "Speed", "MiniSector", "Wake"] if c in board]
                st.dataframe(board[keep].sort_values("Gap to car ahead (s)"), hide_index=True, use_container_width=True)
            focus = snapshot[snapshot["Driver"].astype(str) == selected_car]
            if not focus.empty:
                row = focus.iloc[0]
                st.metric("Gap to car ahead", f"{row.get('DeltaT', float('nan')):.2f} s")
                st.caption(f"Ahead: {row.get('DriverAhead', 'unknown')} · Mini-sector: {row.get('MiniSector', 'unknown')}")
    st.divider()
    st.subheader("Driver blocker analysis")
    if telemetry is not None and not telemetry.empty and selected_car is not None:
        focus_history = telemetry[telemetry["Driver"].astype(str) == selected_car].copy()
        if "LapNumber" in focus_history and "DeltaT" in focus_history:
            focus_history["LapNumber"] = pd.to_numeric(focus_history["LapNumber"], errors="coerce")
            focus_history["DeltaT"] = pd.to_numeric(focus_history["DeltaT"], errors="coerce")
            gap_history = focus_history.groupby("LapNumber", as_index=False)["DeltaT"].mean().dropna()
            if not gap_history.empty:
                history_fig = go.Figure()
                history_fig.add_hrect(y0=0, y1=0.8, fillcolor=RED, opacity=0.14, line_width=0, annotation_text="Severe wake")
                history_fig.add_hrect(y0=0.8, y1=2.0, fillcolor="#FF9900", opacity=0.10, line_width=0, annotation_text="Wake range")
                history_fig.add_hline(y=3.0, line_dash="dash", line_color="#10B981", annotation_text="Clean-air threshold")
                history_fig.add_trace(go.Scatter(x=gap_history["LapNumber"], y=gap_history["DeltaT"], mode="lines+markers", name=f"{selected_car} mean gap"))
                history_fig.update_layout(title=f"{selected_car} · mean gap to car ahead by lap", xaxis_title="Lap", yaxis_title="Gap (s)", plot_bgcolor=CARD, paper_bgcolor=CARBON, font_color=WHITE)
                st.plotly_chart(history_fig, use_container_width=True)
            close = focus_history[focus_history["DeltaT"].between(0, 2.0, inclusive="right")].dropna(subset=["DriverAhead"])
            if not close.empty:
                blockers = close.groupby("DriverAhead", as_index=False).agg(
                    observations=("DeltaT", "size"), mean_gap_sec=("DeltaT", "mean")
                ).sort_values(["observations", "mean_gap_sec"], ascending=[False, True])
                blockers = blockers.rename(columns={"DriverAhead": "Car ahead", "mean_gap_sec": "Mean observed gap (s)"})
                st.caption("Cars ahead ranked by observed close-gap exposure; proximity alone does not establish cause.")
                st.dataframe(blockers, hide_index=True, use_container_width=True)
            else:
                st.info("No close-gap observations are available for this driver.")
        else:
            st.info("Telemetry is missing lap or gap fields for this drilldown.")

with tab_diagnostics:
    st.subheader("Evaluation and physics checks")
    render_report_metrics(report)
    if report and report.get("metadata", {}).get("split", {}).get("test", {}).get("race_ids"):
        held_out_races = ", ".join(report["metadata"]["split"]["test"]["race_ids"])
        st.caption(f"Reported summary metrics cover the held-out split: {held_out_races}.")
    if report:
        baselines = report.get("baselines", {})
        if baselines:
            rows = []
            for name, scores in baselines.items():
                rows.append({"Model": name, "MAE (s)": scores.get("mae_sec"), "R²": scores.get("r2_score")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    prediction_rows = report.get("predictions", []) if report else []
    if selected_race is not None:
        prediction_rows = [row for row in prediction_rows if str(row.get("RaceId")) == selected_race]
    if prediction_rows:
        predictions = pd.DataFrame(prediction_rows)
        if {"y_true", "y_pred"}.issubset(predictions.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predictions["y_true"], y=predictions["y_pred"], mode="markers", name="Test predictions", marker={"color": "#00D2BE", "opacity": 0.55}))
            extent = [float(np.nanmin(predictions[["y_true", "y_pred"]].values)), float(np.nanmax(predictions[["y_true", "y_pred"]].values))]
            fig.add_trace(go.Scatter(x=extent, y=extent, mode="lines", name="Parity", line={"color": "#949498", "dash": "dash"}))
            fig.update_layout(title="Predicted vs actual delay", xaxis_title="Actual delay (s)", yaxis_title="Predicted delay (s)", plot_bgcolor=CARD, paper_bgcolor=CARBON, font_color=WHITE)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prediction-level diagnostics are unavailable until the evaluation report contains test predictions.")
    for visual_name in ("v5_baseline_duel.png", "v6_physics_cards.png", "v8_residuals.png", "v9_clean_air_cal.png"):
        render_visual(VISUALS_DIR / visual_name, visual_name.replace("_", " ").removesuffix(".png"))

with tab_studio:
    st.subheader("On-air insight cards")
    card_files = sorted(CARDS_DIR.glob("*.png")) if CARDS_DIR.is_dir() else []
    if card_files:
        for start in range(0, len(card_files), 2):
            columns = st.columns(2)
            for column, card_path in zip(columns, card_files[start:start + 2]):
                with column:
                    st.image(str(card_path), caption=card_path.stem.replace("_", " "), use_container_width=True)
    else:
        st.info(f"Broadcast cards have not been generated yet. Expected directory: `{CARDS_DIR}`")
