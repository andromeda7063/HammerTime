"""
Visual V7: "Who Is Slowing You Down" Drilldown
Per-car analysis: Top traffic blockers, cumulative seconds lost, and gap progression history.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from theme import (
    apply_broadcast_theme, CARBON_DARK, CARD_BG, F1_RED, PURE_WHITE,
    TEXT_MUTED, DRS_CYAN, TEAM_COLORS
)


def generate_blocker_drilldown(
    focus_driver: str = "HAM",
    total_seconds_lost: float = None,
    top_blockers: list = None,
    lap_history: np.ndarray = None,
    gap_history: np.ndarray = None,
    output_path: str = "visuals/v7_blockers.png"
):
    """
    Renders V7 "Who Is Slowing You Down" driver drilldown card and gap history.
    top_blockers: list of tuples (leader_driver, seconds_lost, mean_gap)
    """
    apply_broadcast_theme()
    fig, (ax_info, ax_plot) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(CARBON_DARK)

    # --- Left Panel: Blocker Summary Card ---
    ax_info.set_facecolor(CARD_BG)
    ax_info.axis("off")

    rect = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        edgecolor=F1_RED, facecolor="#14141E", lw=1.5
    )
    ax_info.add_patch(rect)

    # Focus driver badge
    d_color = TEAM_COLORS.get(focus_driver, PURE_WHITE)
    ax_info.text(0.08, 0.88, "DRIVER DRILLDOWN · TRAFFIC IMPACT", color=F1_RED, fontsize=10, fontweight="heavy")
    ax_info.text(0.08, 0.80, f"{focus_driver} · LIVE TELEMETRY", color=PURE_WHITE, fontsize=20, fontweight="heavy")

    ax_info.text(0.08, 0.68, "TOTAL ESTIMATED DELAY LOST TO TRAFFIC:", color=TEXT_MUTED, fontsize=9)
    lost_label = f"+{total_seconds_lost:.2f} s" if total_seconds_lost is not None else "NOT AVAILABLE"
    ax_info.text(0.08, 0.58, lost_label, color=F1_RED, fontsize=24, fontweight="heavy")

    ax_info.text(0.08, 0.46, "PRIMARY DISTURBANCE SOURCES (BLOCKERS):", color=TEXT_MUTED, fontsize=9, fontweight="bold")

    y_pos = 0.36
    for b_drv, b_sec, b_gap in (top_blockers or []):
        b_col = TEAM_COLORS.get(b_drv, PURE_WHITE)
        ax_info.text(0.08, y_pos, f"● {b_drv}", color=b_col, fontsize=12, fontweight="heavy")
        ax_info.text(0.28, y_pos, f"+{b_sec:.2f} s lost", color=PURE_WHITE, fontsize=11, fontweight="bold")
        ax_info.text(0.65, y_pos, f"Avg Gap: {b_gap:.2f} s", color=TEXT_MUTED, fontsize=10)
        y_pos -= 0.10

    # --- Right Panel: Gap History Timeline ---
    ax_plot.set_facecolor(CARD_BG)
    if lap_history is not None and gap_history is not None:
        lap_history = np.asarray(lap_history, dtype=float).reshape(-1)
        gap_history = np.asarray(gap_history, dtype=float).reshape(-1)
        if lap_history.size != gap_history.size:
            raise ValueError("lap_history and gap_history must have the same length")
        valid = np.isfinite(lap_history) & np.isfinite(gap_history)
        ax_plot.plot(lap_history[valid], gap_history[valid], color=d_color, lw=2.2, label=f"{focus_driver} Gap to Car Ahead")
    else:
        ax_plot.text(0.5, 0.5, "Gap history unavailable", transform=ax_plot.transAxes,
                     color=TEXT_MUTED, ha="center", va="center")
    ax_plot.axhspan(0.0, 0.8, color=F1_RED, alpha=0.15, label="Severe Dirty Air Zone (< 0.8 s)")
    ax_plot.axhspan(0.8, 2.0, color="#FF9900", alpha=0.10, label="Dynamic Aerodynamic Wake (0.8 - 2.0 s)")
    ax_plot.axhline(3.0, color="#10B981", linestyle="--", lw=1.2, label="Clean Air Threshold (> 3.0 s)")

    ax_plot.set_xlabel("Race Lap Number", fontsize=10, labelpad=8)
    ax_plot.set_ylabel("Time Gap Δt to Car Ahead (seconds)", fontsize=10, labelpad=8)
    ax_plot.set_title(f"{focus_driver} PROXIMITY & AERODYNAMIC DISTURBANCE HISTORY", color=PURE_WHITE, fontsize=11, fontweight="bold", pad=10)
    ax_plot.set_ylim(0, 4.5)
    ax_plot.grid(True, linestyle="--", alpha=0.3)

    leg = ax_plot.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg.get_texts(): t.set_color(PURE_WHITE)

    fig.suptitle(f"F1 BROADCAST INSIGHTS · 'WHO IS SLOWING YOU DOWN' ({focus_driver})",
                 color=F1_RED, fontsize=12, fontweight="heavy", y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    generate_blocker_drilldown(output_path="visuals/v7_blockers.png")
