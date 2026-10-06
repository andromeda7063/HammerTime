"""
Visual V4: DRS-Train Timeline
Per-lap gap tracking, dirty air chain detection (< 0.8 s),
and cumulative 0.4 s - 0.8 s penalty band validation.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from theme import (
    apply_broadcast_theme, CARBON_DARK, CARD_BG, F1_RED, PURE_WHITE,
    TEXT_MUTED, DRS_CYAN, TRAIN_YELLOW, TEAM_COLORS
)


def generate_drs_timeline(
    laps: np.ndarray,
    driver_penalties: dict,  # {driver: array of lap penalties}
    drs_laps_flagged: list,  # laps where DRS train chain < 0.8s
    target_band: tuple = (0.40, 0.80),
    output_path: str = "visuals/v4_drs_timeline.png"
):
    """
    Renders V4 DRS-Train Timeline with highlighted cumulative penalty band.
    """
    apply_broadcast_theme()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)
    ax.set_facecolor(CARD_BG)

    # Shaded physics target band: 0.4 s to 0.8 s per lap
    ax.axhspan(
        target_band[0], target_band[1],
        color=TRAIN_YELLOW, alpha=0.18, zorder=1,
        label=f"Expected DRS Train Penalty Band ({target_band[0]:.2f}s - {target_band[1]:.2f}s / lap)"
    )

    # Highlight DRS train laps
    for start_lap, end_lap in drs_laps_flagged:
        ax.axvspan(start_lap, end_lap, color=F1_RED, alpha=0.12, zorder=1)
        ax.text(
            (start_lap + end_lap) / 2.0, 0.95, "ACTIVE DRS TRAIN",
            color=F1_RED, fontsize=8, fontweight="bold", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=CARBON_DARK, edgecolor=F1_RED, lw=1)
        )

    # Plot driver cumulative lap penalties
    for drv, pens in driver_penalties.items():
        color = TEAM_COLORS.get(drv, PURE_WHITE)
        lw = 2.5 if drv in ["SAI", "LEC", "RUS", "NOR"] else 1.2
        alpha = 1.0 if drv in ["SAI", "LEC", "RUS", "NOR"] else 0.4
        ax.plot(laps, pens, color=color, lw=lw, alpha=alpha, label=drv)

    # Clean air threshold indicator
    ax.axhline(0.05, color="#10B981", linestyle="--", lw=1.5, label="Clean Air Baseline Level (< 0.05s / lap)")

    ax.set_xlabel("Race Lap Number", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative Lap Traffic Penalty (seconds / lap)", fontsize=11, labelpad=8)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(min(laps), max(laps))
    ax.grid(True, linestyle="--", alpha=0.3)

    # Broadcast Header Card
    ax.text(
        0.03, 0.94, "F1 BROADCAST INSIGHTS · DIRTY AIR DYNAMICS",
        transform=ax.transAxes, color=F1_RED, fontsize=11, fontweight="heavy"
    )
    ax.text(
        0.03, 0.89, "DRS-TRAIN PENALTY TIMELINE & AERODYNAMIC WAKE TRACKING",
        transform=ax.transAxes, color=PURE_WHITE, fontsize=14, fontweight="bold"
    )

    leg = ax.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8, ncol=2)
    for text in leg.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    laps = np.arange(1, 51)
    pens = {
        "VER": np.full(50, 0.02) + np.random.normal(0, 0.01, 50),
        "SAI": np.full(50, 0.10),
        "LEC": np.full(50, 0.12),
        "NOR": np.full(50, 0.08),
    }
    # Mid-race DRS train between laps 18 and 36
    pens["SAI"][18:36] = np.random.uniform(0.55, 0.72, 18)
    pens["LEC"][18:36] = np.random.uniform(0.60, 0.78, 18)
    pens["NOR"][18:36] = np.random.uniform(0.48, 0.65, 18)

    generate_drs_timeline(laps, pens, [(18, 36)], output_path="visuals/v4_drs_timeline.png")
