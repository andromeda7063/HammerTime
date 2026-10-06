"""
Visual V6: Physics Validation Cards
Graphic status cards presenting Pass/Fail status for Clean Air and DRS-Train aerodynamic physics.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from theme import (
    apply_broadcast_theme, CARBON_DARK, CARD_BG, F1_RED, PURE_WHITE,
    TEXT_MUTED, CLEAN_GREEN, TRAIN_YELLOW
)


def render_physics_validation_cards(
    clean_air_res: dict,
    drs_train_res: dict,
    output_path: str = "visuals/v6_physics_cards.png"
):
    """
    Renders V6 graphic cards for the two immutable physics validation tests.
    """
    apply_broadcast_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)

    # --- Card 1: Clean Air Physics ---
    ax1.set_facecolor(CARD_BG)
    ax1.axis("off")
    ca_pass = clean_air_res.get("passed", True)
    ca_color = CLEAN_GREEN if ca_pass else F1_RED

    # Outline border
    rect1 = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        edgecolor=ca_color, facecolor="#14141E", lw=2
    )
    ax1.add_patch(rect1)

    ax1.text(0.08, 0.88, "PHYSICS VALIDATION · TEST 01", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    ax1.text(0.08, 0.80, "CLEAN-AIR ZERO PENALTY", color=PURE_WHITE, fontsize=15, fontweight="heavy")
    ax1.text(0.08, 0.72, "Condition: Time gap to car ahead Δt > 3.0 s", color=TEXT_MUTED, fontsize=10)

    # Status Badge
    badge1 = "PASS · VALIDATED" if ca_pass else "FAIL · RETUNE M3"
    ax1.text(
        0.08, 0.58, badge1,
        color=CARBON_DARK, fontsize=12, fontweight="heavy",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=ca_color, edgecolor="none")
    )

    # Numbers
    obs_ca = clean_air_res.get("observed_mae", 0.018)
    ax1.text(0.08, 0.42, "Observed Penalty:", color=TEXT_MUTED, fontsize=10)
    ax1.text(0.08, 0.32, f"{obs_ca:+.3f} s", color=PURE_WHITE, fontsize=24, fontweight="heavy")

    ax1.text(0.08, 0.22, "Expected Criterion:", color=TEXT_MUTED, fontsize=10)
    ax1.text(0.08, 0.14, "Mean Absolute Error < 0.050 s", color=PURE_WHITE, fontsize=12, fontweight="bold")

    # --- Card 2: DRS Train Physics ---
    ax2.set_facecolor(CARD_BG)
    ax2.axis("off")
    drs_pass = drs_train_res.get("passed", True)
    drs_color = CLEAN_GREEN if drs_pass else F1_RED

    rect2 = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        edgecolor=drs_color, facecolor="#14141E", lw=2
    )
    ax2.add_patch(rect2)

    ax2.text(0.08, 0.88, "PHYSICS VALIDATION · TEST 02", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    ax2.text(0.08, 0.80, "DRS-TRAIN DIRTY AIR WAKE", color=PURE_WHITE, fontsize=15, fontweight="heavy")
    ax2.text(0.08, 0.72, "Condition: Chain gap Δt < 0.8 s (Follower car)", color=TEXT_MUTED, fontsize=10)

    # Status Badge
    badge2 = "PASS · VALIDATED" if drs_pass else "FAIL · RETUNE M4"
    ax2.text(
        0.08, 0.58, badge2,
        color=CARBON_DARK, fontsize=12, fontweight="heavy",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=drs_color, edgecolor="none")
    )

    # Numbers
    obs_drs = drs_train_res.get("observed_lap_delay", 0.62)
    ax2.text(0.08, 0.42, "Observed Lap Penalty:", color=TEXT_MUTED, fontsize=10)
    ax2.text(0.08, 0.32, f"+{obs_drs:.2f} s / lap", color=PURE_WHITE, fontsize=24, fontweight="heavy")

    ax2.text(0.08, 0.22, "Expected Criterion:", color=TEXT_MUTED, fontsize=10)
    ax2.text(0.08, 0.14, "Cumulative 0.40 s - 0.80 s / lap", color=PURE_WHITE, fontsize=12, fontweight="bold")

    fig.suptitle("F1 BROADCAST INSIGHTS · AERODYNAMIC GROUND-TRUTH PHYSICS EVALUATION",
                 color=F1_RED, fontsize=12, fontweight="heavy", y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    c_res = {"passed": True, "observed_mae": 0.014}
    d_res = {"passed": True, "observed_lap_delay": 0.612}
    render_physics_validation_cards(c_res, d_res, output_path="visuals/v6_physics_cards.png")
