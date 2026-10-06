"""
Visual V2: Predicted vs Actual Delay
Scatter plot with 45-degree parity line, annotated R² and MAE.
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
    TEXT_MUTED, DRS_CYAN
)


def generate_pred_vs_actual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mae: float,
    r2: float,
    circuit_name: str = "Monza",
    output_path: str = "visuals/v2_pred_vs_actual.png"
):
    """
    Renders V2 Predicted vs Actual scatter plot with parity line and metrics.
    """
    apply_broadcast_theme()
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)
    ax.set_facecolor(CARD_BG)

    # Parity Line (y = x)
    lims = [
        min(np.min(y_true), np.min(y_pred), -0.5),
        max(np.max(y_true), np.max(y_pred), 3.0)
    ]
    ax.plot(lims, lims, color=TEXT_MUTED, linestyle="--", lw=1.8, label="Ideal Parity (y = x)")

    # Scatter points with alpha
    ax.scatter(
        y_true, y_pred,
        color=DRS_CYAN, alpha=0.6, s=35, edgecolors="none", zorder=3,
        label="Test Mini-Sector Instances"
    )

    # Broadcast Header Card
    ax.text(
        0.04, 0.94, "F1 BROADCAST INSIGHTS · MODEL CALIBRATION",
        transform=ax.transAxes, color=F1_RED, fontsize=11, fontweight="heavy"
    )
    ax.text(
        0.04, 0.89, f"PREDICTED vs ACTUAL TRAFFIC DELAY (Δt_sector)",
        transform=ax.transAxes, color=PURE_WHITE, fontsize=14, fontweight="bold"
    )
    ax.text(
        0.04, 0.85, f"Validation Session: {circuit_name} · Horizon K = 5 snapshots",
        transform=ax.transAxes, color=TEXT_MUTED, fontsize=9
    )

    # Metrics Badge Box
    badge_text = (
        f"MODEL ACCURACY\n"
        f"─────────────────\n"
        f"MAE:  {mae:.3f} s  [Target < 0.15 s]\n"
        f"R²:   {r2:.3f}    [Target 0.68 - 0.82]"
    )
    props = dict(boxstyle="round,pad=0.8", facecolor="#101018", edgecolor=F1_RED, lw=1.5, alpha=0.95)
    ax.text(
        0.62, 0.15, badge_text,
        transform=ax.transAxes, fontsize=10, family="monospace", color=PURE_WHITE,
        verticalalignment="bottom", bbox=props
    )

    ax.set_xlabel("Actual Delay Δt_sector (seconds)", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted Delay ŷ_sector (seconds)", fontsize=11, labelpad=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle="--", alpha=0.3)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.82), facecolor="#101018", edgecolor="#2D2D3D")
    for text in leg.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    np.random.seed(42)
    y_t = np.random.uniform(0.0, 1.8, 400)
    noise = np.random.normal(0, 0.12, 400)
    y_p = y_t * 0.96 + noise
    generate_pred_vs_actual_plot(y_t, y_p, mae=0.118, r2=0.785, output_path="visuals/v2_pred_vs_actual.png")
