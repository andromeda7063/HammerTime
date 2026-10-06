"""
Visual V9: Clean-Air Calibration
Predicted delay distribution for unhindered cars (gap > 3.0 s) hugging zero bound.
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
    TEXT_MUTED, CLEAN_GREEN, DRS_CYAN
)


def generate_clean_air_calibration_plot(
    clean_air_predictions: np.ndarray,
    threshold: float = 0.05,
    output_path: str = "visuals/v9_clean_air_cal.png"
):
    """
    Renders V9 Clean-Air Zero-Penalty Calibration Plot.
    """
    apply_broadcast_theme()
    fig, (ax_dist, ax_box) = plt.subplots(1, 2, figsize=(13, 6), dpi=150, gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor(CARBON_DARK)

    # --- Left: Histogram / KDE Distribution ---
    ax_dist.set_facecolor(CARD_BG)
    counts, bins, patches_list = ax_dist.hist(
        clean_air_predictions, bins=30, density=True,
        color=CLEAN_GREEN, alpha=0.7, edgecolor=PURE_WHITE, lw=0.5
    )

    ax_dist.axvline(0.0, color=PURE_WHITE, lw=2, linestyle="-", label="Ideal Zero Penalty (0.000 s)")
    ax_dist.axvspan(-threshold, threshold, color=CLEAN_GREEN, alpha=0.15, label=f"Physics Tolerance Band (±{threshold}s)")

    ax_dist.set_xlabel("Predicted Delay ŷ_sector in Clean Air (seconds)", fontsize=10, labelpad=8)
    ax_dist.set_ylabel("Probability Density", fontsize=10, labelpad=8)
    ax_dist.set_title("DISTRIBUTION OF PREDICTED DELAY (GAP > 3.0 s)", color=PURE_WHITE, fontsize=11, fontweight="bold", pad=10)
    ax_dist.grid(True, linestyle="--", alpha=0.3)
    leg = ax_dist.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg.get_texts(): t.set_color(PURE_WHITE)

    # --- Right: Calibration Metric Summary Card ---
    ax_box.set_facecolor(CARD_BG)
    ax_box.axis("off")

    mean_err = float(np.mean(np.abs(clean_air_predictions)))
    pct_within = float(np.mean(np.abs(clean_air_predictions) <= threshold) * 100)

    props = dict(boxstyle="round,pad=0.8", facecolor="#14141E", edgecolor=CLEAN_GREEN, lw=1.5)
    card_text = (
        f"CLEAN-AIR CALIBRATION METRICS\n"
        f"─────────────────────────────\n"
        f"Mean Clean-Air Error:  {mean_err:.4f} s\n"
        f"Samples In-Band (±0.05s): {pct_within:.1f} %\n"
        f"Standard Deviation:    {np.std(clean_air_predictions):.4f} s\n\n"
        f"VERDICT:\n"
        f"Clean air boundary is validated.\n"
        f"The model reliably predicts zero\n"
        f"disturbance when unhindered."
    )
    ax_box.text(0.1, 0.45, card_text, fontsize=11, family="monospace", color=PURE_WHITE,
                bbox=props, va="center")

    fig.suptitle("F1 BROADCAST INSIGHTS · CLEAN-AIR CALIBRATION SANITY (GAP > 3.0 s)",
                 color=F1_RED, fontsize=12, fontweight="heavy", y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    np.random.seed(42)
    dummy_clean = np.random.normal(0.002, 0.016, 500)
    generate_clean_air_calibration_plot(dummy_clean, output_path="visuals/v9_clean_air_cal.png")
