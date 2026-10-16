"""
Visual V8: Residual Diagnostics Sanity
Model residuals vs tyre age and vs time gap to verify zero structural bias.
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


def generate_residual_sanity_plots(
    residuals: np.ndarray,
    tyre_ages: np.ndarray,
    gaps: np.ndarray,
    output_path: str = "visuals/v8_residuals.png"
):
    """
    Renders V8 dual residual diagnostic scatter plots.
    """
    apply_broadcast_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)

    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    tyre_ages = np.asarray(tyre_ages, dtype=float).reshape(-1)
    gaps = np.asarray(gaps, dtype=float).reshape(-1)
    if not (len(residuals) == len(tyre_ages) == len(gaps)):
        raise ValueError("residuals, tyre_ages, and gaps must have the same length")
    valid_age = np.isfinite(residuals) & np.isfinite(tyre_ages)
    valid_gap = np.isfinite(residuals) & np.isfinite(gaps)
    if not valid_age.any() or not valid_gap.any():
        raise ValueError("At least one finite residual sample is required")

    # --- Plot 1: Residuals vs Tyre Age ---
    ax1.set_facecolor(CARD_BG)
    ax1.scatter(tyre_ages[valid_age], residuals[valid_age], color=DRS_CYAN, alpha=0.5, s=25, edgecolors="none")
    ax1.axhline(0, color=F1_RED, linestyle="--", lw=1.5)

    # Rolling median trendline to prove zero structure
    age_values = tyre_ages[valid_age]
    age_residuals = residuals[valid_age]
    if np.ptp(age_values) > 0:
        bins = np.linspace(np.min(age_values), np.max(age_values), min(10, len(age_values) + 1))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        indices = np.digitize(age_values, bins)
        medians = [np.median(age_residuals[indices == i]) if np.sum(indices == i) else np.nan for i in range(1, len(bins))]
        ax1.plot(bin_centers, medians, color=PURE_WHITE, lw=2, label="Binned median")

    ax1.set_xlabel("Tyre Age (Laps Completed on Set)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Prediction Residual (ŷ - y in seconds)", fontsize=10, labelpad=8)
    ax1.set_title("RESIDUALS vs TYRE DEGRADATION AGE", color=PURE_WHITE, fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylim(-0.6, 0.6)
    ax1.grid(True, linestyle="--", alpha=0.3)
    leg1 = ax1.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg1.get_texts(): t.set_color(PURE_WHITE)

    # --- Plot 2: Residuals vs Time Gap ---
    ax2.set_facecolor(CARD_BG)
    ax2.scatter(gaps[valid_gap], residuals[valid_gap], color="#F59E0B", alpha=0.5, s=25, edgecolors="none")
    ax2.axhline(0, color=F1_RED, linestyle="--", lw=1.5)

    gap_values = gaps[valid_gap]
    gap_residuals = residuals[valid_gap]
    if np.ptp(gap_values) > 0:
        gap_bins = np.linspace(np.min(gap_values), np.max(gap_values), min(12, len(gap_values) + 1))
        gap_centers = 0.5 * (gap_bins[:-1] + gap_bins[1:])
        gap_idx = np.digitize(gap_values, gap_bins)
        gap_medians = [np.median(gap_residuals[gap_idx == i]) if np.sum(gap_idx == i) else np.nan for i in range(1, len(gap_bins))]
        ax2.plot(gap_centers, gap_medians, color=PURE_WHITE, lw=2, label="Binned median")

    ax2.set_xlabel("Dynamic Time Gap Δt (seconds)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Prediction Residual (ŷ - y in seconds)", fontsize=10, labelpad=8)
    ax2.set_title("RESIDUALS vs PROXIMITY TIME GAP", color=PURE_WHITE, fontsize=11, fontweight="bold", pad=10)
    ax2.set_ylim(-0.6, 0.6)
    ax2.grid(True, linestyle="--", alpha=0.3)
    leg2 = ax2.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg2.get_texts(): t.set_color(PURE_WHITE)

    fig.suptitle("F1 BROADCAST INSIGHTS · RESIDUAL SANITY & ERROR HOMOSCEDASTICITY CHECK",
                 color=F1_RED, fontsize=12, fontweight="heavy", y=0.97)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    np.random.seed(42)
    n = 350
    t_age = np.random.uniform(1, 35, n)
    t_gap = np.random.uniform(0.2, 4.5, n)
    res = np.random.normal(0, 0.08, n)
    generate_residual_sanity_plots(res, t_age, t_gap, output_path="visuals/v8_residuals.png")
