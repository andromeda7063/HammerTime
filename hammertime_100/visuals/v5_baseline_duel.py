"""
Visual V5: Baseline Duel
Comparative benchmark bars: HammerTime ST-GNN vs Single-Car XGBoost vs Single-Car LSTM.
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


def generate_baseline_duel_chart(
    metrics: dict,
    output_path: str = "visuals/v5_baseline_duel.png"
):
    """
    Renders V5 Baseline Duel comparison chart.
    metrics: dict with keys 'HammerTime', 'XGBoost', 'LSTM',
             each containing 'mae' and 'r2'.
    """
    apply_broadcast_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)

    models = ["HammerTime ST-GNN", "Single-Car XGBoost", "Single-Car LSTM"]
    keys = ["HammerTime", "XGBoost", "LSTM"]
    colors = [F1_RED, "#6B7280", "#4B5563"]

    maes = [metrics[k]["mae"] for k in keys]
    r2s = [metrics[k]["r2"] for k in keys]

    # --- Plot 1: MAE (Lower is Better) ---
    ax1.set_facecolor(CARD_BG)
    bars1 = ax1.bar(models, maes, color=colors, width=0.55, edgecolor=PURE_WHITE, lw=0.8)
    ax1.axhline(0.15, color=DRS_CYAN, linestyle="--", lw=1.5, label="F1 Broadcast Spec Target (< 0.15 s)")

    for bar, val in zip(bars1, maes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f} s", ha="center", va="bottom",
            color=PURE_WHITE, fontweight="bold", fontsize=10
        )

    ax1.set_ylabel("Test Mean Absolute Error (seconds) · Lower is Better", fontsize=10, labelpad=8)
    ax1.set_title("PREDICTION ERROR (MAE)", color=PURE_WHITE, fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0, max(maes) * 1.35)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    leg1 = ax1.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg1.get_texts(): t.set_color(PURE_WHITE)

    # --- Plot 2: R² (Higher is Better) ---
    ax2.set_facecolor(CARD_BG)
    bars2 = ax2.bar(models, r2s, color=colors, width=0.55, edgecolor=PURE_WHITE, lw=0.8)
    ax2.axhspan(0.68, 0.82, color=DRS_CYAN, alpha=0.15, label="Target Band (0.68 - 0.82)")

    for bar, val in zip(bars2, r2s):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom",
            color=PURE_WHITE, fontweight="bold", fontsize=10
        )

    ax2.set_ylabel("Coefficient of Determination (R²) · Higher is Better", fontsize=10, labelpad=8)
    ax2.set_title("EXPLAINED VARIANCE (R²)", color=PURE_WHITE, fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    leg2 = ax2.legend(loc="upper right", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8)
    for t in leg2.get_texts(): t.set_color(PURE_WHITE)

    # Broadcast Header Card
    fig.suptitle("F1 BROADCAST INSIGHTS · BASELINE DUEL BENCHMARK (IDENTICAL DATA SPLIT)",
                 color=F1_RED, fontsize=12, fontweight="heavy", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    demo_metrics = {
        "HammerTime": {"mae": 0.118, "r2": 0.764},
        "XGBoost": {"mae": 0.221, "r2": 0.512},
        "LSTM": {"mae": 0.198, "r2": 0.558},
    }
    generate_baseline_duel_chart(demo_metrics, output_path="visuals/v5_baseline_duel.png")
