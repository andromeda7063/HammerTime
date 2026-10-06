"""
Visual V3: Traffic Delay Heatmap
Cars x Mini-sectors delay matrix with DRS-train congestion band highlighted.
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
    TEXT_MUTED, DRS_CYAN, TRAIN_YELLOW
)


def generate_delay_heatmap(
    delay_matrix: np.ndarray,
    drivers: list,
    mini_sectors: int = 20,
    drs_band: tuple = (4, 9),  # Mini-sectors where DRS train occurs
    drs_cars: list = [4, 5, 6, 7],
    output_path: str = "visuals/v3_delay_heatmap.png"
):
    """
    Renders V3 Delay Heatmap (Cars x Mini-sectors).
    delay_matrix: shape [len(drivers), mini_sectors]
    """
    apply_broadcast_theme()
    fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)
    ax.set_facecolor(CARD_BG)

    # Colormap: dark blue/purple -> red -> yellow
    im = ax.imshow(
        delay_matrix,
        cmap="inferno",
        aspect="auto",
        vmin=-0.1,
        vmax=0.8,
        origin="upper"
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Predicted Traffic Delay Δt_sector (seconds)", color=PURE_WHITE, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=PURE_WHITE)

    # Highlight DRS Train Box
    if drs_band and drs_cars:
        x_min = drs_band[0] - 0.5
        x_max = drs_band[1] + 0.5
        y_min = min(drs_cars) - 0.5
        y_max = max(drs_cars) + 0.5

        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2.5, edgecolor=F1_RED, facecolor="none",
            linestyle="-", zorder=10
        )
        ax.add_patch(rect)

        # Callout annotation
        ax.text(
            x_min + (x_max - x_min) / 2.0, y_min - 0.7,
            "CRITICAL DRS TRAIN CONGESTION BAND (+0.55s / sector)",
            color=F1_RED, fontsize=9, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#101018", edgecolor=F1_RED, lw=1)
        )

    # Set ticks and labels
    ax.set_xticks(np.arange(mini_sectors))
    ax.set_xticklabels([f"S{i+1}" for i in range(mini_sectors)], fontsize=8)
    ax.set_yticks(np.arange(len(drivers)))
    ax.set_yticklabels(drivers, fontsize=9, fontweight="bold")

    ax.set_xlabel("Mini-Sector Index (Track Discretization M=20)", fontsize=11, labelpad=10)
    ax.set_ylabel("Driver Code", fontsize=11, labelpad=10)

    # Broadcast Header Card
    ax.text(
        0.0, 1.08, "F1 BROADCAST INSIGHTS · SPATIAL CONGESTION HEATMAP",
        transform=ax.transAxes, color=F1_RED, fontsize=11, fontweight="heavy"
    )
    ax.text(
        0.0, 1.03, "CARS × MINI-SECTORS TRAFFIC DELAY MATRIX (LIVE ESTIMATE)",
        transform=ax.transAxes, color=PURE_WHITE, fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    drivers = ["VER", "PER", "HAM", "RUS", "SAI", "LEC", "NOR", "PIA", "ALO", "STR",
               "GAS", "OCO", "ALB", "SAR", "TSU", "RIC", "BOT", "ZHO", "MAG", "HUL"]
    mat = np.random.uniform(0.0, 0.15, (20, 20))
    # Inject DRS train into cars 4-7 in sectors 5-9
    mat[4:8, 5:10] += np.random.uniform(0.35, 0.65, (4, 5))
    generate_delay_heatmap(mat, drivers, output_path="visuals/v3_delay_heatmap.png")
