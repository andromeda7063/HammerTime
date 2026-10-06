"""
Visual V1: Track Traffic Map
Circuit outline with cars as colored nodes, 3-letter driver codes,
and directed disturbance edges scaled by RBF weights.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add visuals dir to path
sys.path.append(os.path.dirname(__file__))
from theme import (
    apply_broadcast_theme, CARBON_DARK, CARD_BG, F1_RED, PURE_WHITE,
    TEXT_MUTED, DRS_CYAN, TEAM_COLORS
)


def generate_track_map(
    car_positions: dict,
    edges: list,
    circuit_name: str = "Autodromo Nazionale Monza",
    lap_num: int = 31,
    output_path: str = "visuals/v1_track_map.png"
):
    """
    Renders V1 broadcast track traffic map.
    car_positions: dict {driver_code: (x, y)} in circuit coords
    edges: list of tuples (leader_code, follower_code, rbf_weight, gap_sec)
    """
    apply_broadcast_theme()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor(CARBON_DARK)
    ax.set_facecolor(CARD_BG)

    # Synthetic realistic Monza circuit layout outline
    theta = np.linspace(0, 2 * np.pi, 300)
    # Monza-like oblong perimeter
    cx = 1000 * np.cos(theta) + 200 * np.sin(2 * theta)
    cy = 400 * np.sin(theta) + 100 * np.cos(3 * theta)
    ax.plot(cx, cy, color="#38384A", lw=8, zorder=1, label="Track Surface")
    ax.plot(cx, cy, color="#5A5A6E", lw=2, linestyle="--", zorder=2)

    # Draw directed disturbance edges (leader -> follower)
    for leader, follower, weight, gap in edges:
        if leader in car_positions and follower in car_positions:
            lx, ly = car_positions[leader]
            fx, fy = car_positions[follower]

            # Edge style based on RBF weight
            alpha = float(np.clip(weight, 0.3, 1.0))
            lw = float(np.clip(weight * 5.0, 1.5, 6.0))
            edge_color = F1_RED if gap < 0.8 else "#FF9900"

            # Draw arrow from leader to follower (disturbance flow)
            ax.annotate(
                "",
                xy=(fx, fy), xytext=(lx, ly),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=edge_color,
                    lw=lw,
                    alpha=alpha,
                    mutation_scale=18
                ),
                zorder=3
            )

    # Draw car nodes
    for drv, (x, y) in car_positions.items():
        color = TEAM_COLORS.get(drv, PURE_WHITE)
        # Node circle
        circle = plt.Circle((x, y), radius=45, color=color, ec=PURE_WHITE, lw=1.5, zorder=5)
        ax.add_patch(circle)
        # Driver label
        ax.text(
            x, y, drv,
            color=CARBON_DARK if color in ["#FFFFFF", "#6CD3BF", "#00D2BE", "#B6BABD"] else PURE_WHITE,
            fontsize=8, fontweight="bold", ha="center", va="center", zorder=6
        )

    # Broadcast Header Card
    ax.text(
        0.03, 0.95, "F1 BROADCAST INSIGHTS · TRACK TRAFFIC & DISTURBANCE MAP",
        transform=ax.transAxes, color=F1_RED, fontsize=12, fontweight="heavy"
    )
    ax.text(
        0.03, 0.90, f"{circuit_name.upper()} · LAP {lap_num} · LIVE GRAPH SNAPSHOT",
        transform=ax.transAxes, color=PURE_WHITE, fontsize=16, fontweight="bold"
    )
    ax.text(
        0.03, 0.86, "Edge thickness / opacity = Dynamic RBF disturbance weight e_ij · Threshold Δt ≤ 2.0 s",
        transform=ax.transAxes, color=TEXT_MUTED, fontsize=9
    )

    # Legend indicator
    ax.plot([], [], color=F1_RED, lw=4, label="Severe Dirty Air (gap < 0.8 s)")
    ax.plot([], [], color="#FF9900", lw=3, label="Aerodynamic Disturbance (0.8 s ≤ gap ≤ 2.0 s)")
    leg = ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor="#2D2D3D", fontsize=9)
    for text in leg.get_texts():
        text.set_color(PURE_WHITE)

    ax.axis("off")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, facecolor=CARBON_DARK, edgecolor="none")
    plt.close()
    return str(out_file)


if __name__ == "__main__":
    # Demo generation
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    drivers = ["VER", "PER", "HAM", "RUS", "LEC", "SAI", "NOR", "PIA", "ALO", "STR",
               "GAS", "OCO", "ALB", "SAR", "TSU", "RIC", "BOT", "ZHO", "MAG", "HUL"]
    demo_pos = {
        drv: (1000 * np.cos(theta[i]) + 200 * np.sin(2 * theta[i]),
              400 * np.sin(theta[i]) + 100 * np.cos(3 * theta[i]))
        for i, drv in enumerate(drivers)
    }
    demo_edges = [
        ("VER", "HAM", 0.92, 0.5),
        ("HAM", "LEC", 0.85, 0.7),
        ("LEC", "SAI", 0.78, 0.9),
        ("NOR", "PIA", 0.88, 0.6),
        ("ALO", "GAS", 0.45, 1.4),
        ("ALB", "TSU", 0.35, 1.8),
    ]
    path = generate_track_map(demo_pos, demo_edges, output_path="visuals/v1_track_map.png")
    print(f"Generated V1 Track Map at: {path}")
