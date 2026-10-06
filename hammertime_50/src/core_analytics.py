"""
HammerTime 50%: Core Analytics & Advanced Dynamic Graph Visualizer
Computes core traffic metrics, detects DRS trains, and renders broadcast-grade
dynamic graph topology with explicit Leader -> Follower directional disturbance arrows,
three-tier thresholds (Red, Orange, Green), and driver codes.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Official 2021 Abu Dhabi Driver Mapping & Colors
DRIVER_DETAILS = {
    "33": {"code": "33 VER", "team": "Red Bull", "color": "#3671C6", "name": "Verstappen"},
    "44": {"code": "44 HAM", "team": "Mercedes", "color": "#00D2BE", "name": "Hamilton"},
    "11": {"code": "11 PER", "team": "Red Bull", "color": "#3671C6", "name": "Perez"},
    "77": {"code": "77 BOT", "team": "Mercedes", "color": "#00D2BE", "name": "Bottas"},
    "55": {"code": "55 SAI", "team": "Ferrari", "color": "#F91536", "name": "Sainz"},
    "16": {"code": "16 LEC", "team": "Ferrari", "color": "#F91536", "name": "Leclerc"},
    "4":  {"code": "4 NOR",  "team": "McLaren", "color": "#F58020", "name": "Norris"},
    "3":  {"code": "3 RIC",  "team": "McLaren", "color": "#F58020", "name": "Ricciardo"},
    "14": {"code": "14 ALO", "team": "Alpine", "color": "#2293D1", "name": "Alonso"},
    "31": {"code": "31 OCO", "team": "Alpine", "color": "#2293D1", "name": "Ocon"},
    "10": {"code": "10 GAS", "team": "AlphaTauri", "color": "#5E8FAA", "name": "Gasly"},
    "22": {"code": "22 TSU", "team": "AlphaTauri", "color": "#5E8FAA", "name": "Tsunoda"},
    "5":  {"code": "5 VET",  "team": "Aston Martin", "color": "#358C75", "name": "Vettel"},
    "18": {"code": "18 STR", "team": "Aston Martin", "color": "#358C75", "name": "Stroll"},
    "63": {"code": "63 RUS", "team": "Williams", "color": "#37BEDD", "name": "Russell"},
    "6":  {"code": "6 LAT",  "team": "Williams", "color": "#37BEDD", "name": "Latifi"},
    "7":  {"code": "7 RAI",  "team": "Alfa Romeo", "color": "#C92D4B", "name": "Raikkonen"},
    "99": {"code": "99 GIO", "team": "Alfa Romeo", "color": "#C92D4B", "name": "Giovinazzi"},
    "47": {"code": "47 MSC", "team": "Haas", "color": "#B6BABD", "name": "Schumacher"},
}


def detect_core_drs_trains(df: pd.DataFrame, gap_threshold: float = 0.8, min_chain_len: int = 3) -> list:
    """
    Identifies DRS trains (continuous chains of cars with gap < 0.8 s).
    """
    drs_events = []
    for t, grp in df.groupby("TimeSec"):
        grp_sorted = grp.sort_values("Distance", ascending=False)
        chain = []
        for _, row in grp_sorted.iterrows():
            gap = row.get("DeltaT", 99.0)
            if 0 < gap <= gap_threshold:
                chain.append(row["Driver"])
            else:
                if len(chain) >= min_chain_len:
                    drs_events.append({"time_sec": t, "lap": row["LapNumber"], "cars": list(chain)})
                chain = [row["Driver"]]
        if len(chain) >= min_chain_len:
            drs_events.append({"time_sec": t, "lap": row["LapNumber"], "cars": list(chain)})
    return drs_events


def generate_core_traffic_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = "core_analytics.png"):
    """
    Renders basic scatter comparison and traffic delay distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    fig.patch.set_facecolor("#15151E")
    ax1.set_facecolor("#1F1F2B")
    ax2.set_facecolor("#1F1F2B")

    # Parity scatter
    ax1.scatter(y_true, y_pred, color="#E10600", alpha=0.6, s=28)
    lims = [min(min(y_true), min(y_pred), -0.5), max(max(y_true), max(y_pred), 3.0)]
    ax1.plot(lims, lims, color="#FFFFFF", linestyle="--", lw=1.5, label="Parity (y = x)")
    ax1.set_xlabel("Actual Delay Δt_sector (s)", color="#FFFFFF")
    ax1.set_ylabel("Predicted Delay ŷ_sector (s)", color="#FFFFFF")
    ax1.set_title("Core ST-GNN: Predicted vs Actual Delay", color="#FFFFFF", fontweight="bold")
    ax1.tick_params(colors="#FFFFFF")
    ax1.grid(True, linestyle="--", alpha=0.3)
    leg = ax1.legend(facecolor="#15151E", edgecolor="#2D2D3D")
    for t in leg.get_texts(): t.set_color("#FFFFFF")

    # Distribution
    ax2.hist(y_true, bins=25, color="#3671C6", alpha=0.75, edgecolor="#FFFFFF", lw=0.5)
    ax2.set_xlabel("Traffic Delay Δt_sector (s)", color="#FFFFFF")
    ax2.set_ylabel("Frequency (Sector Windows)", color="#FFFFFF")
    ax2.set_title("Distribution of Traffic Delay (2021 Abu Dhabi)", color="#FFFFFF", fontweight="bold")
    ax2.tick_params(colors="#FFFFFF")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor="#15151E", edgecolor="none")
    plt.close()
    return str(out)


def get_edge_tier_info(weight: float):
    """
    Categorizes edge weight into 3 distinct aerodynamic disturbance tiers:
    - RED: Severe Dirty Air (gap < 0.8s, weight >= 0.726)
    - ORANGE: Moderate Wake (0.8s <= gap < 1.4s, 0.375 <= weight < 0.726)
    - GREEN: Mild Wake (1.4s <= gap <= 2.0s, 0.135 <= weight < 0.375)
    """
    # Inverse RBF gap: dt = sqrt(-2 * sigma^2 * ln(w))
    w_clamped = max(min(weight, 1.0), 0.01)
    gap_sec = float(np.sqrt(-2.0 * np.log(w_clamped)))

    if weight >= 0.726 or gap_sec < 0.80:
        return "#E10600", "SEVERE DIRTY AIR", 4.0, gap_sec
    elif weight >= 0.375 or gap_sec < 1.40:
        return "#FF9900", "MODERATE WAKE", 2.6, gap_sec
    else:
        return "#10B981", "LIGHT WAKE", 1.8, gap_sec


def plot_core_graph_snapshot(
    snapshot,
    drv_to_idx: dict,
    title: str = "2021 Abu Dhabi Grand Prix · Dynamic Spatio-Temporal Graph",
    output_path: str = "core_graph_visualization.png"
):
    """
    Renders the dynamic graph snapshot:
      - Left: Circular Network Topology with explicit directed disturbance arrows (Leader j -> Follower i)
              and three-tier color coding: RED (severe), ORANGE (moderate), GREEN (light).
      - Right: Weighted Disturbance Adjacency Matrix (A_ji = e_ij) and live interaction table.
    """
    idx_to_drv = {i: d for d, i in drv_to_idx.items()}
    num_nodes = len(drv_to_idx)

    fig = plt.figure(figsize=(18, 9), dpi=140)
    fig.patch.set_facecolor("#15151E")

    # Left: Network (2 columns span)
    ax_net = plt.subplot2grid((1, 10), (0, 0), colspan=6)
    ax_net.set_facecolor("#1F1F2B")

    # Right: Adjacency Matrix & Table (4 columns span)
    ax_adj = plt.subplot2grid((1, 10), (0, 6), colspan=4)
    ax_adj.set_facecolor("#1F1F2B")

    # --- 1. NETWORK TOPOLOGY ---
    ax_net.set_title("AERODYNAMIC DISTURBANCE FLOW: LEADER (j) ➔ FOLLOWER (i)",
                     color="#FFFFFF", fontsize=12, fontweight="bold", pad=12)

    # Circular layout ordered by car index
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    # Start at top (12 o'clock) and proceed clockwise
    angles = (np.pi / 2.0) - angles
    radius = 10.0
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    edge_index = snapshot.edge_index
    edge_attr = snapshot.edge_attr
    num_edges = edge_index.size(1)

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_records = []

    red_count = 0
    orange_count = 0
    green_count = 0

    if num_edges > 0:
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        weights = edge_attr.squeeze(-1).cpu().numpy()

        for s_idx, d_idx, w in zip(src, dst, weights):
            adj_matrix[s_idx, d_idx] = w
            lead_drv = idx_to_drv.get(s_idx, str(s_idx))
            foll_drv = idx_to_drv.get(d_idx, str(d_idx))

            lead_info = DRIVER_DETAILS.get(lead_drv, {"code": lead_drv, "color": "#FFFFFF"})
            foll_info = DRIVER_DETAILS.get(foll_drv, {"code": foll_drv, "color": "#FFFFFF"})

            edge_col, tier_label, edge_lw, gap_sec = get_edge_tier_info(w)

            if tier_label == "SEVERE DIRTY AIR":
                red_count += 1
            elif tier_label == "MODERATE WAKE":
                orange_count += 1
            else:
                green_count += 1

            edge_records.append({
                "lead": lead_info["code"],
                "foll": foll_info["code"],
                "w": w,
                "gap": gap_sec,
                "color": edge_col,
                "tier": tier_label
            })

            # Calculate curved arrow or straight arrow from Leader (x_s, y_s) to Follower (x_d, y_d)
            xs, ys = x_coords[s_idx], y_coords[s_idx]
            xd, yd = x_coords[d_idx], y_coords[d_idx]

            # Shorten arrow slightly so it points cleanly at node boundary
            dx, dy = xd - xs, yd - ys
            dist = np.hypot(dx, dy)
            if dist > 0:
                ux, uy = dx / dist, dy / dist
                arrow_start_x = xs + ux * 1.0
                arrow_start_y = ys + uy * 1.0
                arrow_end_x = xd - ux * 1.0
                arrow_end_y = yd - uy * 1.0

                ax_net.annotate(
                    "",
                    xy=(arrow_end_x, arrow_end_y),
                    xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=edge_col,
                        lw=edge_lw,
                        alpha=0.90,
                        mutation_scale=18,
                        connectionstyle="arc3,rad=0.08"
                    ),
                    zorder=2
                )

    # Draw Nodes with team colors and driver labels
    for i in range(num_nodes):
        raw_drv = idx_to_drv.get(i, str(i))
        info = DRIVER_DETAILS.get(raw_drv, {"code": raw_drv, "color": "#FFFFFF", "team": ""})

        # Car circle badge
        circle = plt.Circle(
            (x_coords[i], y_coords[i]), radius=1.05,
            color="#14141E", ec=info["color"], lw=2.5, zorder=4
        )
        ax_net.add_patch(circle)

        # Driver code text inside node
        ax_net.text(
            x_coords[i], y_coords[i], info["code"],
            color="#FFFFFF", fontsize=7.5, fontweight="bold",
            ha="center", va="center", zorder=5
        )

    ax_net.set_xlim(-13.5, 13.5)
    ax_net.set_ylim(-13.5, 13.5)
    ax_net.axis("off")

    # Legend on Network Panel with explicit counts
    ax_net.plot([], [], color="#E10600", lw=4.0, label=f"Severe Dirty Air (gap < 0.8s · e_ij ≥ 0.73) [{red_count} active]")
    ax_net.plot([], [], color="#FF9900", lw=2.8, label=f"Moderate Disturbance (0.8s ≤ gap < 1.4s) [{orange_count} active]")
    ax_net.plot([], [], color="#10B981", lw=2.0, label=f"Light Wake (1.4s ≤ gap ≤ 2.0s) [{green_count} active]")
    ax_net.plot([], [], color="#FFFFFF", lw=0, label="Arrow points: Leader ➔ Follower (Disturbance flow)")

    leg = ax_net.legend(loc="lower left", facecolor="#101018", edgecolor="#2D2D3D", fontsize=8.5)
    for text in leg.get_texts():
        text.set_color("#FFFFFF")

    # --- 2. ADJACENCY MATRIX & INTERACTION LIST ---
    ax_adj.set_facecolor("#1F1F2B")
    ax_adj.set_title("WEIGHTED DISTURBANCE MATRIX A[leader, follower]",
                     color="#FFFFFF", fontsize=11, fontweight="bold", pad=12)

    im = ax_adj.imshow(adj_matrix, cmap="inferno", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax_adj, pad=0.03, shrink=0.75)
    cbar.set_label("Gaussian RBF Disturbance Weight (e_ij)", color="#FFFFFF", fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color="#949498")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#FFFFFF")

    tick_labels = [DRIVER_DETAILS.get(idx_to_drv.get(i, str(i)), {}).get("code", str(i)) for i in range(num_nodes)]
    ax_adj.set_xticks(np.arange(num_nodes))
    ax_adj.set_xticklabels(tick_labels, rotation=90, fontsize=7.5, color="#FFFFFF")
    ax_adj.set_yticks(np.arange(num_nodes))
    ax_adj.set_yticklabels(tick_labels, fontsize=7.5, color="#FFFFFF")
    ax_adj.set_xlabel("Follower Car Receiving Wake (i)", color="#FFFFFF", fontsize=9.5, labelpad=6)
    ax_adj.set_ylabel("Leader Car Ahead (j)", color="#FFFFFF", fontsize=9.5, labelpad=6)
    ax_adj.grid(False)

    # Master Title Header
    fig.suptitle(f"{title.upper()} · {num_edges} ACTIVE DISTURBANCE EDGES (RED: {red_count} · ORANGE: {orange_count} · GREEN: {green_count})",
                 color="#E10600", fontsize=13, fontweight="heavy", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor="#15151E", edgecolor="none")
    plt.close()
    return str(out)
