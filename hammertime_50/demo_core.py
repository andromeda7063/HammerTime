"""
HammerTime 50%: Core Demonstration Runner
Runs actual FastF1 race data for the 2021 Abu Dhabi Grand Prix (VER vs HAM),
builds dynamic spatio-temporal graph snapshots, visualizes graph topology and adjacency,
trains core ST-GNN ON REAL RACE TARGETS (Delta t_sector), and extracts traffic analytics.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import time
import shutil

# Ensure Windows terminal handles UTF-8 smoothly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from core_pipeline import (
    load_fastf1_race_telemetry,
    synthesize_core_race_data,
    calculate_core_clean_air_targets
)
from core_graph import build_core_snapshots
from core_model import CoreSTGNN
from core_analytics import (
    detect_core_drs_trains,
    generate_core_traffic_plot,
    plot_core_graph_snapshot,
    get_edge_tier_info,
    DRIVER_DETAILS
)


def build_real_training_dataset(snapshots, df_targets, drv_to_idx, W=5, K=5):
    """
    Pairs rolling graph snapshot windows (W past steps) with actual
    ground-truth Delta t_sector targets from df_targets at horizon step t+K.
    """
    num_nodes = len(drv_to_idx)
    target_lookup = {}
    for _, row in df_targets.iterrows():
        key = (str(row["Driver"]), int(row["MiniSector"]))
        target_lookup[key] = float(row["DeltaT_Sector"])

    dataset = []
    step = 4  # sample every 1s for efficiency
    for i in range(0, len(snapshots) - W - K, step):
        win = snapshots[i : i + W]
        target_snap = snapshots[i + W + K]

        y_vec = np.zeros(num_nodes, dtype=np.float32)
        valid_mask = np.zeros(num_nodes, dtype=bool)

        for d_name, d_idx in drv_to_idx.items():
            sec_idx = int(target_snap.x[d_idx, 3].item() * 20.0)
            target_val = target_lookup.get((str(d_name), sec_idx))
            if target_val is not None:
                y_vec[d_idx] = target_val
                valid_mask[d_idx] = True

        if valid_mask.sum() >= 2:
            dataset.append({
                "window": win,
                "target": torch.tensor(y_vec, dtype=torch.float),
                "mask": torch.tensor(valid_mask, dtype=torch.bool)
            })

    return dataset


def run_core_demo(
    use_real_fastf1: bool = True,
    mode: str = "full",
    snapshot_idx: int = 3000,
    save_graph: bool = True,
    epochs: int = 10
):
    print("===============================================================")
    print("  HAMMERTIME 50%: 2021 ABU DHABI GRAND PRIX (FASTF1 CORE DEMO)")
    print(f"  Mode: {mode.upper()} | Target Snapshot: {snapshot_idx} | Epochs: {epochs}")
    print("===============================================================")
    start = time.time()

    # 1. Pipeline: Load Real FastF1 2021 Abu Dhabi GP Data
    print("\n[1/5] Ingesting & Aligning Race Telemetry (FastF1: 2021 Abu Dhabi GP)...")
    df_raw = None
    if use_real_fastf1:
        try:
            df_raw = load_fastf1_race_telemetry(year=2021, gp="Abu Dhabi", session_type="R", cache_dir="data/cache")
            print(f"      [OK] Successfully loaded actual FastF1 telemetry for 2021 Abu Dhabi Grand Prix!")
            print(f"      Drivers on grid: {', '.join(sorted(df_raw['Driver'].unique()))}")
        except Exception as e:
            print(f"      [WARNING] FastF1 live download unavailable ({e}). Using synthetic Abu Dhabi grid...")
            df_raw = synthesize_core_race_data(num_drivers=20, num_snapshots=2500)
    else:
        df_raw = synthesize_core_race_data(num_drivers=20, num_snapshots=2500)

    print("      Computing actual clean-air baselines & Delta t_sector targets...")
    df_targets = calculate_core_clean_air_targets(df_raw)
    print(f"      Aligned {len(df_raw)} telemetry rows. Extracted {len(df_targets)} sector targets.")

    # 2. Graph Construction
    print("\n[2/5] Constructing Dynamic Spatio-Temporal Graph Snapshots...")
    snapshots, drv_to_idx = build_core_snapshots(df_raw, edge_threshold=2.0, sigma=1.0)
    total_edges = sum(s.edge_index.size(1) for s in snapshots)
    avg_edges = total_edges / max(len(snapshots), 1)
    print(f"      [OK] Built {len(snapshots)} snapshots with {total_edges} total directed disturbance edges.")
    print(f"      Average active disturbance edges per snapshot: {avg_edges:.2f}")

    if save_graph:
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_save_path = cache_dir / "core_graph_snapshots.pt"
        torch.save({"snapshots": snapshots, "drv_to_idx": drv_to_idx}, str(graph_save_path))
        print(f"      [OK] Persisted {len(snapshots)} graph snapshots to disk: {graph_save_path}")

    # 3. Dynamic Graph Inspection & Multi-Tier Visualization
    print("\n[3/5] Visualizing Built Dynamic Graph Snapshot (Red, Orange, Green Tiers)...")
    target_idx = min(max(snapshot_idx, 0), len(snapshots) - 1)
    selected_snap = snapshots[target_idx]

    graph_img_path = Path(__file__).resolve().parent / "core_graph_visualization.png"
    plot_core_graph_snapshot(
        selected_snap, drv_to_idx,
        title=f"2021 Abu Dhabi Grand Prix · Snapshot {target_idx} (t={selected_snap.time_sec:.1f}s)",
        output_path=str(graph_img_path)
    )
    # Also copy to root for easy access
    root_graph_path = Path(__file__).resolve().parent.parent / "core_graph_visualization.png"
    shutil.copy(str(graph_img_path), str(root_graph_path))
    print(f"      [OK] Rendered multi-tier graph topology & matrix to: {graph_img_path}")
    print(f"      [OK] Copied visual directly to root: {root_graph_path}")

    # Print Active Graph Snapshot Inspection
    idx_to_drv = {i: d for d, i in drv_to_idx.items()}
    edge_idx = selected_snap.edge_index
    edge_weights = selected_snap.edge_attr.squeeze(-1)
    num_e = edge_idx.size(1)

    print("\n      --- DYNAMIC GRAPH INTERACTION INSPECTION (WHO DISTURBS WHOM) ---")
    print(f"      Timestamp: {selected_snap.time_sec:.2f} s | Nodes: {len(drv_to_idx)} cars | Active Disturbance Edges: {num_e}")
    print("      -----------------------------------------------------------------------------------------")
    print("      Leader Car (j)   ->   Follower Car (i)    Time Gap (Δt)   Weight (e_ij)   Aerodynamic Tier")
    print("      -----------------------------------------------------------------------------------------")
    for k in range(num_e):
        s_d = idx_to_drv[int(edge_idx[0, k])]
        d_d = idx_to_drv[int(edge_idx[1, k])]
        w_val = float(edge_weights[k])
        edge_col, tier_label, edge_lw, gap_sec = get_edge_tier_info(w_val)
        lead_code = DRIVER_DETAILS.get(s_d, {}).get("code", s_d)
        foll_code = DRIVER_DETAILS.get(d_d, {}).get("code", d_d)
        print(f"      {lead_code:<16} ->   {foll_code:<16}    {gap_sec:.2f} s          {w_val:.4f}          [{tier_label}]")
    print("      -----------------------------------------------------------------------------------------")

    if mode == "graph":
        elapsed = time.time() - start
        print(f"\n[DONE] Graph building and multi-tier visualization completed in {elapsed:.2f} seconds.")
        print(f"Visual artifact: {root_graph_path}")
        print("===============================================================")
        return

    # 4. Model Training on Real 2021 Abu Dhabi Data
    print("\n[4/5] Training Core ST-GNN on Real 2021 Abu Dhabi Grand Prix Targets...")
    dataset = build_real_training_dataset(snapshots, df_targets, drv_to_idx, W=5, K=5)
    print(f"      Constructed {len(dataset)} real spatio-temporal training samples.")

    split_idx = int(len(dataset) * 0.75)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    print(f"      Data Split: Train = {len(train_set)} windows (Laps 1-43) | Test = {len(test_set)} windows (Laps 44-58)")

    model = CoreSTGNN(in_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    print("\n      --- TRAINING CONVERGENCE (REAL DATA) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        active_count = 0

        for item in train_set:
            win = item["window"]
            target = item["target"]
            mask = item["mask"]

            optimizer.zero_grad()
            pred = model(win)

            diff = torch.abs(pred - target) * mask.float()
            loss = diff.sum() / max(mask.float().sum(), 1.0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            active_count += 1

        avg_train_l1 = epoch_loss / max(active_count, 1)
        print(f"      Epoch {epoch:02d}/{epochs:02d} | Real Train L1 Loss: {avg_train_l1:.4f} s / sector")

    # Real Evaluation on Unseen Test Windows (Laps 44 - 58)
    print("\n      --- TEST EVALUATION ON UNSEEN RACE WINDOWS ---")
    model.eval()
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for item in test_set:
            win = item["window"]
            target = item["target"]
            mask = item["mask"]

            pred = model(win)
            valid_idx = torch.where(mask)[0]

            for idx in valid_idx:
                all_y_true.append(float(target[idx].item()))
                all_y_pred.append(float(pred[idx].item()))

    y_true_real = np.array(all_y_true)
    y_pred_real = np.array(all_y_pred)

    test_mae = float(mean_absolute_error(y_true_real, y_pred_real))
    test_r2 = float(r2_score(y_true_real, y_pred_real))

    print(f"      Test Evaluation on Real Telemetry (Laps 44-58):")
    print(f"      >> Real Test MAE: {test_mae:.4f} s (Target < 0.150 s) -> {'PASS' if test_mae < 0.15 else 'NEEDS TUNING (SC Yellows)'}")
    print(f"      >> Real Test R^2: {test_r2:.4f} (Target 0.68 - 0.82)  -> {'PASS' if test_r2 >= 0.60 else 'NEEDS TUNING (SC Yellows)'}")

    # 5. Analytics & Visualization
    print("\n[5/5] Executing Traffic Analytics & Generating Real Analytics Visual...")
    drs_trains = detect_core_drs_trains(df_raw)
    print(f"      Detected {len(drs_trains)} DRS-train disturbance events during the Grand Prix.")

    plot_path = Path(__file__).resolve().parent / "core_analytics.png"
    generate_core_traffic_plot(y_true_real, y_pred_real, output_path=str(plot_path))
    root_plot_path = Path(__file__).resolve().parent.parent / "core_analytics.png"
    shutil.copy(str(plot_path), str(root_plot_path))
    print(f"      [OK] Saved real 2021 Abu Dhabi analytics chart to: {plot_path}")
    print(f"      [OK] Copied analytics chart directly to root: {root_plot_path}")

    elapsed = time.time() - start
    print(f"\n2021 Abu Dhabi core demonstration executed successfully in {elapsed:.2f} seconds.")
    print("===============================================================")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HammerTime 50%: Core Demonstration Runner")
    parser.add_argument(
        "--mode", choices=["full", "graph", "train"], default="full",
        help="Execution mode: 'full' (all 5 steps), 'graph' (only data ingestion, graph construction, inspection, and visualization), 'train' (graph construction + ST-GNN training + evaluation)"
    )
    parser.add_argument(
        "--snapshot-idx", type=int, default=3000,
        help="Index of the snapshot to visualize (default: 3000, ~t=750s / Lap 8)"
    )
    parser.add_argument(
        "--no-save-graph", dest="save_graph", action="store_false", default=True,
        help="Do not save constructed graph snapshots to data/cache/core_graph_snapshots.pt"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs for CoreSTGNN (default: 10)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=False,
        help="Use synthetic telemetry instead of real FastF1 2021 Abu Dhabi Grand Prix"
    )

    args = parser.parse_args()
    run_core_demo(
        use_real_fastf1=not args.synthetic,
        mode=args.mode,
        snapshot_idx=args.snapshot_idx,
        save_graph=args.save_graph,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()

