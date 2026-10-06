"""
Module M4: Graph Builder
Responsibility: Construct dynamic spatio-temporal graph snapshots for each time step.
Nodes: 20 Formula 1 cars.
Edges: Directed leader j -> follower i when 0 < Delta t_ij <= 2.0 s.
Weight: e_ij = exp(-Delta t_ij^2 / (2 * sigma^2)), sigma = 1.0 s.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.graph")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


COMPOUND_MAP = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4
}


def build_node_feature_vector(row: pd.Series, num_compounds: int = 5) -> np.ndarray:
    """
    Constructs raw feature vector for a single driver:
    [velocity, throttle, compound_one_hot (5), tyre_age, mini_sector] -> dim 9
    """
    speed = float(row.get("Speed", 0.0))
    throttle = float(row.get("Throttle", 0.0))
    tyre_age = float(row.get("TyreLife", 1.0))
    mini_sec = float(row.get("MiniSector", 0.0))

    # One-hot compound
    compound_str = str(row.get("Compound", "MEDIUM")).upper()
    cmp_idx = COMPOUND_MAP.get(compound_str, 1)  # default MEDIUM
    cmp_onehot = np.zeros(num_compounds, dtype=np.float32)
    cmp_onehot[cmp_idx] = 1.0

    feat = np.array([speed, throttle], dtype=np.float32)
    feat = np.concatenate([feat, cmp_onehot, [tyre_age, mini_sec]])
    return feat


def build_snapshot_edges(driver_ahead_dict: dict, gap_dict: dict, driver_to_idx: dict, max_gap: float = 2.0, sigma: float = 1.0):
    """
    Constructs directed edges (leader j -> follower i) if 0 < Delta t_ij <= max_gap.
    Computes Gaussian RBF edge weight: exp(-dt^2 / (2 * sigma^2)).
    """
    src, dst, weights = [], [], []

    for follower_drv, leader_drv in driver_ahead_dict.items():
        if leader_drv is None or follower_drv not in driver_to_idx or leader_drv not in driver_to_idx:
            continue
        dt = gap_dict.get(follower_drv, 99.0)
        if 0.0 < dt <= max_gap:
            j = driver_to_idx[leader_drv]      # Leader index (source)
            i = driver_to_idx[follower_drv]    # Follower index (destination)
            w = np.exp(- (dt ** 2) / (2.0 * (sigma ** 2)))
            src.append(j)
            dst.append(i)
            weights.append(w)

    if len(src) == 0:
        # Zero edge snapshot: maintain tensor shape [2, 0] and [0, 1]
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    return edge_index, edge_attr


def build_race_graph_snapshots(aligned_df: pd.DataFrame, config: dict):
    """
    Constructs the sequence of dynamic graph snapshots G_t from aligned telemetry.
    Returns: list of PyG Data objects with:
      - x: [20, 9] node features
      - edge_index: [2, E] directed leader -> follower edges
      - edge_attr: [E, 1] RBF weights
      - mask: [20] boolean mask (True = active car on track)
      - time_sec: timestamp
    """
    max_gap = config["graph"]["edge_threshold_sec"]
    sigma = config["graph"]["rbf_sigma"]
    num_nodes = config["graph"]["num_nodes"]

    # Fix consistent driver ordering (20 drivers)
    unique_drivers = sorted(aligned_df["Driver"].unique())
    if len(unique_drivers) > num_nodes:
        unique_drivers = unique_drivers[:num_nodes]
    driver_to_idx = {drv: idx for idx, drv in enumerate(unique_drivers)}
    idx_to_driver = {idx: drv for drv, idx in driver_to_idx.items()}

    snapshots = []
    timestamps = sorted(aligned_df["TimeSec"].unique())
    logger.info(f"Building graph snapshots across {len(timestamps)} time steps for {len(unique_drivers)} drivers...")

    # Group by timestamp for efficient vectorization
    grouped = aligned_df.groupby("TimeSec")

    for t_sec in timestamps:
        group = grouped.get_group(t_sec)
        x_mat = np.zeros((num_nodes, 9), dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)

        ahead_map = {}
        gap_map = {}

        for _, row in group.iterrows():
            drv = row["Driver"]
            if drv in driver_to_idx:
                idx = driver_to_idx[drv]
                # Check if active (not retired / not in pit)
                is_active = not bool(row.get("IsPitLap", False))
                mask[idx] = is_active
                if is_active:
                    x_mat[idx] = build_node_feature_vector(row)
                else:
                    # Degraded node: zero features, keep node in graph
                    x_mat[idx] = 0.0

                ahead_map[drv] = row.get("DriverAhead")
                gap_map[drv] = float(row.get("DeltaT", 99.0))

        edge_index, edge_attr = build_snapshot_edges(
            ahead_map, gap_map, driver_to_idx, max_gap=max_gap, sigma=sigma
        )

        data = Data(
            x=torch.tensor(x_mat, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask=torch.tensor(mask, dtype=torch.bool),
            time_sec=float(t_sec)
        )
        snapshots.append(data)

    logger.info(f"Successfully generated {len(snapshots)} graph snapshots.")
    avg_edges = sum(s.edge_index.size(1) for s in snapshots) / max(len(snapshots), 1)
    logger.info(f"Average active edges per snapshot: {avg_edges:.2f}")

    return snapshots, driver_to_idx


if __name__ == "__main__":
    cfg = load_config()
    proc_file = Path(cfg["session"]["processed_dir"]) / "aligned_telemetry.parquet"
    if not proc_file.exists():
        logger.error(f"Missing {proc_file}. Run preprocess.py first.")
        sys.exit(1)
    df = pd.read_parquet(proc_file)
    build_race_graph_snapshots(df, cfg)
