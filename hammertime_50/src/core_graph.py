"""
HammerTime 50%: Core Graph Builder
Builds dynamic snapshot graphs with Gaussian RBF edge weights (e_ij = exp(-dt^2 / 2*sigma^2))
and normalized node features.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.core_graph")


class CoreGraphSnapshot:
    """Lightweight graph container for core implementation."""
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, time_sec: float):
        self.x = x                    # [N, in_dim]
        self.edge_index = edge_index  # [2, E]
        self.edge_attr = edge_attr    # [E, 1]
        self.time_sec = time_sec


def build_core_snapshots(df: pd.DataFrame, edge_threshold: float = 2.0, sigma: float = 1.0, num_nodes: int = 20):
    """
    Builds dynamic graph snapshot list for core demonstration.
    Normalizes node features: Speed/300.0, Throttle/100.0, TyreLife/50.0, MiniSector/20.0.
    """
    drivers = sorted(df["Driver"].unique())[:num_nodes]
    actual_num_nodes = len(drivers)
    drv_to_idx = {d: i for i, d in enumerate(drivers)}
    timestamps = sorted(df["TimeSec"].unique())

    snapshots = []
    grouped = df.groupby("TimeSec")

    for t in timestamps:
        grp = grouped.get_group(t)
        x_mat = np.zeros((actual_num_nodes, 4), dtype=np.float32)
        src, dst, weights = [], [], []

        for _, row in grp.iterrows():
            d = row["Driver"]
            if d in drv_to_idx:
                idx = drv_to_idx[d]
                # Normalized features for fast, stable ST-GNN training
                spd_norm = float(row["Speed"]) / 320.0
                thr_norm = float(row["Throttle"]) / 100.0
                tyr_norm = float(row["TyreLife"]) / 55.0
                sec_norm = float(row["MiniSector"]) / 20.0
                x_mat[idx] = [spd_norm, thr_norm, tyr_norm, sec_norm]

                ahead = row.get("DriverAhead")
                gap = float(row.get("DeltaT", 99.0))
                if ahead in drv_to_idx and 0.0 < gap <= edge_threshold:
                    leader_idx = drv_to_idx[ahead]
                    w = np.exp(- (gap ** 2) / (2.0 * (sigma ** 2)))
                    src.append(leader_idx)
                    dst.append(idx)
                    weights.append(w)

        if len(src) == 0:
            edge_idx = torch.empty((2, 0), dtype=torch.long)
            edge_w = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_idx = torch.tensor([src, dst], dtype=torch.long)
            edge_w = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

        snap = CoreGraphSnapshot(
            x=torch.tensor(x_mat, dtype=torch.float),
            edge_index=edge_idx,
            edge_attr=edge_w,
            time_sec=t
        )
        snapshots.append(snap)

    logger.info(f"Built {len(snapshots)} core graph snapshots for {actual_num_nodes} drivers.")
    return snapshots, drv_to_idx
