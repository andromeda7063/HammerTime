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
from context_features import build_node_feature_vector as _build_context_node_feature_vector
from context_features import feature_names, resolve_feature_schema
from physical_graph import (
    EDGE_FEATURES_BY_SCHEMA,
    PhysicalEdgeSet,
    build_physical_circuit_edges,
    is_position_valid,
)

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


def build_node_feature_vector(
    row: pd.Series,
    num_compounds: int = 5,
    compound_categories: list[str] | None = None,
    feature_schema: str = "legacy_v1",
) -> np.ndarray:
    """Build a configured node vector while preserving the legacy public API."""
    return _build_context_node_feature_vector(
        row,
        num_compounds=num_compounds,
        compound_categories=compound_categories,
        feature_schema=feature_schema,
    )


def build_snapshot_edges(driver_ahead_dict: dict, gap_dict: dict, driver_to_idx: dict, max_gap: float = 2.0, sigma: float = 1.0):
    """
    Constructs directed edges (leader j -> follower i) if 0 < Delta t_ij <= max_gap.
    Computes Gaussian RBF edge weight: exp(-dt^2 / (2 * sigma^2)).
    """
    if max_gap <= 0 or sigma <= 0:
        raise ValueError("graph.edge_threshold_sec and graph.rbf_sigma must be positive.")
    src, dst, weights = [], [], []

    for follower_drv, leader_drv in driver_ahead_dict.items():
        if leader_drv is None or pd.isna(leader_drv) or follower_drv not in driver_to_idx or leader_drv not in driver_to_idx:
            continue
        try:
            dt = float(gap_dict.get(follower_drv, 99.0))
        except (TypeError, ValueError):
            continue
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
      - edge_attr: [E, 1] RBF weights by default (or [E, 4] in physical_v1)
      - mask: [20] boolean mask (True = active car on track)
      - position_valid: [20] position support, independent of target/node mask
      - time_sec: timestamp
    """
    # Pandas copies DataFrame.attrs into every row Series returned by iterrows.
    # Coverage/provenance attrs can be large, so create a no-copy local frame
    # without metadata rather than paying that deepcopy once per telemetry row.
    # The caller's DataFrame and its attrs remain untouched.
    if aligned_df.attrs:
        aligned_df = pd.DataFrame(aligned_df, copy=False)
    max_gap = config["graph"]["edge_threshold_sec"]
    sigma = config["graph"]["rbf_sigma"]
    num_nodes = config["graph"]["num_nodes"]
    compound_categories = config["graph"].get("compound_categories", list(COMPOUND_MAP))
    node_feature_schema = resolve_feature_schema(config)
    neighbor_mode = config["graph"].get("neighbor_mode", "race_progress")
    edge_feature_schema = config["graph"].get("edge_feature_schema", "scalar_rbf")
    if neighbor_mode not in ("race_progress", "physical_circuit"):
        raise ValueError("graph.neighbor_mode must be race_progress or physical_circuit.")
    if edge_feature_schema not in EDGE_FEATURES_BY_SCHEMA:
        raise ValueError("graph.edge_feature_schema must be scalar_rbf or physical_v1.")
    if edge_feature_schema == "physical_v1" and neighbor_mode != "physical_circuit":
        raise ValueError("graph.edge_feature_schema=physical_v1 requires physical_circuit neighbors.")
    if num_nodes <= 0:
        raise ValueError("graph.num_nodes must be positive.")
    node_dim = len(feature_names(node_feature_schema, compound_categories))
    expected_dim = config.get("model", {}).get("in_dim")
    if expected_dim is not None and expected_dim != node_dim:
        raise ValueError(
            f"Configured model.in_dim ({expected_dim}) does not match graph feature width ({node_dim})."
        )
    expected_edge_dim = len(EDGE_FEATURES_BY_SCHEMA[edge_feature_schema])
    configured_edge_dim = config.get("model", {}).get("edge_dim", 1)
    if configured_edge_dim != expected_edge_dim:
        raise ValueError(
            f"Configured model.edge_dim ({configured_edge_dim}) does not match "
            f"graph.edge_feature_schema={edge_feature_schema!r} width ({expected_edge_dim})."
        )
    session_cfg = config.get("session", {})
    default_race_id = str(session_cfg.get("race_id") or (
        f"{session_cfg.get('year', 'unknown')}-{session_cfg.get('gp', 'unknown')}-"
        f"{session_cfg.get('session_type', 'R')}"
    ))

    # Fix consistent driver ordering (20 drivers)
    unique_drivers = sorted(aligned_df["Driver"].unique())
    if len(unique_drivers) > num_nodes:
        raise ValueError(
            f"graph.num_nodes ({num_nodes}) cannot hold {len(unique_drivers)} observed drivers."
        )
    if aligned_df.duplicated(["TimeSec", "Driver"]).any():
        raise ValueError("Duplicate driver rows at the same graph timestamp are not supported.")
    driver_to_idx = {drv: idx for idx, drv in enumerate(unique_drivers)}

    lap_lengths_by_race = _configured_lap_lengths(config)

    snapshots = []
    # GroupBy iteration is sorted by timestamp and yields the same source rows
    # as get_group(), without a separate timestamp list and repeated lookups.
    grouped = aligned_df.groupby("TimeSec", sort=True)
    logger.info(f"Building graph snapshots across {len(grouped)} time steps for {len(unique_drivers)} drivers...")

    for t_sec, group in grouped:
        x_mat = np.zeros((num_nodes, node_dim), dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)
        position_valid = np.zeros(num_nodes, dtype=bool)
        # Retain aligned, unscaled speed only as future-speed supervision
        # metadata. It is never included in the node feature tensor.
        aligned_speed_kph = np.full(num_nodes, np.nan, dtype=np.float32)
        aligned_speed_valid = np.zeros(num_nodes, dtype=bool)
        driver_names = [None] * num_nodes

        ahead_map = {}
        gap_map = {}
        distance_by_driver = {}
        speed_by_driver_kmh = {}
        position_valid_by_driver = {}
        node_active_by_driver = {}
        pit_by_driver = {}

        for _, row in group.iterrows():
            drv = row["Driver"]
            if drv in driver_to_idx:
                idx = driver_to_idx[drv]
                driver_names[idx] = str(drv)
                # Check if active (not retired / not in pit)
                is_valid = bool(row.get("IsValid", True)) if pd.notna(row.get("IsValid", True)) else False
                is_pit = bool(row.get("IsPitLap", False)) if pd.notna(row.get("IsPitLap", False)) else False
                is_active = is_valid and not is_pit
                mask[idx] = is_active
                try:
                    speed_kph = float(row.get("Speed"))
                except (TypeError, ValueError):
                    speed_kph = np.nan
                aligned_speed_kph[idx] = speed_kph
                aligned_speed_valid[idx] = bool(is_active and np.isfinite(speed_kph))
                position_ok = is_position_valid(
                    row.get("Distance"), row.get("Speed"), row.get("LapNumber"), is_pit
                )
                position_valid[idx] = position_ok
                position_valid_by_driver[drv] = position_ok
                node_active_by_driver[drv] = is_active
                pit_by_driver[drv] = is_pit
                if position_ok:
                    distance_by_driver[drv] = float(row.get("Distance"))
                    speed_by_driver_kmh[drv] = float(row.get("Speed"))
                if is_active:
                    x_mat[idx] = build_node_feature_vector(
                        row,
                        num_compounds=len(compound_categories),
                        compound_categories=compound_categories,
                        feature_schema=node_feature_schema,
                    )
                else:
                    # Degraded node: zero features, keep node in graph
                    x_mat[idx] = 0.0

                if is_active:
                    ahead_map[drv] = row.get("DriverAhead")
                    gap_value = row.get("DeltaT", 99.0)
                    try:
                        gap_map[drv] = float(gap_value)
                    except (TypeError, ValueError):
                        gap_map[drv] = np.nan

        active_driver_to_idx = {driver: idx for driver, idx in driver_to_idx.items() if mask[idx]}
        edge_index_race_progress, edge_attr_race_progress = build_snapshot_edges(
            ahead_map, gap_map, active_driver_to_idx, max_gap=max_gap, sigma=sigma
        )

        race_id = group["RaceId"].iloc[0] if "RaceId" in group.columns else default_race_id
        if "RaceId" in group.columns and group["RaceId"].nunique(dropna=True) > 1:
            raise ValueError(f"Timestamp {t_sec} contains rows from multiple races. Build snapshots per race.")
        race_id = str(race_id)
        expected_on_track = sum(
            not pit_by_driver.get(driver, False) for driver in driver_to_idx
        )
        lap_length_m = _row_lap_length(group, lap_lengths_by_race, race_id)
        physical_graph_available = lap_length_m is not None
        if physical_graph_available:
            physical_edges = build_physical_circuit_edges(
                distance_by_driver=distance_by_driver,
                speed_by_driver_kmh=speed_by_driver_kmh,
                position_valid_by_driver=position_valid_by_driver,
                node_active_by_driver=node_active_by_driver,
                driver_to_idx=driver_to_idx,
                lap_length_m=lap_length_m,
                max_gap_sec=max_gap,
                sigma_sec=sigma,
                expected_on_track_count=expected_on_track,
                edge_schema=edge_feature_schema,
            )
        elif neighbor_mode == "physical_circuit" or edge_feature_schema in ("physical_v1", "physical_v2"):
            raise ValueError(
                f"Physical graph mode requires a configured lap length for race {race_id!r}."
            )
        else:
            # Minimal legacy configs may not contain circuit metadata. Keep
            # their active race-progress graph intact and report unavailable
            # physical topology without fabricating a lap length or edges.
            known_positions = sum(position_valid_by_driver.values())
            coverage = (known_positions / expected_on_track) if expected_on_track else 0.0
            fallback_dim = 5 if edge_feature_schema == "physical_v2" else 4
            physical_edges = PhysicalEdgeSet(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, fallback_dim), dtype=torch.float32),
                position_coverage=float(coverage),
                blocked_neighbor_count=0,
            )
        edge_index_physical_circuit = physical_edges.edge_index
        edge_attr_physical_circuit = physical_edges.edge_attr
        if neighbor_mode == "race_progress":
            active_edge_index = edge_index_race_progress
            active_edge_attr = edge_attr_race_progress
        else:
            active_edge_index = edge_index_physical_circuit
            active_edge_attr = (
                edge_attr_physical_circuit[:, :1]
                if edge_feature_schema == "scalar_rbf" else edge_attr_physical_circuit
            )
        if edge_feature_schema in ("physical_v1", "physical_v2"):
            active_edge_index = edge_index_physical_circuit
            active_edge_attr = edge_attr_physical_circuit
        is_safety_car = (
            group["IsSafetyCar"].fillna(False).astype(bool).any()
            if "IsSafetyCar" in group.columns else False
        )
        data = Data(
            x=torch.tensor(x_mat, dtype=torch.float),
            edge_index=active_edge_index,
            edge_attr=active_edge_attr,
            edge_index_race_progress=edge_index_race_progress,
            edge_attr_race_progress=edge_attr_race_progress,
            edge_index_physical_circuit=edge_index_physical_circuit,
            edge_attr_physical_circuit=edge_attr_physical_circuit,
            mask=torch.tensor(mask, dtype=torch.bool),
            position_valid=torch.tensor(position_valid, dtype=torch.bool),
            aligned_speed_kph=torch.tensor(aligned_speed_kph, dtype=torch.float32),
            aligned_speed_valid=torch.tensor(aligned_speed_valid, dtype=torch.bool),
            position_coverage=float(physical_edges.position_coverage),
            blocked_neighbor_count=int(physical_edges.blocked_neighbor_count),
            physical_graph_available=bool(physical_graph_available),
            time_sec=float(t_sec),
            race_id=race_id,
            driver_names=tuple(driver_names),
            is_safety_car=bool(is_safety_car),
        )
        snapshots.append(data)

    logger.info(f"Successfully generated {len(snapshots)} graph snapshots.")
    avg_edges = sum(s.edge_index.size(1) for s in snapshots) / max(len(snapshots), 1)
    logger.info(f"Average active edges per snapshot: {avg_edges:.2f}")

    return snapshots, driver_to_idx


def _configured_lap_lengths(config: dict) -> dict[str, float]:
    """Return each configured race's physical circuit length."""
    base = config.get("session", {})
    sessions = config.get("sessions") or ([base] if base else [])
    lengths = {}
    for item in sessions:
        spec = {**base, **item}
        race_id = str(spec.get("race_id") or (
            f"{spec.get('year', 'unknown')}-{spec.get('gp', 'unknown')}-"
            f"{spec.get('session_type', 'R')}"
        ))
        value = spec.get("lap_length_m", config.get("track", {}).get("lap_length_m"))
        try:
            length = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(length) and length > 0:
            lengths[race_id] = length
    return lengths


def _row_lap_length(group: pd.DataFrame, lengths: dict[str, float], race_id: str) -> float | None:
    if "lap_length_m" in group.columns:
        values = pd.to_numeric(group["lap_length_m"], errors="coerce").dropna().unique()
        if len(values) == 1 and float(values[0]) > 0:
            return float(values[0])
        if len(values) > 1:
            raise ValueError(f"Race {race_id!r} has conflicting lap_length_m values at one timestamp.")
    return lengths.get(race_id)


def remap_cohort_driver_nodes(snapshots: list, session_driver_maps: dict,
                              num_nodes: int) -> dict:
    """Apply one stable node index to each distinct driver across race sessions."""
    drivers = sorted({str(driver) for mapping in session_driver_maps.values()
                      for driver in mapping})
    if len(drivers) > num_nodes:
        raise ValueError(
            f"Cohort has {len(drivers)} distinct drivers, exceeding graph.num_nodes={num_nodes}."
        )
    canonical = {driver: index for index, driver in enumerate(drivers)}
    for snapshot in snapshots:
        race_id = str(getattr(snapshot, "race_id", ""))
        if race_id not in session_driver_maps:
            raise ValueError(f"Snapshot has no configured driver map for race {race_id!r}.")
        local = session_driver_maps[race_id]
        old_to_new = {old_index: canonical[str(driver)]
                      for driver, old_index in local.items()}
        features = snapshot.x.clone()
        mask = snapshot.mask.clone()
        position_valid = getattr(snapshot, "position_valid", None)
        if position_valid is not None:
            position_valid = position_valid.clone()
        aligned_speed_kph = getattr(snapshot, "aligned_speed_kph", None)
        aligned_speed_valid = getattr(snapshot, "aligned_speed_valid", None)
        if aligned_speed_kph is not None:
            aligned_speed_kph = aligned_speed_kph.clone()
        if aligned_speed_valid is not None:
            aligned_speed_valid = aligned_speed_valid.clone()
        names = list(snapshot.driver_names)
        remapped_features = torch.zeros_like(features)
        remapped_mask = torch.zeros_like(mask)
        remapped_position_valid = (torch.zeros(num_nodes, dtype=torch.bool, device=mask.device)
                                   if position_valid is not None else None)
        remapped_speed = (torch.full((num_nodes,), float("nan"), dtype=torch.float32,
                                     device=mask.device) if aligned_speed_kph is not None else None)
        remapped_speed_valid = (torch.zeros(num_nodes, dtype=torch.bool, device=mask.device)
                                if aligned_speed_valid is not None else None)
        remapped_names = [None] * num_nodes
        for old_index, driver in enumerate(names):
            if driver is None:
                continue
            if old_index not in old_to_new:
                raise ValueError(f"Snapshot driver {driver!r} is absent from its session node map.")
            new_index = old_to_new[old_index]
            remapped_features[new_index] = features[old_index]
            remapped_mask[new_index] = mask[old_index]
            if remapped_position_valid is not None:
                remapped_position_valid[new_index] = position_valid[old_index]
            if remapped_speed is not None:
                remapped_speed[new_index] = aligned_speed_kph[old_index]
            if remapped_speed_valid is not None:
                remapped_speed_valid[new_index] = aligned_speed_valid[old_index]
            remapped_names[new_index] = str(driver)

        def remap_edges(edge_index):
            edges = edge_index.clone()
            if not edges.numel():
                return edges
            remapped_edges = []
            for node in edges.detach().cpu().reshape(-1).tolist():
                if node not in old_to_new:
                    raise ValueError(f"Snapshot edge references unmapped node {node} for race {race_id!r}.")
                remapped_edges.append(old_to_new[node])
            return torch.tensor(remapped_edges, dtype=torch.long, device=edges.device).reshape(2, -1)

        snapshot.x = remapped_features
        snapshot.mask = remapped_mask
        if remapped_position_valid is not None:
            snapshot.position_valid = remapped_position_valid
        if remapped_speed is not None:
            snapshot.aligned_speed_kph = remapped_speed
        if remapped_speed_valid is not None:
            snapshot.aligned_speed_valid = remapped_speed_valid
        snapshot.driver_names = remapped_names
        for field in ("edge_index", "edge_index_race_progress", "edge_index_physical_circuit"):
            if hasattr(snapshot, field):
                setattr(snapshot, field, remap_edges(getattr(snapshot, field)))
    return canonical


if __name__ == "__main__":
    cfg = load_config()
    proc_file = Path(cfg["session"]["processed_dir"]) / "aligned_telemetry.parquet"
    if not proc_file.exists():
        logger.error(f"Missing {proc_file}. Run preprocess.py first.")
        sys.exit(1)
    df = pd.read_parquet(proc_file)
    build_race_graph_snapshots(df, cfg)
