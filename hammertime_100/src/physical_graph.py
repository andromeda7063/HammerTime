"""Physical-circuit neighbor and edge-feature helpers.

The default race-progress graph remains in :mod:`graph`.  This module holds
the opt-in circular-track definition and the train-only scaler contract for
its wider edge schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch


EDGE_FEATURES_BY_SCHEMA = {
    "scalar_rbf": ("rbf_gap",),
    "physical_v1": (
        "rbf_gap",
        "gap_m",
        "leader_minus_follower_speed_mps",
        "position_coverage",
    ),
    "physical_v2": (
        "rbf_gap",
        "gap_m",
        "leader_minus_follower_speed_mps",
        "position_coverage",
        "in_wake_corridor",
    ),
}
PHYSICAL_EDGE_CONTINUOUS_COLUMNS = (1, 2)


@dataclass(frozen=True)
class PhysicalEdgeSet:
    """Edges and observability diagnostics for one timestamp."""

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    position_coverage: float
    blocked_neighbor_count: int


def is_position_valid(distance_m, speed_kmh, lap_number, is_pit=False) -> bool:
    """Return whether a row can locate a car on the circuit.

    This intentionally ignores the general node/target ``IsValid`` flag:
    throttle, compound, or tyre metadata can be missing while distance and
    speed still provide a usable physical position.
    """
    if bool(is_pit):
        return False
    # LapNumber is deliberately not required: circular Distance supplies the
    # physical ordering even when a car is lapped or crosses the start line.
    try:
        distance = float(distance_m)
        speed = float(speed_kmh)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(distance) and distance >= 0.0
            and math.isfinite(speed) and speed >= 0.0)


def build_physical_circuit_edges(
    distance_by_driver: Mapping[str, float],
    speed_by_driver_kmh: Mapping[str, float],
    position_valid_by_driver: Mapping[str, bool],
    node_active_by_driver: Mapping[str, bool],
    driver_to_idx: Mapping[str, int],
    lap_length_m: float,
    max_gap_sec: float = 2.0,
    sigma_sec: float = 1.0,
    expected_on_track_count: int | None = None,
    edge_schema: str = "physical_v1",
) -> PhysicalEdgeSet:
    """Connect each car to the next observed car around the circular track.

    Positions are sorted by ``Distance % lap_length_m`` in driving direction,
    without using lap number, so a lapped car can be physically ahead.  The
    nearest position-supported car is chosen before checking node-feature
    activity.  If that selected car has no usable model node, the edge is
    omitted and counted; the search does not bridge to a farther car.

    The raw edge channels are ``[RBF, gap_m, leader_minus_follower_speed_mps,
    position_coverage]`` (and optionally ``in_wake_corridor`` for physical_v2).
    Only ``gap_m`` and relative speed are normalized later, using edges from
    training input snapshots only.
    """
    if (not math.isfinite(float(lap_length_m)) or lap_length_m <= 0
            or not math.isfinite(float(max_gap_sec)) or max_gap_sec <= 0
            or not math.isfinite(float(sigma_sec)) or sigma_sec <= 0):
        raise ValueError("lap length, edge cutoff, and RBF sigma must be finite and positive.")

    eligible = [
        str(driver) for driver in driver_to_idx
        if position_valid_by_driver.get(driver, False)
        and driver in distance_by_driver and driver in speed_by_driver_kmh
        and math.isfinite(float(distance_by_driver[driver]))
        and math.isfinite(float(speed_by_driver_kmh[driver]))
    ]
    expected_count = (len(driver_to_idx) if expected_on_track_count is None
                      else max(0, int(expected_on_track_count)))
    coverage = float(len(eligible) / expected_count) if expected_count else 0.0
    ordered = sorted(
        eligible,
        key=lambda driver: (
            float(distance_by_driver[driver]) % float(lap_length_m), driver
        ),
    )

    src: list[int] = []
    dst: list[int] = []
    attrs: list[list[float]] = []
    blocked = 0
    if len(ordered) > 1:
        for index, follower in enumerate(ordered):
            leader = ordered[(index + 1) % len(ordered)]
            follower_distance = float(distance_by_driver[follower]) % lap_length_m
            leader_distance = float(distance_by_driver[leader]) % lap_length_m
            gap_m = (leader_distance - follower_distance) % lap_length_m
            if not math.isfinite(gap_m) or gap_m <= 0.0:
                continue
            follower_speed = float(speed_by_driver_kmh[follower]) / 3.6
            leader_speed = float(speed_by_driver_kmh[leader]) / 3.6
            speed_floor = max(follower_speed, 10.0)
            gap_sec = gap_m / speed_floor
            if not 0.0 < gap_sec <= max_gap_sec:
                continue

            # Do not bridge around a car whose position is known but whose
            # model node is unavailable. The incomplete adjacency is recorded.
            if (not node_active_by_driver.get(follower, False)
                    or not node_active_by_driver.get(leader, False)):
                blocked += 1
                continue

            src.append(int(driver_to_idx[leader]))
            dst.append(int(driver_to_idx[follower]))
            attr = [
                math.exp(-(gap_sec ** 2) / (2.0 * sigma_sec ** 2)),
                gap_m,
                leader_speed - follower_speed,
                coverage,
            ]
            if edge_schema == "physical_v2":
                attr.append(1.0 if (gap_m <= 35.0 and gap_sec <= 0.6) else 0.0)
            attrs.append(attr)

    edge_dim = len(EDGE_FEATURES_BY_SCHEMA.get(edge_schema, EDGE_FEATURES_BY_SCHEMA["physical_v1"]))
    edge_index = (torch.tensor([src, dst], dtype=torch.long) if src
                  else torch.empty((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(attrs, dtype=torch.float32) if attrs
                 else torch.empty((0, edge_dim),
                                  dtype=torch.float32))
    return PhysicalEdgeSet(edge_index, edge_attr, coverage, blocked)


class PhysicalEdgeFeatureScaler:
    """Train-only z-scaler for physical edge distance and relative speed.

    The input edges are raw ``physical_v1`` attrs stored on graph snapshots.
    Call :meth:`fit` with unique snapshot indices touched by training inputs;
    the RBF and position coverage channels are passed through unchanged.
    """

    def __init__(self) -> None:
        self.mean_ = np.zeros(2, dtype=np.float64)
        self.var_ = np.zeros(2, dtype=np.float64)
        self.scale_ = np.ones(2, dtype=np.float64)
        self.n_samples_seen_ = 0
        self.fitted_ = False

    def fit(self, snapshots: Sequence, train_snapshot_indices: Sequence[int]):
        blocks = []
        schema_seen = "physical_v1"
        for index in sorted(set(int(i) for i in train_snapshot_indices)):
            attrs = getattr(snapshots[index], "edge_attr_physical_circuit", None)
            if attrs is None or attrs.numel() == 0:
                continue
            if attrs.ndim != 2 or attrs.shape[1] not in (4, 5):
                raise ValueError("physical_circuit edge attrs must have shape [E, 4] or [E, 5].")
            if attrs.shape[1] == 5:
                schema_seen = "physical_v2"
            values = attrs[:, list(PHYSICAL_EDGE_CONTINUOUS_COLUMNS)].detach().cpu().numpy()
            valid = np.isfinite(values).all(axis=1)
            if valid.any():
                blocks.append(values[valid])
        self.schema_ = schema_seen
        if blocks:
            values = np.concatenate(blocks, axis=0).astype(np.float64, copy=False)
            self.mean_ = values.mean(axis=0)
            self.var_ = values.var(axis=0)
            self.scale_ = np.sqrt(self.var_)
            self.scale_[self.scale_ == 0.0] = 1.0
            self.n_samples_seen_ = int(values.shape[0])
        else:
            # No physical edges in the training inputs: keep the continuous
            # channels finite and unchanged rather than dropping samples.
            self.mean_ = np.zeros(2, dtype=np.float64)
            self.var_ = np.zeros(2, dtype=np.float64)
            self.scale_ = np.ones(2, dtype=np.float64)
            self.n_samples_seen_ = 0
        self.fitted_ = True
        return self

    def transform(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if not self.fitted_:
            raise ValueError("Physical edge scaler must be fitted on training edges first.")
        if edge_attr.ndim != 2 or edge_attr.shape[1] not in (4, 5):
            raise ValueError("physical edge attrs must have shape [E, 4] or [E, 5].")
        transformed = edge_attr.clone()
        if transformed.numel():
            mean = torch.as_tensor(self.mean_, dtype=transformed.dtype, device=transformed.device)
            scale = torch.as_tensor(self.scale_, dtype=transformed.dtype, device=transformed.device)
            columns = list(PHYSICAL_EDGE_CONTINUOUS_COLUMNS)
            transformed[:, columns] = (transformed[:, columns] - mean) / scale
        return transformed

    def metadata(self) -> dict:
        if not self.fitted_:
            raise ValueError("Physical edge scaler metadata is unavailable before fitting.")
        schema_name = getattr(self, "schema_", "physical_v1")
        return {
            "schema": schema_name,
            "continuous_features": [EDGE_FEATURES_BY_SCHEMA["physical_v1"][i]
                                    for i in PHYSICAL_EDGE_CONTINUOUS_COLUMNS],
            "mean_": self.mean_.tolist(),
            "var_": self.var_.tolist(),
            "scale_": self.scale_.tolist(),
            "n_samples_seen_": int(self.n_samples_seen_),
            "fit_scope": "unique snapshots touched by training inputs only",
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "PhysicalEdgeFeatureScaler":
        if metadata.get("schema") not in ("physical_v1", "physical_v2"):
            raise ValueError("Unsupported physical edge scaler metadata schema.")
        if metadata.get("continuous_features") != [
                EDGE_FEATURES_BY_SCHEMA["physical_v1"][i]
                for i in PHYSICAL_EDGE_CONTINUOUS_COLUMNS]:
            raise ValueError("Physical edge scaler feature order does not match physical_v1.")
        scaler = cls()
        scaler.schema_ = metadata.get("schema", "physical_v1")
        scaler.mean_ = np.asarray(metadata["mean_"], dtype=np.float64)
        scaler.var_ = np.asarray(metadata["var_"], dtype=np.float64)
        scaler.scale_ = np.asarray(metadata["scale_"], dtype=np.float64)
        scaler.n_samples_seen_ = int(metadata["n_samples_seen_"])
        if any(array.shape != (2,) for array in (scaler.mean_, scaler.var_, scaler.scale_)):
            raise ValueError("Physical edge scaler metadata must have two continuous channels.")
        if (not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
                or np.any(scaler.scale_ <= 0)):
            raise ValueError("Physical edge scaler metadata contains invalid values.")
        scaler.fitted_ = True
        return scaler
