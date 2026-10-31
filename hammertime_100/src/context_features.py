"""Causal raw node features and train-only speed reference calibration.

``legacy_v1`` is the original nine-column graph schema.  The optional
``circuit_context_v1`` schema appends brake, within-lap phase, track geometry,
and a training-fitted speed reference.  Circuit identity is used only for
calibration lookup and is never a numeric model input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


LEGACY_SCHEMA = "legacy_v1"
CONTEXT_SCHEMA = "circuit_context_v1"

DEFAULT_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
CONTEXT_FEATURES = (
    "Brake",
    "BrakeSupported",
    "lap_sin",
    "lap_cos",
    "lap_length_m",
    "abs_curvature",
    "lateral_accel_proxy",
    "GeometrySupported",
    "SpeedDeltaToTrainReference",
    "ReferenceAvailable",
)


def resolve_feature_schema(config: Mapping) -> str:
    """Return the configured graph schema; old configurations stay legacy."""
    graph_config = config.get("graph", {})
    schema = graph_config.get("node_feature_schema", graph_config.get("feature_schema", LEGACY_SCHEMA))
    if schema not in (LEGACY_SCHEMA, CONTEXT_SCHEMA):
        raise ValueError(f"Unsupported graph.node_feature_schema {schema!r}.")
    return str(schema)


def node_feature_schema(config: Mapping) -> str:
    """Canonical resolver name used by graph construction and config loading."""
    return resolve_feature_schema(config)


def feature_names(
    schema: str = LEGACY_SCHEMA,
    compound_categories: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the exact, stable feature order for a schema."""
    if schema not in (LEGACY_SCHEMA, CONTEXT_SCHEMA):
        raise ValueError(f"Unsupported feature schema {schema!r}.")
    categories = tuple(str(value).upper() for value in (compound_categories or DEFAULT_COMPOUNDS))
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Compound categories must be non-empty and unique.")
    names = ("Speed", "Throttle", *(f"Compound_{value}" for value in categories), "TyreLife", "MiniSector")
    if schema == CONTEXT_SCHEMA:
        names += CONTEXT_FEATURES
    return tuple(names)


def feature_schema_metadata(
    schema: str = LEGACY_SCHEMA,
    compound_categories: Sequence[str] | None = None,
) -> dict:
    """Serializable schema description, including raw units and field order."""
    names = feature_names(schema, compound_categories)
    categories = tuple(str(value).upper() for value in (compound_categories or DEFAULT_COMPOUNDS))
    units = {
        "Speed": "km/h",
        "Throttle": "source scale (typically percent)",
        "TyreLife": "laps",
        "MiniSector": "integer index",
        "Brake": "binary 0/1; interpolated channel thresholded at 0.5",
        "BrakeSupported": "binary 0/1",
        "lap_sin": "unitless",
        "lap_cos": "unitless",
        "lap_length_m": "m",
        "abs_curvature": "1/m",
        "lateral_accel_proxy": "m/s^2",
        "GeometrySupported": "binary 0/1",
        "SpeedDeltaToTrainReference": "km/h",
        "ReferenceAvailable": "binary 0/1",
    }
    for category in categories:
        units[f"Compound_{category}"] = "one-hot indicator"
    return {
        "id": schema,
        "feature_names": list(names),
        "raw_units": [units[name] for name in names],
        "geometry_lookback_frames": 2 if schema == CONTEXT_SCHEMA else 0,
        "speed_reference": (
            "median Speed by CircuitId and MiniSector over unique active training input frames; "
            "unseen groups are unavailable"
            if schema == CONTEXT_SCHEMA else None
        ),
    }


def _truthy(value, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    return bool(value)


def _finite(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame_key(race_id, time_sec) -> tuple[str, float]:
    value = _finite(time_sec)
    if race_id is None or pd.isna(race_id) or value is None:
        raise ValueError("Training frame membership requires finite race_id and time_sec values.")
    return str(race_id), round(value, 9)


def _membership_keys(membership: Iterable) -> set[tuple[str, float]]:
    keys: set[tuple[str, float]] = set()
    for item in membership:
        if isinstance(item, Mapping):
            race_id = item.get("race_id", item.get("RaceId"))
            time_sec = item.get("time_sec", item.get("TimeSec"))
        else:
            try:
                race_id, time_sec = item
            except (TypeError, ValueError) as exc:
                raise ValueError("Training input membership items must be (race_id, time_sec) pairs.") from exc
        keys.add(_frame_key(race_id, time_sec))
    if not keys:
        raise ValueError("Training input membership is empty.")
    return keys


def prepare_context_features(
    aligned_df: pd.DataFrame,
    sessions_by_race: Mapping[str, Mapping],
    sample_hz: float = 4.0,
    mini_sectors: int = 20,
    schema: str = CONTEXT_SCHEMA,
) -> pd.DataFrame:
    """Attach causal context columns without changing input rows or their order.

    ``sessions_by_race`` maps each RaceId to ``circuit_id`` and
    ``lap_length_m``. Geometry uses position and speed at the current and two
    immediately preceding grid frames for the same race and driver. It does
    not inspect future samples.
    """
    if schema not in (LEGACY_SCHEMA, CONTEXT_SCHEMA):
        raise ValueError(f"Unsupported feature schema {schema!r}.")
    if schema == LEGACY_SCHEMA:
        return aligned_df.copy()
    if sample_hz <= 0 or mini_sectors <= 0:
        raise ValueError("sample_hz and mini_sectors must be positive.")
    required = {"RaceId", "Driver", "TimeSec", "MiniSector", "Speed"}
    missing = required.difference(aligned_df.columns)
    if missing:
        raise ValueError(f"Context feature input is missing columns: {sorted(missing)}")

    result = aligned_df.copy().reset_index(drop=True)
    race_ids = result["RaceId"].astype(str)
    circuit_ids, lap_lengths = [], []
    for race_id in race_ids:
        metadata = sessions_by_race.get(race_id)
        if metadata is None:
            raise ValueError(f"No circuit metadata supplied for race {race_id!r}.")
        circuit_id = metadata.get("circuit_id")
        lap_length = _finite(metadata.get("lap_length_m"))
        if circuit_id is None or not str(circuit_id).strip() or lap_length is None or lap_length <= 0:
            raise ValueError(f"Invalid circuit metadata for race {race_id!r}.")
        circuit_ids.append(str(circuit_id))
        lap_lengths.append(lap_length)
    result["RaceId"] = race_ids
    result["CircuitId"] = circuit_ids
    result["lap_length_m"] = np.asarray(lap_lengths, dtype=np.float64)

    sectors = pd.to_numeric(result["MiniSector"], errors="coerce").to_numpy(dtype=float)
    valid_sector = np.isfinite(sectors)
    valid_sector &= (sectors >= 0) & (sectors < mini_sectors)
    phase = np.zeros(len(result), dtype=np.float64)
    phase[valid_sector] = 2.0 * np.pi * (sectors[valid_sector] + 0.5) / float(mini_sectors)
    result["lap_sin"] = np.sin(phase)
    result["lap_cos"] = np.cos(phase)

    brake = pd.to_numeric(result.get("Brake", pd.Series(np.nan, index=result.index)), errors="coerce")
    brake_supported = np.isfinite(brake.to_numpy(dtype=float))
    result["Brake"] = np.where(brake_supported, (brake.to_numpy(dtype=float) >= 0.5).astype(float), 0.0)
    result["BrakeSupported"] = brake_supported.astype(float)
    result["abs_curvature"] = 0.0
    result["lateral_accel_proxy"] = 0.0
    result["GeometrySupported"] = 0.0

    if {"X", "Y"}.issubset(result.columns):
        times = pd.to_numeric(result["TimeSec"], errors="coerce").to_numpy(dtype=float)
        speeds = pd.to_numeric(result["Speed"], errors="coerce").to_numpy(dtype=float) / 3.6
        x = pd.to_numeric(result["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(result["Y"], errors="coerce").to_numpy(dtype=float)
        geometry = np.zeros(len(result), dtype=float)
        for _, group in result.groupby(["RaceId", "Driver"], sort=False, dropna=False):
            indexes = group.sort_values("TimeSec", kind="stable").index.to_numpy()
            positions = result.index.get_indexer(indexes)
            for i in range(2, len(positions)):
                p0, p1, p2 = positions[i - 2:i + 1]
                if not all(np.isfinite(value) for value in (x[p0], y[p0], x[p1], y[p1], x[p2], y[p2],
                                                             times[p0], times[p1], times[p2],
                                                             speeds[p0], speeds[p1], speeds[p2])):
                    continue
                dt0, dt1 = times[p1] - times[p0], times[p2] - times[p1]
                expected_dt = 1.0 / float(sample_hz)
                tolerance = max(1e-6, expected_dt * 0.05)
                if abs(dt0 - expected_dt) > tolerance or abs(dt1 - expected_dt) > tolerance:
                    continue
                dx0, dy0 = x[p1] - x[p0], y[p1] - y[p0]
                dx1, dy1 = x[p2] - x[p1], y[p2] - y[p1]
                if math.hypot(dx0, dy0) <= 1e-6 or math.hypot(dx1, dy1) <= 1e-6:
                    continue
                heading_change = math.atan2(dx0 * dy1 - dy0 * dx1, dx0 * dx1 + dy0 * dy1)
                distance = 0.5 * (speeds[p0] + speeds[p1]) * dt0 + 0.5 * (speeds[p1] + speeds[p2]) * dt1
                if not math.isfinite(distance) or distance <= 1e-6:
                    continue
                curvature = abs(heading_change / distance)
                lateral_accel = speeds[p2] ** 2 * curvature
                if math.isfinite(curvature) and math.isfinite(lateral_accel):
                    geometry[p2] = curvature
                    result.iloc[p2, result.columns.get_loc("GeometrySupported")] = 1.0
                    result.iloc[p2, result.columns.get_loc("lateral_accel_proxy")] = lateral_accel
        result["abs_curvature"] = geometry

    result["SpeedDeltaToTrainReference"] = 0.0
    result["ReferenceAvailable"] = 0.0
    return result


def fit_context_calibration(
    aligned_df: pd.DataFrame,
    train_input_membership: Iterable,
    schema: str = CONTEXT_SCHEMA,
) -> dict:
    """Fit per-circuit/mini-sector median speed using unique train frames only."""
    if schema != CONTEXT_SCHEMA:
        return {"schema_id": schema, "references": [], "training_frame_count": 0}
    required = {"RaceId", "TimeSec", "Driver", "CircuitId", "MiniSector", "Speed"}
    missing = required.difference(aligned_df.columns)
    if missing:
        raise ValueError(f"Calibration input is missing columns: {sorted(missing)}")
    frame_keys = _membership_keys(train_input_membership)
    race_values = aligned_df["RaceId"].astype(str)
    time_values = pd.to_numeric(aligned_df["TimeSec"], errors="coerce")
    def is_training_frame(race, time) -> bool:
        if _finite(time) is None or pd.isna(race):
            return False
        return (str(race), round(float(time), 9)) in frame_keys

    selected = np.fromiter((is_training_frame(race, time)
                            for race, time in zip(race_values, time_values)),
                           dtype=bool, count=len(aligned_df))
    active = aligned_df.get("IsValid", pd.Series(True, index=aligned_df.index)).map(lambda value: _truthy(value, True))
    not_pit = ~aligned_df.get("IsPitLap", pd.Series(False, index=aligned_df.index)).map(_truthy)
    speeds = pd.to_numeric(aligned_df["Speed"], errors="coerce")
    sectors = pd.to_numeric(aligned_df["MiniSector"], errors="coerce")
    usable = selected & active.to_numpy(dtype=bool) & not_pit.to_numpy(dtype=bool)
    usable &= np.isfinite(speeds.to_numpy(dtype=float)) & np.isfinite(sectors.to_numpy(dtype=float))
    work = aligned_df.loc[usable, ["RaceId", "TimeSec", "Driver", "CircuitId"]].copy()
    work["MiniSector"] = sectors.loc[usable].astype(int).to_numpy()
    work["Speed"] = speeds.loc[usable].to_numpy(dtype=float)
    work = work.drop_duplicates(["RaceId", "TimeSec", "Driver"])
    if work.empty:
        raise ValueError("No active finite-speed training rows are available for context calibration.")
    grouped = work.groupby(["CircuitId", "MiniSector"], sort=True)["Speed"]
    references = [
        {"circuit_id": str(circuit), "mini_sector": int(sector),
         "median_speed_kph": float(values.median()), "training_row_count": int(values.count())}
        for (circuit, sector), values in grouped
    ]
    return {
        "schema_id": CONTEXT_SCHEMA,
        "algorithm": "median_speed_by_circuit_and_mini_sector_v1",
        "training_frame_count": len(frame_keys),
        "training_row_count": int(len(work)),
        "references": references,
    }


def apply_context_calibration(aligned_df: pd.DataFrame, calibration: Mapping) -> pd.DataFrame:
    """Apply only an exact circuit/mini-sector reference; unseen keys stay unsupported."""
    result = aligned_df.copy()
    refs = {}
    for item in calibration.get("references", []):
        key = (str(item["circuit_id"]), int(item["mini_sector"]))
        if key in refs:
            raise ValueError(f"Duplicate speed reference for {key!r}.")
        value = _finite(item.get("median_speed_kph"))
        if value is None:
            raise ValueError(f"Invalid speed reference for {key!r}.")
        refs[key] = value
    speeds = pd.to_numeric(result.get("Speed", pd.Series(np.nan, index=result.index)), errors="coerce")
    sectors = pd.to_numeric(result.get("MiniSector", pd.Series(np.nan, index=result.index)), errors="coerce")
    deltas = np.zeros(len(result), dtype=np.float32)
    supported = np.zeros(len(result), dtype=np.float32)
    if "CircuitId" in result.columns:
        circuits = result["CircuitId"].astype(str).to_numpy()
        for i, (circuit, sector, speed) in enumerate(zip(circuits, sectors, speeds)):
            if pd.isna(sector) or not np.isfinite(speed):
                continue
            reference = refs.get((circuit, int(sector)))
            if reference is not None:
                deltas[i] = float(speed) - reference
                supported[i] = 1.0
    result["SpeedDeltaToTrainReference"] = deltas
    result["ReferenceAvailable"] = supported
    return result


def build_node_feature_vector(
    row,
    num_compounds: int = 5,
    compound_categories: Sequence[str] | None = None,
    feature_schema: str = LEGACY_SCHEMA,
    node_feature_schema: str | None = None,
) -> np.ndarray:
    """Build the raw float32 input vector in the declared schema order.

    The first three arguments preserve the original graph helper call shape.
    """
    if node_feature_schema is not None:
        if feature_schema != LEGACY_SCHEMA and feature_schema != node_feature_schema:
            raise ValueError("Conflicting feature_schema and node_feature_schema values.")
        feature_schema = node_feature_schema
    categories = tuple(str(value).upper() for value in (compound_categories or DEFAULT_COMPOUNDS))
    if num_compounds != len(categories):
        raise ValueError("num_compounds must match the configured compound_categories length.")
    feature_names(feature_schema, categories)
    speed = _finite(row.get("Speed", 0.0)) or 0.0
    throttle = _finite(row.get("Throttle", 0.0)) or 0.0
    tyre_age = _finite(row.get("TyreLife", 1.0))
    mini_sector = _finite(row.get("MiniSector", 0.0))
    compound = str(row.get("Compound", "MEDIUM")).upper()
    fallback = categories.index("MEDIUM") if "MEDIUM" in categories else 0
    compound_idx = categories.index(compound) if compound in categories else fallback
    one_hot = np.zeros(len(categories), dtype=np.float32)
    one_hot[compound_idx] = 1.0
    values = [speed, throttle, *one_hot.tolist(),
              1.0 if tyre_age is None else tyre_age,
              0.0 if mini_sector is None else mini_sector]
    if feature_schema == CONTEXT_SCHEMA:
        extra = []
        for name in CONTEXT_FEATURES:
            value = _finite(row.get(name, 0.0))
            extra.append(0.0 if value is None else value)
        values.extend(extra)
    return np.asarray(values, dtype=np.float32)
