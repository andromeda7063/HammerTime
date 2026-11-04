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
CONTEXT_V2_SCHEMA = "circuit_context_v2"
CONTEXT_V3_SCHEMA = "circuit_context_v3"
CONTEXT_SCHEMAS = (CONTEXT_SCHEMA, CONTEXT_V2_SCHEMA, CONTEXT_V3_SCHEMA)
ALL_SCHEMAS = (LEGACY_SCHEMA, *CONTEXT_SCHEMAS)

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
CONTEXT_V2_FEATURES = (
    "SpeedRatioToTrainReference",
    "SpeedDeltaToTrainReference",
    "Brake",
    "DRSActive",
    "a_long",
    "lap_sin",
    "lap_cos",
    "lap_phase",
    "abs_curvature",
    "lateral_accel_proxy",
    "ReferenceAvailable",
)
# circuit_context_v3 replaces the train-circuit speed reference with causal,
# within-race references that exist for unseen circuits, and adds race
# progress inputs for the fuel-burn / stint trend contained in the target.
CONTEXT_V3_FEATURES = (
    "SpeedRatioToOwnRaceReference",
    "SpeedDeltaToOwnRaceReference",
    "OwnReferenceAvailable",
    "SpeedRatioToFieldRaceReference",
    "FieldReferenceAvailable",
    "Brake",
    "DRSActive",
    "a_long",
    "lap_sin",
    "lap_cos",
    "lap_phase",
    "abs_curvature",
    "lateral_accel_proxy",
    "RaceProgress",
    "StintLap",
)
RATIO_FEATURES = frozenset({
    "SpeedRatioToTrainReference", "SpeedRatioToOwnRaceReference", "SpeedRatioToFieldRaceReference",
})
# Causal reference windows: the own-car reference uses this driver's previous
# completed passes of the same mini-sector; the field reference uses the most
# recent completed passes of that mini-sector by any car.
OWN_REFERENCE_PASSES = 5
FIELD_REFERENCE_PASSES = 60


def resolve_feature_schema(config: Mapping) -> str:
    """Return the configured graph schema; old configurations stay legacy."""
    graph_config = config.get("graph", {})
    schema = graph_config.get("node_feature_schema", graph_config.get("feature_schema", LEGACY_SCHEMA))
    if schema not in ALL_SCHEMAS:
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
    if schema not in ALL_SCHEMAS:
        raise ValueError(f"Unsupported feature schema {schema!r}.")
    categories = tuple(str(value).upper() for value in (compound_categories or DEFAULT_COMPOUNDS))
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Compound categories must be non-empty and unique.")
    names = ("Speed", "Throttle", *(f"Compound_{value}" for value in categories), "TyreLife", "MiniSector")
    if schema == CONTEXT_SCHEMA:
        names += CONTEXT_FEATURES
    elif schema == CONTEXT_V2_SCHEMA:
        names += CONTEXT_V2_FEATURES
    elif schema == CONTEXT_V3_SCHEMA:
        names += CONTEXT_V3_FEATURES
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
        "lap_phase": "unitless track progress in [0, 1]",
        "lap_length_m": "m",
        "abs_curvature": "1/m",
        "lateral_accel_proxy": "m/s^2",
        "GeometrySupported": "binary 0/1",
        "SpeedDeltaToTrainReference": "km/h",
        "SpeedRatioToTrainReference": "dimensionless ratio v / v_ref; defaults to 1.0",
        "DRSActive": "binary 0/1; active rear wing open when DRS >= 10",
        "a_long": "m/s^2; longitudinal acceleration dv / dt",
        "ReferenceAvailable": "binary 0/1",
        "SpeedRatioToOwnRaceReference": "ratio v / median(own previous passes of this mini-sector); defaults to 1.0",
        "SpeedDeltaToOwnRaceReference": "km/h; v minus own causal in-race reference",
        "OwnReferenceAvailable": "binary 0/1",
        "SpeedRatioToFieldRaceReference": "ratio v / median(recent completed passes by any car); defaults to 1.0",
        "FieldReferenceAvailable": "binary 0/1",
        "RaceProgress": "LapNumber / maximum observed race lap, in [0, 1]",
        "StintLap": "laps completed since this car's last pit lap in the race",
    }
    for category in categories:
        units[f"Compound_{category}"] = "one-hot indicator"
    return {
        "id": schema,
        "feature_names": list(names),
        "raw_units": [units[name] for name in names],
        "geometry_lookback_frames": 2 if schema in CONTEXT_SCHEMAS else 0,
        "speed_reference": (
            "median Speed by CircuitId and MiniSector over unique active training input frames; "
            "unseen groups are unavailable"
            if schema in (CONTEXT_SCHEMA, CONTEXT_V2_SCHEMA) else (
                f"causal within-race references: own = median mean-speed of the car's last "
                f"{OWN_REFERENCE_PASSES} completed passes of the mini-sector; field = median of the last "
                f"{FIELD_REFERENCE_PASSES} completed passes by any car; only passes whose exit time is "
                "strictly before the frame are used; pit, safety-car and invalid rows are excluded"
                if schema == CONTEXT_V3_SCHEMA else None)
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
    if schema not in ALL_SCHEMAS:
        raise ValueError(f"Unsupported feature schema {schema!r}.")
    if schema == LEGACY_SCHEMA:
        return aligned_df
    if sample_hz <= 0 or mini_sectors <= 0:
        raise ValueError("sample_hz and mini_sectors must be positive.")
    required = {"RaceId", "Driver", "TimeSec", "MiniSector", "Speed"}
    missing = required.difference(aligned_df.columns)
    if missing:
        raise ValueError(f"Context feature input is missing columns: {sorted(missing)}")

    # Build a local frame without carrying potentially large provenance attrs
    # into another deep copy. The source frame and its attrs remain untouched.
    result = pd.DataFrame(aligned_df, copy=False)
    result.attrs = {}
    result = result.reset_index(drop=True)
    race_ids = result["RaceId"].astype(str)
    normalized_metadata = {}
    for race_id, metadata in sessions_by_race.items():
        key = str(race_id)
        circuit_id = metadata.get("circuit_id")
        lap_length = _finite(metadata.get("lap_length_m"))
        if circuit_id is None or not str(circuit_id).strip() or lap_length is None or lap_length <= 0:
            raise ValueError(f"Invalid circuit metadata for race {key!r}.")
        normalized_metadata[key] = (str(circuit_id), lap_length)
    observed_races = set(race_ids.unique())
    unknown_races = observed_races.difference(normalized_metadata)
    if unknown_races:
        raise ValueError(f"No circuit metadata supplied for race IDs: {sorted(unknown_races)}")
    circuit_map = {race: metadata[0] for race, metadata in normalized_metadata.items()}
    lap_length_map = {race: metadata[1] for race, metadata in normalized_metadata.items()}
    result["RaceId"] = race_ids
    result["CircuitId"] = race_ids.map(circuit_map).to_numpy()
    result["lap_length_m"] = race_ids.map(lap_length_map).to_numpy(dtype=np.float64)

    sectors = pd.to_numeric(result["MiniSector"], errors="coerce").to_numpy(dtype=float)
    valid_sector = np.isfinite(sectors)
    valid_sector &= (sectors >= 0) & (sectors < mini_sectors)
    lap_progress = np.zeros(len(result), dtype=np.float64)
    lap_progress[valid_sector] = (sectors[valid_sector] + 0.5) / float(mini_sectors)
    phase = 2.0 * np.pi * lap_progress
    result["lap_phase"] = lap_progress
    result["lap_sin"] = np.sin(phase)
    result["lap_cos"] = np.cos(phase)

    brake = pd.to_numeric(result.get("Brake", pd.Series(np.nan, index=result.index)), errors="coerce")
    brake_supported = np.isfinite(brake.to_numpy(dtype=float))
    result["Brake"] = np.where(brake_supported, (brake.to_numpy(dtype=float) >= 0.5).astype(float), 0.0)
    result["BrakeSupported"] = brake_supported.astype(float)

    drs = pd.to_numeric(aligned_df.get("DRSActive", pd.Series(0.0, index=result.index)), errors="coerce")
    result["DRSActive"] = np.where(np.isfinite(drs.to_numpy(dtype=float)), drs.to_numpy(dtype=float), 0.0)

    a_long_val = pd.to_numeric(aligned_df.get("a_long", pd.Series(0.0, index=result.index)), errors="coerce")
    result["a_long"] = np.where(np.isfinite(a_long_val.to_numpy(dtype=float)), a_long_val.to_numpy(dtype=float), 0.0)
    result["abs_curvature"] = 0.0
    result["lateral_accel_proxy"] = 0.0
    result["GeometrySupported"] = 0.0

    if {"X", "Y"}.issubset(result.columns):
        times = pd.to_numeric(result["TimeSec"], errors="coerce").to_numpy(dtype=float)
        speeds = pd.to_numeric(result["Speed"], errors="coerce").to_numpy(dtype=float) / 3.6
        x = pd.to_numeric(result["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(result["Y"], errors="coerce").to_numpy(dtype=float)
        row_position = np.arange(len(result), dtype=np.int64)
        trajectory = pd.DataFrame({
            "RaceId": result["RaceId"].to_numpy(),
            "Driver": result["Driver"].to_numpy(),
            "TimeSec": times,
            "SpeedMps": speeds,
            "X": x,
            "Y": y,
            "_row_position": row_position,
        }).sort_values(["RaceId", "Driver", "TimeSec"], kind="stable")
        grouped = trajectory.groupby(["RaceId", "Driver"], sort=False, dropna=False)
        previous1 = grouped[["TimeSec", "SpeedMps", "X", "Y"]].shift(1)
        previous2 = grouped[["TimeSec", "SpeedMps", "X", "Y"]].shift(2)
        t0 = previous2["TimeSec"].to_numpy(dtype=float)
        t1 = previous1["TimeSec"].to_numpy(dtype=float)
        t2 = trajectory["TimeSec"].to_numpy(dtype=float)
        v0 = previous2["SpeedMps"].to_numpy(dtype=float)
        v1 = previous1["SpeedMps"].to_numpy(dtype=float)
        v2 = trajectory["SpeedMps"].to_numpy(dtype=float)
        x0, y0 = previous2["X"].to_numpy(dtype=float), previous2["Y"].to_numpy(dtype=float)
        x1, y1 = previous1["X"].to_numpy(dtype=float), previous1["Y"].to_numpy(dtype=float)
        x2 = trajectory["X"].to_numpy(dtype=float)
        y2 = trajectory["Y"].to_numpy(dtype=float)
        dt0, dt1 = t1 - t0, t2 - t1
        expected_dt = 1.0 / float(sample_hz)
        tolerance = max(1e-6, expected_dt * 0.05)
        dx0, dy0 = x1 - x0, y1 - y0
        dx1, dy1 = x2 - x1, y2 - y1
        segment0 = np.hypot(dx0, dy0)
        segment1 = np.hypot(dx1, dy1)
        distance = 0.5 * (v0 + v1) * dt0 + 0.5 * (v1 + v2) * dt1
        finite = np.isfinite(np.column_stack((t0, t1, t2, v0, v1, v2, x0, y0, x1, y1, x2, y2))).all(axis=1)
        supported = finite & (np.abs(dt0 - expected_dt) <= tolerance)
        supported &= np.abs(dt1 - expected_dt) <= tolerance
        supported &= (segment0 > 1e-6) & (segment1 > 1e-6) & (distance > 1e-6)
        heading_change = np.arctan2(dx0 * dy1 - dy0 * dx1, dx0 * dx1 + dy0 * dy1)
        curvature = np.zeros(len(trajectory), dtype=np.float64)
        curvature[supported] = np.abs(heading_change[supported] / distance[supported])
        lateral_accel = np.zeros(len(trajectory), dtype=np.float64)
        lateral_accel[supported] = v2[supported] ** 2 * curvature[supported]
        support_by_row = np.zeros(len(result), dtype=np.float64)
        curvature_by_row = np.zeros(len(result), dtype=np.float64)
        lateral_by_row = np.zeros(len(result), dtype=np.float64)
        original_positions = trajectory["_row_position"].to_numpy(dtype=np.int64)
        support_by_row[original_positions[supported]] = 1.0
        curvature_by_row[original_positions[supported]] = curvature[supported]
        lateral_by_row[original_positions[supported]] = lateral_accel[supported]
        result["GeometrySupported"] = support_by_row
        result["abs_curvature"] = curvature_by_row
        result["lateral_accel_proxy"] = lateral_by_row

    result["SpeedDeltaToTrainReference"] = 0.0
    result["ReferenceAvailable"] = 0.0
    if schema == CONTEXT_V3_SCHEMA:
        result = add_causal_race_references(result)
        result = add_race_progress_features(result)
    return result


def _reference_usable_rows(frame: pd.DataFrame) -> np.ndarray:
    usable = np.ones(len(frame), dtype=bool)
    if "IsValid" in frame:
        usable &= frame["IsValid"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    for column in ("IsPitLap", "IsSafetyCar"):
        if column in frame:
            usable &= ~frame[column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    usable &= np.isfinite(pd.to_numeric(frame["Speed"], errors="coerce").to_numpy(dtype=float))
    usable &= np.isfinite(pd.to_numeric(frame["MiniSector"], errors="coerce").to_numpy(dtype=float))
    usable &= np.isfinite(pd.to_numeric(frame["LapNumber"], errors="coerce").to_numpy(dtype=float))
    return usable


def _sector_passes(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per completed (race, driver, lap, mini-sector) pass.

    ``ExitTimeSec`` is the last usable sample of the pass. A pass can only be
    used for frames strictly after that time, so a reference never includes
    the pass that is still in progress.
    """
    usable = _reference_usable_rows(frame)
    work = pd.DataFrame({
        "RaceId": frame["RaceId"].astype(str).to_numpy()[usable],
        "Driver": frame["Driver"].astype(str).to_numpy()[usable],
        "LapNumber": pd.to_numeric(frame["LapNumber"], errors="coerce").to_numpy(dtype=float)[usable],
        "MiniSector": pd.to_numeric(frame["MiniSector"], errors="coerce").to_numpy(dtype=float)[usable].astype(int),
        "TimeSec": pd.to_numeric(frame["TimeSec"], errors="coerce").to_numpy(dtype=float)[usable],
        "Speed": pd.to_numeric(frame["Speed"], errors="coerce").to_numpy(dtype=float)[usable],
    })
    if work.empty:
        return work.assign(PassSpeed=pd.Series(dtype=float), ExitTimeSec=pd.Series(dtype=float))
    passes = (work.groupby(["RaceId", "Driver", "LapNumber", "MiniSector"], sort=False)
              .agg(PassSpeed=("Speed", "mean"), ExitTimeSec=("TimeSec", "max"))
              .reset_index())
    return passes


def _asof_reference(frame: pd.DataFrame, passes: pd.DataFrame, by: list[str],
                    window: int, column: str) -> np.ndarray:
    """Rolling median of the last ``window`` passes, joined strictly backward in time."""
    values = np.full(len(frame), np.nan, dtype=float)
    if passes.empty:
        return values
    ordered = passes.sort_values([*by, "ExitTimeSec"], kind="stable").copy()
    ordered[column] = (ordered.groupby(by, sort=False)["PassSpeed"]
                       .transform(lambda series: series.rolling(window, min_periods=1).median()))
    reference = ordered[[*by, "ExitTimeSec", column]].rename(columns={"ExitTimeSec": "TimeSec"})
    reference = reference.sort_values("TimeSec", kind="stable")
    query = pd.DataFrame({
        "_row": np.arange(len(frame), dtype=np.int64),
        "TimeSec": pd.to_numeric(frame["TimeSec"], errors="coerce").to_numpy(dtype=float),
    })
    for key in by:
        if key == "MiniSector":
            query[key] = pd.to_numeric(frame[key], errors="coerce").fillna(-1).astype(int).to_numpy()
        else:
            query[key] = frame[key].astype(str).to_numpy()
    query = query[np.isfinite(query["TimeSec"].to_numpy())].sort_values("TimeSec", kind="stable")
    joined = pd.merge_asof(query, reference, on="TimeSec", by=by,
                           direction="backward", allow_exact_matches=False)
    values[joined["_row"].to_numpy()] = joined[column].to_numpy(dtype=float)
    return values


def add_causal_race_references(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach own-car and field causal in-race speed references (circuit_context_v3).

    Unlike the train-circuit reference used by v1/v2, these references are
    built from earlier completed passes of the same race. They are available
    for circuits that never appear in training and use no future samples.
    """
    required = {"RaceId", "Driver", "TimeSec", "MiniSector", "Speed", "LapNumber"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Causal race references need columns: {sorted(missing)}")
    result = frame
    passes = _sector_passes(result)
    own = _asof_reference(result, passes, ["RaceId", "Driver", "MiniSector"],
                          OWN_REFERENCE_PASSES, "_own_ref")
    field = _asof_reference(result, passes, ["RaceId", "MiniSector"],
                            FIELD_REFERENCE_PASSES, "_field_ref")
    speed = pd.to_numeric(result["Speed"], errors="coerce").to_numpy(dtype=float)
    own_ok = np.isfinite(own) & np.isfinite(speed) & (own > 5.0)
    field_ok = np.isfinite(field) & np.isfinite(speed) & (field > 5.0)
    own_ratio = np.ones(len(result), dtype=np.float32)
    own_delta = np.zeros(len(result), dtype=np.float32)
    field_ratio = np.ones(len(result), dtype=np.float32)
    own_ratio[own_ok] = np.clip(speed[own_ok] / own[own_ok], 0.2, 2.5)
    own_delta[own_ok] = speed[own_ok] - own[own_ok]
    field_ratio[field_ok] = np.clip(speed[field_ok] / field[field_ok], 0.2, 2.5)
    result["SpeedRatioToOwnRaceReference"] = own_ratio
    result["SpeedDeltaToOwnRaceReference"] = own_delta
    result["OwnReferenceAvailable"] = own_ok.astype(np.float32)
    result["SpeedRatioToFieldRaceReference"] = field_ratio
    result["FieldReferenceAvailable"] = field_ok.astype(np.float32)
    return result


def add_race_progress_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach RaceProgress and StintLap (circuit_context_v3).

    RaceProgress divides by the race's maximum observed lap, which equals the
    scheduled distance except for shortened races; it is a race-length
    normalizer known before the start, not an outcome. StintLap counts laps
    since the car's most recent pit lap and only uses earlier laps.
    """
    result = frame
    laps = pd.to_numeric(result["LapNumber"], errors="coerce")
    race_ids = result["RaceId"].astype(str)
    max_lap = laps.groupby(race_ids).transform("max")
    progress = (laps / max_lap.where(max_lap > 0)).clip(0.0, 1.0)
    result["RaceProgress"] = progress.fillna(0.0).to_numpy(dtype=np.float32)
    stint = np.zeros(len(result), dtype=np.float32)
    pit = (result["IsPitLap"].astype("boolean").fillna(False)
           if "IsPitLap" in result else pd.Series(False, index=result.index))
    lap_table = pd.DataFrame({"RaceId": race_ids, "Driver": result["Driver"].astype(str),
                              "LapNumber": laps, "Pit": pit.astype(bool)}).dropna(subset=["LapNumber"])
    if not lap_table.empty:
        per_lap = (lap_table.groupby(["RaceId", "Driver", "LapNumber"], sort=True)["Pit"].any()
                   .reset_index())
        last_pit = per_lap["LapNumber"].where(per_lap["Pit"])
        per_lap["_last_pit"] = last_pit.groupby([per_lap["RaceId"], per_lap["Driver"]]).ffill().fillna(0.0)
        per_lap["StintLap"] = (per_lap["LapNumber"] - per_lap["_last_pit"]).clip(lower=0.0)
        keys = pd.MultiIndex.from_frame(per_lap[["RaceId", "Driver", "LapNumber"]])
        lookup = pd.Series(per_lap["StintLap"].to_numpy(dtype=float), index=keys)
        row_keys = pd.MultiIndex.from_arrays([race_ids.to_numpy(), result["Driver"].astype(str).to_numpy(),
                                              laps.to_numpy(dtype=float)])
        mapped = lookup.reindex(row_keys).to_numpy(dtype=float)
        stint = np.where(np.isfinite(mapped), mapped, 0.0).astype(np.float32)
    result["StintLap"] = stint
    return result


def fit_context_calibration(
    aligned_df: pd.DataFrame | Sequence[pd.DataFrame],
    train_input_membership: Iterable,
    schema: str = CONTEXT_SCHEMA,
) -> dict:
    """Fit per-circuit/mini-sector median speed using unique train frames only."""
    if schema not in (CONTEXT_SCHEMA, CONTEXT_V2_SCHEMA):
        return {"schema_id": schema, "references": [], "training_frame_count": 0}
    frame_keys = _membership_keys(train_input_membership)
    train_races = {race_id for race_id, _ in frame_keys}

    if isinstance(aligned_df, pd.DataFrame):
        tables = [aligned_df]
    else:
        tables = [tbl for tbl in aligned_df if not tbl.empty and str(tbl["RaceId"].iloc[0]) in train_races]

    required = {"RaceId", "TimeSec", "Driver", "CircuitId", "MiniSector", "Speed"}
    for tbl in tables:
        missing = required.difference(tbl.columns)
        if missing:
            raise ValueError(f"Calibration input is missing columns: {sorted(missing)}")

    membership_index = pd.MultiIndex.from_tuples(sorted(frame_keys), names=["RaceId", "TimeSec"])
    filtered_subsets = []
    for tbl in tables:
        frame = pd.DataFrame(tbl, copy=False)
        frame.attrs = {}
        race_values = frame["RaceId"].astype(str)
        time_values = pd.to_numeric(frame["TimeSec"], errors="coerce")
        aligned_index = pd.MultiIndex.from_arrays(
            [race_values.to_numpy(), np.round(time_values.to_numpy(dtype=float), 9)],
            names=["RaceId", "TimeSec"],
        )
        selected = aligned_index.isin(membership_index)
        active = frame.get("IsValid", pd.Series(True, index=frame.index)).fillna(True).astype(bool)
        not_pit = ~frame.get("IsPitLap", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        speeds = pd.to_numeric(frame["Speed"], errors="coerce")
        sectors = pd.to_numeric(frame["MiniSector"], errors="coerce")
        usable = selected & active.to_numpy(dtype=bool) & not_pit.to_numpy(dtype=bool)
        usable &= np.isfinite(speeds.to_numpy(dtype=float)) & np.isfinite(sectors.to_numpy(dtype=float))
        if not usable.any():
            continue
        work_piece = frame.loc[usable, ["RaceId", "TimeSec", "Driver", "CircuitId"]].copy()
        work_piece.attrs = {}
        work_piece["MiniSector"] = sectors.loc[usable].astype(int).to_numpy()
        work_piece["Speed"] = speeds.loc[usable].to_numpy(dtype=float)
        filtered_subsets.append(work_piece)

    if not filtered_subsets:
        raise ValueError("No active finite-speed training rows are available for context calibration.")
    work = pd.concat(filtered_subsets, ignore_index=True).drop_duplicates(["RaceId", "TimeSec", "Driver"])
    if work.empty:
        raise ValueError("No active finite-speed training rows are available for context calibration.")
    grouped = work.groupby(["CircuitId", "MiniSector"], sort=True)["Speed"]
    references = [
        {"circuit_id": str(circuit), "mini_sector": int(sector),
         "median_speed_kph": float(values.median()), "training_row_count": int(values.count())}
        for (circuit, sector), values in grouped
    ]
    return {
        "schema_id": schema,
        "algorithm": "median_speed_by_circuit_and_mini_sector_v1",
        "training_frame_count": len(frame_keys),
        "training_row_count": int(len(work)),
        "references": references,
    }


def apply_context_calibration(aligned_df: pd.DataFrame, calibration: Mapping) -> pd.DataFrame:
    """Apply only an exact circuit/mini-sector reference; unseen keys stay unsupported."""
    result = pd.DataFrame(aligned_df, copy=False)
    result.attrs = {}
    result = result.reset_index(drop=True)
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
    ratios = np.ones(len(result), dtype=np.float32)
    supported = np.zeros(len(result), dtype=np.float32)
    if "CircuitId" in result.columns:
        circuit_values = result["CircuitId"].astype(str)
        sector_values = sectors.fillna(-1).astype(int)
        reference_index = pd.MultiIndex.from_tuples(list(refs), names=["CircuitId", "MiniSector"])
        reference_array = np.asarray(list(refs.values()), dtype=float)
        aligned_keys = pd.MultiIndex.from_arrays(
            [circuit_values.to_numpy(), sector_values.to_numpy()],
            names=["CircuitId", "MiniSector"],
        )
        reference_positions = reference_index.get_indexer(aligned_keys)
        reference_values = np.full(len(result), np.nan, dtype=float)
        found_reference = reference_positions >= 0
        reference_values[found_reference] = reference_array[reference_positions[found_reference]]
        available = np.isfinite(reference_values) & np.isfinite(speeds.to_numpy(dtype=float))
        deltas[available] = speeds.to_numpy(dtype=float)[available] - reference_values[available]
        valid_ref = available & (reference_values > 5.0)
        ratios[valid_ref] = np.clip(speeds.to_numpy(dtype=float)[valid_ref] / reference_values[valid_ref], 0.2, 2.5)
        supported[available] = 1.0
    result["SpeedDeltaToTrainReference"] = deltas
    result["SpeedRatioToTrainReference"] = ratios
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
    elif feature_schema == CONTEXT_V2_SCHEMA:
        # Kept byte-identical to the historical v2 builder: an absent column
        # yields 0.0 (including the ratio); only a non-finite value yields 1.0.
        extra = []
        for name in CONTEXT_V2_FEATURES:
            value = _finite(row.get(name, 0.0))
            if value is None:
                extra.append(1.0 if name == "SpeedRatioToTrainReference" else 0.0)
            else:
                extra.append(value)
        values.extend(extra)
    elif feature_schema == CONTEXT_V3_SCHEMA:
        extra = []
        for name in CONTEXT_V3_FEATURES:
            default = 1.0 if name in RATIO_FEATURES else 0.0
            value = _finite(row.get(name, default))
            extra.append(default if value is None else value)
        values.extend(extra)
    return np.asarray(values, dtype=np.float32)
