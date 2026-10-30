"""Normalize cached OpenF1 endpoint tables into HammerTime's aligned schema."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from preprocess import align_driver_telemetry, compute_time_gaps


def _table(bundle: dict, *names: str) -> pd.DataFrame:
    tables = bundle.get("tables", {})
    empty: pd.DataFrame | None = None
    for name in names:
        frame = tables.get(name)
        if isinstance(frame, pd.DataFrame):
            if not frame.empty:
                return frame.copy()
            empty = frame
    return empty.copy() if empty is not None else pd.DataFrame()


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")


def _seconds_from_origin(values: pd.Series, origin: pd.Timestamp) -> pd.Series:
    return (_utc_series(values) - origin).dt.total_seconds()


def _driver_key(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _is_safety_message(message: object) -> bool:
    return bool(re.search(r"\b(safety car|virtual safety car|\bvsc\b)\b", str(message), re.I))


def _control_windows(frame: pd.DataFrame, origin: pd.Timestamp) -> list[tuple[float, float]]:
    """Map deployment through an observed green/clear message to SC windows."""
    if frame.empty:
        return []
    if not {"date", "message"}.issubset(frame.columns):
        raise ValueError("OpenF1 race-control records require date and message.")
    rows = frame.copy()
    rows["_time"] = _seconds_from_origin(rows["date"], origin)
    if rows["_time"].isna().any():
        raise ValueError("OpenF1 race-control records contain an invalid date.")
    rows = rows.sort_values("_time")
    windows: list[tuple[float, float]] = []
    active_start: float | None = None
    for _, row in rows.iterrows():
        message = str(row["message"]).strip().lower()
        stamp = float(row["_time"])
        scope = str(row.get("scope", "")).casefold()
        global_scope = (scope == "track" or
                        (scope in ("", "none", "nan") and pd.isna(row.get("sector"))
                         and pd.isna(row.get("driver_number"))))
        clear = global_scope and (str(row.get("flag", "")).upper() == "GREEN" or any(token in message for token in (
            "green flag", "track clear", "all clear", "clear of the track", "safety car withdrawn"
        )))
        if clear:
            if active_start is not None:
                windows.append((active_start, stamp))
                active_start = None
            continue
        if not _is_safety_message(message):
            continue
        deployed = any(token in message for token in ("deployed", "deploiement"))
        ongoing = "in this lap" in message or "ending" in message
        if deployed:
            if active_start is None:
                active_start = stamp
        elif ongoing:
            # Ending notices signal a transition, while the restriction remains
            # active until an observed green/clear message.
            if active_start is None:
                raise ValueError("OpenF1 safety-car ending notice has no observed deployment.")
        else:
            raise ValueError(f"Ambiguous OpenF1 safety-car race-control message: {row['message']!r}")
    if active_start is not None:
        windows.append((active_start, np.inf))
    return windows


def _boundary_value(times: np.ndarray, values: np.ndarray, target: float,
                    max_gap: float) -> float | None:
    right = int(np.searchsorted(times, target, side="left"))
    if right < len(times) and np.isclose(times[right], target, atol=1e-9, rtol=0):
        return float(values[right])
    if right == 0 or right >= len(times):
        return None
    if times[right] - times[right - 1] > max_gap + 1e-9:
        return None
    return float(np.interp(target, times[right - 1:right + 1], values[right - 1:right + 1]))


def _interpolate_supported(source: pd.DataFrame, column: str, targets: np.ndarray,
                           max_gap: float) -> np.ndarray:
    values = pd.to_numeric(source[column], errors="coerce")
    valid = source.loc[values.notna(), ["TimeSec"]].copy()
    valid[column] = values.loc[values.notna()].to_numpy(dtype=float)
    valid = valid.sort_values("TimeSec").drop_duplicates("TimeSec", keep="last")
    result = np.full(len(targets), np.nan, dtype=float)
    if valid.empty:
        return result
    times = valid["TimeSec"].to_numpy(dtype=float)
    samples = valid[column].to_numpy(dtype=float)
    for index, target in enumerate(targets):
        estimate = _boundary_value(times, samples, float(target), max_gap)
        if estimate is not None:
            result[index] = estimate
    return result


def _integrate_lap(samples: pd.DataFrame, start: float, end: float,
                   max_gap: float) -> tuple[pd.DataFrame, str | None, float | None]:
    """Build a lap-local distance trace from supported speed observations."""
    speed = samples[["TimeSec", "Speed"]].copy()
    speed["Speed"] = pd.to_numeric(speed["Speed"], errors="coerce")
    speed = speed.dropna().sort_values("TimeSec").drop_duplicates("TimeSec", keep="last")
    times = speed["TimeSec"].to_numpy(dtype=float)
    values = speed["Speed"].to_numpy(dtype=float)
    if len(times) < 2:
        return pd.DataFrame(), "insufficient observed speed for lap integration", None
    start_speed = _boundary_value(times, values, start, max_gap)
    end_speed = _boundary_value(times, values, end, max_gap)
    if start_speed is None or end_speed is None:
        return pd.DataFrame(), "lap boundary lacks supported speed observations", None
    inside = (times > start) & (times < end)
    lap_times = np.concatenate(([start], times[inside], [end]))
    lap_speed = np.concatenate(([start_speed], values[inside], [end_speed]))
    if np.any(np.diff(lap_times) > max_gap + 1e-9):
        return pd.DataFrame(), "unsupported speed gap within lap", None
    # OpenF1 speed is km/h. Trapezoidal integration yields distance in meters.
    increments = (lap_speed[:-1] + lap_speed[1:]) / 2.0 / 3.6 * np.diff(lap_times)
    distances = np.concatenate(([0.0], np.cumsum(increments)))
    return pd.DataFrame({"TimeSec": lap_times, "Speed": lap_speed,
                         "Distance": distances}), None, float(distances[-1])


def normalize_openf1_tables(bundle: dict, config: dict) -> pd.DataFrame:
    """Convert an OpenF1 bundle to the aligned telemetry table used by the pipeline.

    The bundle contract is ``{"session": ..., "tables": {endpoint: DataFrame},
    "raw_artifacts": ..., "provenance": ...}``. This function is offline and
    performs no acquisition or filesystem writes.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get("tables"), dict):
        raise ValueError("OpenF1 bundle must contain a tables mapping.")
    session = bundle.get("session") or {}
    origin_value = session.get("date_start")
    origin = pd.to_datetime(origin_value, utc=True, errors="coerce")
    if pd.isna(origin):
        raise ValueError("OpenF1 session.date_start is required as the UTC session origin.")

    prep, track = config["preprocess"], config["track"]
    sample_hz = float(prep["sample_hz"])
    max_gap = float(prep["max_interpolate_sec"])
    mini_sectors = int(track["mini_sectors"])
    lap_length = float(track["lap_length_m"])
    if sample_hz <= 0 or max_gap < 0 or mini_sectors <= 0 or lap_length <= 0:
        raise ValueError("OpenF1 normalization requires valid preprocess and track settings.")

    drivers = _table(bundle, "drivers")
    laps = _table(bundle, "laps")
    stints = _table(bundle, "stints")
    car = _table(bundle, "car_data", "car-data")
    position = _table(bundle, "position", "location")
    pit_table = _table(bundle, "pit")
    race_control = _table(bundle, "race_control", "race-control")
    if laps.empty:
        raise ValueError("OpenF1 laps table is required for lap timing metadata.")
    if car.empty:
        raise ValueError("OpenF1 car_data telemetry with observed speed is required.")

    for frame, stamp_col in ((laps, "date_start"), (car, "date"), (position, "date")):
        if not frame.empty and stamp_col in frame:
            frame["TimeSec"] = _seconds_from_origin(frame[stamp_col], origin)
    if "driver_number" not in laps or "driver_number" not in car:
        raise ValueError("OpenF1 laps and car_data tables require driver_number.")
    if "lap_number" not in laps or "TimeSec" not in laps:
        raise ValueError("OpenF1 laps table requires lap_number and date_start.")
    if "speed" not in car or "TimeSec" not in car:
        raise ValueError("OpenF1 car_data table requires date and observed speed.")

    # Driver metadata is only retained when one non-conflicting team is present.
    team_by_driver: dict[str, str | None] = {}
    invalid_driver_metadata: set[str] = set()
    if not drivers.empty and "driver_number" in drivers:
        for key, rows in drivers.groupby(drivers["driver_number"].map(_driver_key), dropna=True):
            if "team_name" not in rows:
                team_by_driver[key] = None
                invalid_driver_metadata.add(key)
                continue
            teams = rows["team_name"].dropna().astype(str).str.strip().unique()
            team_by_driver[key] = teams[0] if len(teams) == 1 else None
            if len(teams) != 1:
                invalid_driver_metadata.add(key)

    # Normalize speed and optional position records on one sparse per-driver clock.
    car["Driver"] = car["driver_number"].map(_driver_key)
    car["Speed"] = pd.to_numeric(car["speed"], errors="coerce")
    observed_speed = car["Speed"].dropna().to_numpy(dtype=float)
    if np.any(~np.isfinite(observed_speed)) or np.any(observed_speed < 0):
        raise ValueError("OpenF1 car_data speed must be finite and non-negative when observed.")
    if "throttle" in car:
        car["Throttle"] = pd.to_numeric(car["throttle"], errors="coerce")
    else:
        car["Throttle"] = np.nan
    if "brake" in car:
        raw_brake = pd.to_numeric(car["brake"], errors="coerce")
        # OpenF1 encodes brake as 0/100 (off/on); keep missing values.
        car["Brake"] = np.where(raw_brake.notna(), (raw_brake > 0).astype(float), np.nan)
    else:
        car["Brake"] = np.nan
    telemetry_parts = [car[["Driver", "TimeSec", "Speed", "Throttle", "Brake"]]]
    if not position.empty and {"driver_number", "TimeSec"}.issubset(position.columns):
        position["Driver"] = position["driver_number"].map(_driver_key)
        for source, target in (("x", "X"), ("y", "Y"), ("z", "Z")):
            if source in position:
                position[target] = pd.to_numeric(position[source], errors="coerce")
            else:
                position[target] = np.nan
        telemetry_parts.append(position[["Driver", "TimeSec", "X", "Y", "Z"]])
    telemetry = pd.concat(telemetry_parts, ignore_index=True, sort=False)
    for optional_channel in ("X", "Y", "Z"):
        if optional_channel not in telemetry:
            telemetry[optional_channel] = np.nan
    telemetry = telemetry.dropna(subset=["Driver", "TimeSec"])
    # Coalesce independent endpoint rows sharing the same instant without losing
    # channels that were absent from one of the endpoints.
    telemetry = telemetry.groupby(["Driver", "TimeSec"], as_index=False, sort=True).first()

    # Build the common session grid from observed lap timing boundaries.
    laps["Driver"] = laps["driver_number"].map(_driver_key)
    laps["LapNumber"] = pd.to_numeric(laps["lap_number"], errors="coerce")
    raw_lap_numbers = laps["LapNumber"].dropna().to_numpy(dtype=float)
    if np.any(~np.isfinite(raw_lap_numbers)) or np.any(raw_lap_numbers % 1 != 0):
        raise ValueError("OpenF1 lap_number values must be finite integers.")
    laps["LapDurationSec"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
    laps = laps.dropna(subset=["Driver", "LapNumber", "TimeSec", "LapDurationSec"])
    laps = laps[(laps["LapDurationSec"] > 0) & np.isfinite(laps["LapDurationSec"])]
    if laps.empty:
        raise ValueError("OpenF1 laps table has no valid date_start/lap_duration rows.")
    session_start = float(laps["TimeSec"].min())
    session_end = float((laps["TimeSec"] + laps["LapDurationSec"]).max())
    dt = 1.0 / sample_hz
    shared_grid = np.arange(session_start, session_end + dt * 1e-6, dt)

    # Resolve each lap's tyre metadata by stint range. Stint starting tyre age
    # advances by lap offset; no currentTyreLife sample is used as a substitute.
    stints["Driver"] = stints.get("driver_number", pd.Series(dtype=object)).map(_driver_key)
    pit_lap_numbers: dict[str, set[int]] = defaultdict(set)
    if not pit_table.empty and {"driver_number", "lap_number"}.issubset(pit_table.columns):
        for _, pit_row in pit_table.iterrows():
            pit_driver = _driver_key(pit_row["driver_number"])
            pit_lap_no = pd.to_numeric(pd.Series([pit_row["lap_number"]]), errors="coerce").iloc[0]
            if pit_driver is not None and pd.notna(pit_lap_no):
                pit_lap_numbers[pit_driver].add(int(pit_lap_no))
    control_windows = _control_windows(race_control, origin)
    race_id = config.get("session", {}).get("race_id") or (
        f"{config.get('session', {}).get('year', 'unknown')}-"
        f"{config.get('session', {}).get('gp', 'unknown')}-"
        f"{config.get('session', {}).get('session_type', 'R')}"
    )

    aligned_parts: list[pd.DataFrame] = []
    invalid_reasons: dict[str, list[str]] = {}

    def record_invalid(lap_key: str, reason: str) -> None:
        reasons = invalid_reasons.setdefault(lap_key, [])
        if reason not in reasons:
            reasons.append(reason)

    distance_errors: dict[str, float] = {}
    retained_laps: set[tuple[str, int]] = set()
    for _, lap in laps.sort_values(["Driver", "LapNumber"]).iterrows():
        driver = lap["Driver"]
        lap_no = int(lap["LapNumber"])
        lap_start = float(lap["TimeSec"])
        lap_end = lap_start + float(lap["LapDurationSec"])
        lap_key = f"{driver}:{lap_no}"
        driver_samples = telemetry[telemetry["Driver"] == driver].copy()
        trace, reason, integrated_m = _integrate_lap(driver_samples, lap_start, lap_end, max_gap)
        if reason:
            record_invalid(lap_key, reason)
            distance_trace = pd.DataFrame({"TimeSec": [], "Distance": []})
        else:
            distance_errors[lap_key] = float(integrated_m - lap_length)
            retained_laps.add((driver, lap_no))
            distance_trace = trace[["TimeSec", "Distance"]].copy()
        # Keep the union of asynchronous OpenF1 channels; unsupported brackets
        # remain missing, and no channel is extrapolated beyond observations.
        channel_times = driver_samples.loc[
            (driver_samples["TimeSec"] >= lap_start) & (driver_samples["TimeSec"] <= lap_end),
            "TimeSec",
        ].to_numpy(dtype=float)
        targets = np.unique(np.concatenate((distance_trace["TimeSec"].to_numpy(dtype=float), channel_times,
                                            np.asarray([lap_start, lap_end]))))
        trace = pd.DataFrame({"TimeSec": targets})
        trace["Distance"] = _interpolate_supported(distance_trace, "Distance", targets, max_gap)
        for channel in ("Speed", "Throttle", "Brake", "X", "Y", "Z"):
            trace[channel] = _interpolate_supported(driver_samples, channel, targets, max_gap)

        team = team_by_driver.get(driver)
        compound, tyre_age = None, np.nan
        if not stints.empty and {"Driver", "lap_start", "lap_end", "compound", "tyre_age_at_start"}.issubset(stints.columns):
            candidates = stints[(stints["Driver"] == driver)
                                & (pd.to_numeric(stints["lap_start"], errors="coerce") <= lap_no)
                                & (pd.to_numeric(stints["lap_end"], errors="coerce") >= lap_no)]
            if len(candidates) == 1:
                stint = candidates.iloc[0]
                compound = str(stint["compound"]).upper() if pd.notna(stint["compound"]) else None
                try:
                    # FastF1 defines the stint's first lap as tyre_age_at_start + 1.
                    tyre_age = float(stint["tyre_age_at_start"]) + lap_no - int(stint["lap_start"]) + 1
                except (TypeError, ValueError):
                    tyre_age = np.nan
            elif len(candidates) > 1:
                record_invalid(lap_key, "conflicting stint metadata")
        if compound is None or not np.isfinite(tyre_age):
            record_invalid(lap_key, "missing tyre compound or age metadata")
        if driver not in team_by_driver or driver in invalid_driver_metadata:
            record_invalid(lap_key, "missing or conflicting driver team metadata")

        is_pit = False
        if "pitlap_number" in laps:
            pit_no = pd.to_numeric(pd.Series([lap.get("pitlap_number")]), errors="coerce").iloc[0]
            is_pit |= bool(pd.notna(pit_no) and int(pit_no) == lap_no)
        is_pit |= lap_no in pit_lap_numbers.get(driver, set())
        if "is_pit_out_lap" in laps:
            is_pit |= bool(lap.get("is_pit_out_lap") is True or str(lap.get("is_pit_out_lap")).lower() == "true")
        lap_safety = any(left < lap_end and right > lap_start for left, right in control_windows)
        lap_frame = pd.DataFrame({
            "LapNumber": [lap_no],
            "LapStartTime": pd.to_timedelta([lap_start], unit="s"),
            "LapTime": pd.to_timedelta([float(lap["LapDurationSec"])], unit="s"),
            "Compound": [compound], "TyreLife": [tyre_age], "Team": [team],
            "PitInTime": [pd.NaT], "PitOutTime": [pd.NaT],
            "TrackStatus": ["4" if lap_safety else "1"],
        })
        # The shared grid remains aligned to the session origin; each lap is
        # passed separately so neither distance nor interpolation crosses resets.
        lap_grid = shared_grid[(shared_grid >= lap_start - 1e-9) & (shared_grid < lap_end - 1e-9)]
        aligned = align_driver_telemetry(
            driver, trace, lap_frame, sample_hz=sample_hz, max_interpolate_sec=max_gap,
            mini_sectors=mini_sectors, lap_length_m=lap_length, time_grid=lap_grid,
            flag_safety_car=bool(prep.get("flag_safety_car", True)),
            flag_pit_laps=bool(prep.get("flag_pit_laps", True)),
        )
        if aligned.empty:
            continue
        aligned["IsPitLap"] = bool(is_pit) if prep.get("flag_pit_laps", True) else False
        aligned["IsSafetyCar"] = bool(lap_safety) if prep.get("flag_safety_car", True) else False
        if reason:
            aligned["IsValid"] = False
        invalid_metadata = set(invalid_reasons.get(lap_key, ())) & {
            "conflicting stint metadata", "missing or conflicting driver team metadata"
        }
        if invalid_metadata:
            aligned["IsValid"] = False
        aligned["RaceId"] = str(race_id)
        aligned_parts.append(aligned)

    if not aligned_parts:
        raise ValueError("No OpenF1 driver laps could be aligned to the shared session grid.")
    combined = pd.concat(aligned_parts, ignore_index=True)
    processed = compute_time_gaps(combined, sample_hz=sample_hz, lap_length_m=lap_length)
    valid = processed.get("IsValid", pd.Series(False, index=processed.index)).fillna(False).astype(bool)
    coverage = {
        "input_laps": int(len(laps)),
        "retained_distance_laps": int(len(retained_laps)),
        "aligned_rows": int(len(processed)),
        "valid_rows": int(valid.sum()),
        "retained_samples": int(valid.sum()),
        "retained_sectors": int(processed.loc[valid, "MiniSector"].nunique()),
        "invalid_lap_reasons": invalid_reasons,
        "lap_distance_errors_m": distance_errors,
    }
    processed.attrs["coverage"] = coverage
    processed.attrs["provenance"] = bundle.get("provenance", {})
    return processed
