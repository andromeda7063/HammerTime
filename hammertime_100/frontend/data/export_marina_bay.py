"""Export the archived 2023 Singapore race positions for the static dashboard.

Run from the repository root with:
    uv run --frozen python hammertime_100/frontend/data/export_marina_bay.py

This reads only cached FastF1 source tables and the existing aligned table. It
does not download data, train a model, or run model inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / ".local/research-history/2023-fastf1-season/race_cache"
RAW_DIR = CACHE / "raw_sessions/2023-Singapore-R"
STAGE_DIR = CACHE / "sessions/2023-Singapore-R"
RUN_DIR = ROOT / ".local/research-history/2023-fastf1-season/training/769a76aab164b514cc0a"
OUTPUT_PATH = Path(__file__).resolve().parent / "marina_bay_replay.json"
FRAME_STRIDE = 2
SOURCE_HZ = 4.0
OUTLINE_DRIVER = "55"
OUTLINE_LAP = 46
OFFICIAL_EVENT_URL = "https://www.formula1.com/en/racing/2023/singapore"
LAYOUT_REFERENCE_URL = "https://singaporegp.sg/en/news/22/2022-updates-to-the-marina-bay-street-circuit/"
FASTF1_TELEMETRY_DOCS_URL = "https://github.com/theOehrly/Fast-F1/blob/main/fastf1/core.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seconds(value: object) -> float:
    if pd.isna(value):
        return math.nan
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required archived source is missing: {path}")


def atomic_write(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_PATH.name}.", suffix=".tmp", dir=OUTPUT_PATH.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, OUTPUT_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    raw_manifest_path = RAW_DIR / "raw_manifest.json"
    stage_manifest_path = STAGE_DIR / "stage_manifest.json"
    raw_laps_path = RAW_DIR / "raw_laps.parquet"
    raw_telemetry_path = RAW_DIR / "raw_telemetry.parquet"
    aligned_path = STAGE_DIR / "aligned_telemetry.parquet"
    run_metrics_path = RUN_DIR / "test_metrics.json"
    required = (
        raw_manifest_path, stage_manifest_path, raw_laps_path,
        raw_telemetry_path, aligned_path, run_metrics_path,
    )
    for path in required:
        require_file(path)

    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))

    event = stage_manifest.get("event", {})
    if (event.get("race_id"), event.get("year"), event.get("circuit_id")) != (
        "2023-Singapore-R", 2023, "singapore"
    ):
        raise ValueError("Aligned cache manifest does not identify the 2023 Singapore race.")
    if event.get("lap_length_m") != 4940.0:
        raise ValueError("Singapore cache lap length differs from the documented 4,940 m.")
    if raw_manifest.get("event", {}).get("race_id") != event["race_id"]:
        raise ValueError("Raw and aligned manifests identify different race sessions.")
    for path, manifest, section, filename in (
        (raw_laps_path, raw_manifest, "files", "raw_laps.parquet"),
        (raw_telemetry_path, raw_manifest, "files", "raw_telemetry.parquet"),
        (aligned_path, stage_manifest, "files", "aligned_telemetry.parquet"),
    ):
        record = manifest.get(section, {}).get(filename)
        if not record or sha256(path) != record.get("sha256"):
            raise ValueError(f"Cached source hash does not match its manifest: {path.name}")

    train_ids = run_metrics.get("split", {}).get("train", {}).get("race_ids", [])
    test_ids = run_metrics.get("split", {}).get("test", {}).get("race_ids", [])
    if event["race_id"] not in train_ids or event["race_id"] in test_ids:
        raise ValueError("Singapore must remain a training race, not a held-out test race.")

    laps = pd.read_parquet(raw_laps_path)
    raw_telemetry = pd.read_parquet(raw_telemetry_path)
    aligned = pd.read_parquet(aligned_path)
    if set(aligned["RaceId"].dropna().unique()) != {event["race_id"]}:
        raise ValueError("Aligned table contains unexpected race IDs.")

    # FastF1 raw X/Y coordinates are in tenths of a metre. Convert to metres.
    clean_lap = laps[
        (laps["DriverNumber"].astype(str) == OUTLINE_DRIVER)
        & (pd.to_numeric(laps["LapNumber"], errors="coerce") == OUTLINE_LAP)
        & laps["IsAccurate"].fillna(False).astype(bool)
        & (laps["TrackStatus"].astype(str) == "1")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    ]
    if len(clean_lap) != 1:
        raise ValueError("The selected Sainz lap is missing or ambiguous in the cached lap table.")
    lap = clean_lap.iloc[0]
    lap_start = seconds(lap["LapStartTime"])
    lap_duration = seconds(lap["LapTime"])
    if not math.isfinite(lap_start) or not math.isfinite(lap_duration) or lap_duration <= 0:
        raise ValueError("Selected outline lap has invalid timing metadata.")
    outline_samples = raw_telemetry[
        (raw_telemetry["Driver"].astype(str) == OUTLINE_DRIVER)
        & (raw_telemetry["Source"].astype(str) == "pos")
        & (raw_telemetry["SessionTime"].dt.total_seconds() >= lap_start)
        & (raw_telemetry["SessionTime"].dt.total_seconds() < lap_start + lap_duration)
    ].copy()
    outline_samples = outline_samples.dropna(subset=["X", "Y", "SessionTime"])
    outline_samples = outline_samples.sort_values("SessionTime", kind="stable")
    if len(outline_samples) < 100:
        raise ValueError("Selected clean lap has insufficient original position samples.")
    outline_points = [
        {
            "session_time_sec": round(seconds(row.SessionTime), 6),
            "x_m": round(float(row.X) * 0.1, 3),
            "y_m": round(float(row.Y) * 0.1, 3),
        }
        for row in outline_samples.itertuples(index=False)
    ]
    xs = [point["x_m"] for point in outline_points]
    ys = [point["y_m"] for point in outline_points]

    # The replay deliberately uses the existing 4 Hz aligned cache. Position
    # presence is based on finite X/Y, not the stricter model-window IsValid flag.
    driver_ids = sorted(aligned["Driver"].astype(str).unique(), key=lambda item: int(item))
    if len(driver_ids) != 19:
        raise ValueError(f"Expected 19 Singapore race drivers, found {len(driver_ids)}.")
    expected_grid_rows = 0
    expected_grid_times = None
    for driver_id, group in aligned.groupby(aligned["Driver"].astype(str), sort=False):
        times = group["TimeSec"].astype(float).sort_values().to_numpy()
        if len(times) != 25942 or len(set(times)) != len(times):
            raise ValueError(f"Unexpected aligned timestamp coverage for driver {driver_id}.")
        if expected_grid_times is None:
            expected_grid_times = times
        elif not (times == expected_grid_times).all():
            raise ValueError("Aligned drivers do not share the archived common time grid.")
        expected_grid_rows += len(times)
    if expected_grid_rows != len(aligned):
        raise ValueError("Aligned table row count differs from one row per driver per timestamp.")
    tick_deltas = pd.Series(expected_grid_times).diff().dropna().to_numpy()
    if not (abs(tick_deltas - 1.0 / SOURCE_HZ) < 1e-7).all():
        raise ValueError("Aligned cache is not the expected uniform 4 Hz grid.")

    first_race_laps = laps[pd.to_numeric(laps["LapNumber"], errors="coerce") == 1]
    race_start = min(seconds(value) for value in first_race_laps["LapStartTime"].dropna())
    race_date = pd.to_datetime(first_race_laps["LapStartDate"], errors="coerce").min()
    if pd.isna(race_date):
        raise ValueError("Lap-one source rows do not contain a race date.")
    if abs(float(expected_grid_times[0]) - race_start) > 1e-6:
        raise ValueError("Race-relative origin no longer matches the aligned grid start.")

    descriptor_rows = laps.dropna(subset=["DriverNumber"]).copy()
    descriptor_rows["DriverNumber"] = descriptor_rows["DriverNumber"].astype(str)
    descriptors = {}
    for _, row in descriptor_rows.sort_values("LapNumber").iterrows():
        number = row["DriverNumber"]
        descriptors.setdefault(number, {
            "number": number,
            "acronym": str(row["Driver"]),
            "team": str(row["Team"]) if pd.notna(row["Team"]) else None,
        })
    drivers = [descriptors[number] for number in driver_ids]
    if any(driver["number"] not in descriptors for driver in drivers):
        raise ValueError("A replay driver lacks source lap metadata.")

    aligned = aligned.copy()
    aligned["Driver"] = aligned["Driver"].astype(str)
    selected_times = set(expected_grid_times[::FRAME_STRIDE])
    sampled = aligned[aligned["TimeSec"].isin(selected_times)].copy()
    sampled = sampled.sort_values(["TimeSec", "Driver"], key=lambda col: col.map(int) if col.name == "Driver" else col, kind="stable")
    positions_by_time: dict[float, list[dict | None]] = {}
    driver_index = {driver_id: index for index, driver_id in enumerate(driver_ids)}
    for row in sampled.itertuples(index=False):
        time_sec = float(row.TimeSec)
        positions = positions_by_time.setdefault(time_sec, [None] * len(driver_ids))
        x_m = finite(row.X)
        y_m = finite(row.Y)
        if x_m is None or y_m is None:
            continue
        speed = finite(row.Speed)
        lap_number = finite(row.LapNumber)
        if pd.isna(row.IsPitLap):
            raise ValueError(f"Aligned IsPitLap is missing for #{row.Driver} at {time_sec} s.")
        positions[driver_index[str(row.Driver)]] = {
            "x_m": round(x_m * 0.1, 3),
            "y_m": round(y_m * 0.1, 3),
            "lap_number": int(lap_number) if lap_number is not None else None,
            "speed_kph": round(speed, 1) if speed is not None else None,
            "is_pit_lap": bool(row.IsPitLap),
        }
    frames = [
        {
            "session_time_sec": round(time_sec, 3),
            "race_elapsed_sec": round(time_sec - race_start, 3),
            "positions": positions_by_time[time_sec],
        }
        for time_sec in sorted(positions_by_time)
    ]
    if len(frames) != math.ceil(len(expected_grid_times) / FRAME_STRIDE):
        raise ValueError("Replay frame count does not match deterministic stride sampling.")

    source_hashes = {
        "raw_manifest_sha256": sha256(raw_manifest_path),
        "stage_manifest_sha256": sha256(stage_manifest_path),
        "raw_laps_sha256": sha256(raw_laps_path),
        "raw_telemetry_sha256": sha256(raw_telemetry_path),
        "aligned_telemetry_sha256": sha256(aligned_path),
    }
    payload = {
        "schema_version": 1,
        "event": {
            "year": 2023,
            "date": race_date.date().isoformat(),
            "grand_prix": "Singapore Grand Prix",
            "venue": "Marina Bay Street Circuit",
            "session_type": "Race",
            "race_id": "2023-Singapore-R",
            "round_number": 15,
            "circuit_id": "singapore",
            "lap_length_m": 4940,
            "turn_count": 19,
            "lap_count": 62,
            "race_laps": 62,
            "official_url": OFFICIAL_EVENT_URL,
            "official_circuit_url": OFFICIAL_EVENT_URL,
            "layout_reference_url": LAYOUT_REFERENCE_URL,
            "metadata_note": "2023 Marina Bay layout: 19 turns, 4.94 km, 62 laps.",
        },
        "evaluation_context": {
            "season_run_id": "769a76aab164b514cc0a",
            "split_role": "training race",
            "held_out_test": False,
            "note": "This replay is historical training-race telemetry, not a held-out evaluation or model prediction.",
        },
        "provenance": {
            "provider": "FastF1 archived raw session and project-aligned telemetry cache",
            "raw_cache_fingerprint": raw_manifest.get("fingerprint"),
            "aligned_cache_fingerprint": stage_manifest["fingerprint"],
            "source_sha256": source_hashes,
            "split_evidence": {
                "path": ".local/research-history/2023-fastf1-season/training/769a76aab164b514cc0a/test_metrics.json",
                "sha256": sha256(run_metrics_path),
                "training_race_ids_include_this_event": True,
                "held_out_test_race_ids_exclude_this_event": True,
            },
        },
        "coordinate_system": {
            "source_fields": ["X", "Y"],
            "source_unit": "FastF1 coordinate units (0.1 m)",
            "export_unit": "m",
            "conversion": "exported x_m/y_m = cached X/Y * 0.1",
            "source_documentation_url": FASTF1_TELEMETRY_DOCS_URL,
            "aspect_ratio": "Use equal x/y scale when drawing; source coordinates are not a projected official circuit map.",
        },
        "time_axis": {
            "source_field": "TimeSec (aligned) / SessionTime (raw outline)",
            "session_time_unit": "seconds since FastF1 session start",
            "race_start_session_sec": round(race_start, 3),
            "race_elapsed_definition": "session_time_sec - minimum LapStartTime for LapNumber 1 across race drivers",
        },
        "drivers": drivers,
        "track_outline": {
            "method": "One driver-lap trace only: original raw Source=pos points within the selected accurate clean lap interval [LapStartTime, LapStartTime + LapTime), sorted by SessionTime. No interpolation or driver concatenation.",
            "trace_note": "Observed lap trace; sparse position sampling leaves small gaps near timing boundaries, so this is not surveyed circuit geometry.",
            "driver_number": OUTLINE_DRIVER,
            "driver_acronym": "SAI",
            "team": "Ferrari",
            "lap_number": OUTLINE_LAP,
            "lap_start_session_sec": round(lap_start, 3),
            "lap_duration_sec": round(lap_duration, 3),
            "source_sample_count": len(outline_points),
            "source_field": "raw FastF1 Source=pos X/Y",
            "coordinate_unit": "m",
            "distance_delta_m": round(float(outline_samples["Distance"].max() - outline_samples["Distance"].min()), 3),
            "points": outline_points,
            "bounds_m": {
                "min_x": round(min(xs), 3), "max_x": round(max(xs), 3),
                "min_y": round(min(ys), 3), "max_y": round(max(ys), 3),
            },
        },
        "replay": {
            "source": "archived aligned_telemetry.parquet",
            "source_fingerprint": stage_manifest["fingerprint"],
            "source_sample_hz": SOURCE_HZ,
            "frame_stride": FRAME_STRIDE,
            "output_sample_hz": SOURCE_HZ / FRAME_STRIDE,
            "frame_count": len(frames),
            "position_row_count": int((aligned["X"].map(finite).notna() & aligned["Y"].map(finite).notna()).sum()),
            "sampled_position_count": sum(value is not None for frame in frames for value in frame["positions"]),
            "position_presence_rule": "Emit position only when this sampled aligned row has finite X and Y; model IsValid is intentionally not used.",
            "missing_position_policy": "Null means no finite cached position at this frame; do not carry the last position forward.",
            "resampling_provenance": "Existing project alignment is a 4 Hz shared grid. X/Y are linearly interpolated only between finite source coordinates when their bracket gap is <=1.0 s, with no extrapolation. This exporter adds only deterministic frame stride sampling.",
            "optional_channels": "lap_number, speed_kph, and is_pit_lap are read from the same aligned row; no model prediction is included.",
            "drivers_order": "Each positions array is indexed by the drivers array order.",
            "frames": frames,
        },
    }
    atomic_write(payload)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"Frames: {len(frames):,}; drivers: {len(drivers)}; sampled positions: {payload['replay']['sampled_position_count']:,}")
    print(f"Outline points: {len(outline_points)}; SHA-256: {sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
