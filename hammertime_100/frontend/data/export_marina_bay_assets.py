"""Export a source-checked asset lookup for the Marina Bay replay UI.

Run from the repository root with:
    uv run --frozen python hammertime_100/frontend/data/export_marina_bay_assets.py

The frontend contains team marks only; this file does not imply driver portraits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import fastf1


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parent
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "teams"
RAW_DIR = ROOT / ".local/research-history/2023-fastf1-season/race_cache/raw_sessions/2023-Singapore-R"
STAGE_DIR = ROOT / ".local/research-history/2023-fastf1-season/race_cache/sessions/2023-Singapore-R"
REPLAY_PATH = DATA_DIR / "marina_bay_replay.json"
OUTPUT_PATH = DATA_DIR / "marina_bay_driver_assets.json"

TEAM_ASSETS = {
    "Alpine": "alp.jpg",
    "Mercedes": "amg.jpg",
    "Aston Martin": "amr.jpg",
    "Alfa Romeo": "ar.jpg",
    "McLaren": "mcl.jpg",
    "Red Bull Racing": "rb.jpg",
    "AlphaTauri": "sat.jpg",
    "Ferrari": "sf.jpg",
    "Haas F1 Team": "tgr.jpg",
    "Williams": "wil.jpg",
}
DNS_DRIVER = {"number": "18", "acronym": "STR", "team": None}
TEAM_COLOR_KEYS = {
    "Alpine": "alpine",
    "Mercedes": "mercedes",
    "Aston Martin": "aston martin",
    "Alfa Romeo": "alfa romeo",
    "McLaren": "mclaren",
    "Red Bull Racing": "red bull",
    "AlphaTauri": "alphatauri",
    "Ferrari": "ferrari",
    "Haas F1 Team": "haas",
    "Williams": "williams",
}
FASTF1_PALETTE_URL = "https://github.com/theOehrly/Fast-F1/blob/v3.8.3/fastf1/plotting/constants.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(payload: dict) -> None:
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


def load_2023_team_colors() -> tuple[dict[str, dict[str, str]], Path]:
    constants_path = Path(fastf1.__file__).resolve().parent / "plotting" / "constants.json"
    constants = json.loads(constants_path.read_text(encoding="utf-8"))
    teams = constants["2023"]["teams"]
    palette = {}
    for canonical_name, key in TEAM_COLOR_KEYS.items():
        source = teams.get(key)
        if not source:
            raise ValueError(f"Locked FastF1 2023 palette lacks {canonical_name} ({key}).")
        colors = source.get("colors", {})
        record = {
            "official_hex": colors.get("official"),
            "fastf1_hex": colors.get("fastf1"),
        }
        if any(not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
               for color in record.values()):
            raise ValueError(f"Locked FastF1 2023 colors for {canonical_name} are missing or malformed.")
        palette[canonical_name] = {name: color.lower() for name, color in record.items()}
    return palette, constants_path


def main() -> None:
    required = [
        REPLAY_PATH,
        RAW_DIR / "raw_manifest.json",
        RAW_DIR / "raw_laps.parquet",
        STAGE_DIR / "stage_manifest.json",
        STAGE_DIR / "aligned_telemetry.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Marina Bay source or asset file is missing: " + ", ".join(missing))

    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    raw_manifest = json.loads((RAW_DIR / "raw_manifest.json").read_text(encoding="utf-8"))
    laps = pd.read_parquet(RAW_DIR / "raw_laps.parquet")
    aligned = pd.read_parquet(STAGE_DIR / "aligned_telemetry.parquet")
    replay_drivers = replay["drivers"]
    replay_ids = {str(driver["number"]) for driver in replay_drivers}
    aligned_ids = set(aligned["Driver"].astype(str).unique())
    lap_ids = set(laps["DriverNumber"].dropna().astype(str).unique())
    manifest_ids = set(map(str, raw_manifest["drivers"]))
    if replay_ids != aligned_ids or replay_ids != lap_ids:
        raise ValueError("Replay, aligned telemetry, and cached lap driver IDs do not agree.")
    if manifest_ids != replay_ids | {DNS_DRIVER["number"]}:
        raise ValueError("Raw FastF1 manifest participants do not match the replay and DNS mapping.")
    if DNS_DRIVER["number"] in lap_ids or DNS_DRIVER["number"] in aligned_ids:
        raise ValueError("Driver 18 has source lap/position data; review the DNS entry before exporting.")

    team_colors, constants_path = load_2023_team_colors()

    team_sources = {}
    entries = []
    for driver in replay_drivers:
        number = str(driver["number"])
        acronym = str(driver["acronym"])
        if not driver.get("team") or driver["team"] not in TEAM_ASSETS:
            raise ValueError(f"No known team logo mapping for cached driver #{number} {acronym}.")
        filename = TEAM_ASSETS[driver["team"]]
        path = ASSET_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Mapped team logo does not exist: {path}")
        team_sources[driver["team"]] = {
            "asset": f"assets/teams/{filename}",
            "kind": "team_logo",
            "sha256": sha256(path),
        }
        entries.append({
            "number": number,
            "acronym": acronym,
            "team": driver["team"],
            "official_team_color_hex": team_colors[driver["team"]]["official_hex"],
            "fastf1_team_accent_hex": team_colors[driver["team"]]["fastf1_hex"],
            "replay_available": True,
            "team_logo_asset": f"assets/teams/{filename}",
            "driver_portrait_asset": None,
        })

    entries.append({
        **DNS_DRIVER,
        "replay_available": False,
        "availability_status": "dns_no_cached_lap_or_aligned_position",
        "team_logo_asset": None,
        "driver_portrait_asset": None,
        "availability_note": "Driver number 18 is listed in the archived FastF1 participant manifest but has no raw lap row or aligned position rows in this Singapore race cache; no team image is assigned without an event lap record.",
    })

    payload = {
        "schema_version": 1,
        "event_race_id": "2023-Singapore-R",
        "asset_scope": "The existing frontend contains team logo JPEGs only; no driver portrait/headshot assets are available.",
        "identity_source": {
            "replay_drivers": "marina_bay_replay.json from archived FastF1 2023 Singapore cache",
            "participant_ids": "raw_manifest.json",
            "driver_18_identity": "Session task mapping identifies driver 18 as Stroll; source manifest confirms participant number 18, while lap and aligned caches contain no rows for that number.",
        },
        "provenance": {
            "replay_json_sha256": sha256(REPLAY_PATH),
            "raw_manifest_sha256": sha256(RAW_DIR / "raw_manifest.json"),
            "stage_manifest_sha256": sha256(STAGE_DIR / "stage_manifest.json"),
            "fastf1_version": fastf1.__version__,
            "fastf1_color_constants_sha256": sha256(constants_path),
            "fastf1_color_constants_url": FASTF1_PALETTE_URL,
        },
        "team_colors_2023": team_colors,
        "team_color_roles": {
            "official_hex": "FastF1's official-brand swatch for the 2023 season.",
            "fastf1_hex": "FastF1's plotting accent swatch; use this as the rendered marker color.",
        },
        "team_assets": team_sources,
        "drivers": sorted(entries, key=lambda entry: int(entry["number"])),
    }
    atomic_write(payload)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({OUTPUT_PATH.stat().st_size:,} bytes)")
    print(f"Replay drivers: {len(replay_drivers)}; explicit DNS/no-cache entry: #{DNS_DRIVER['number']} {DNS_DRIVER['acronym']}")
    print(f"SHA-256: {sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
