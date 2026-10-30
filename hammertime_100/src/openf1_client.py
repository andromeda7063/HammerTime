"""Acquisition client for historical OpenF1 session data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
import requests


class OpenF1AcquisitionError(RuntimeError):
    """Raised when required OpenF1 data cannot be acquired reliably."""


class OpenF1UnavailableError(OpenF1AcquisitionError):
    """Raised when a transient HTTP or network failure exhausts its retries."""


class _OpenF1Client:
    def __init__(self, section: dict, cache_dir: Path, *, get: Callable,
                 sleeper: Callable, clock: Callable):
        self.base_url = section.get("base_url", "https://api.openf1.org/v1").rstrip("/")
        self.timeout = section.get("timeout_sec", 30)
        self.max_attempts = section.get("max_attempts", 3)
        self.interval = section.get("request_interval_sec", 2.1)
        self.get = get
        self.sleep = sleeper
        self.clock = clock
        self.last_request: float | None = None
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_artifacts: list[str] = []
        self.provenance: dict[str, dict] = {}

    @staticmethod
    def _canonical_query(params: dict) -> dict:
        return {key: str(value) for key, value in sorted(params.items())}

    def _pace(self) -> None:
        if self.last_request is not None:
            remaining = self.interval - (self.clock() - self.last_request)
            if remaining > 0:
                self.sleep(remaining)
        self.last_request = self.clock()

    @staticmethod
    def _retry_after(response) -> float | None:
        value = response.headers.get("Retry-After") if response is not None else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def fetch(self, endpoint: str, params: dict) -> list[dict]:
        query = self._canonical_query(params)
        query_json = json.dumps(query, sort_keys=True, separators=(",", ":"))
        url = f"{self.base_url}/{endpoint}"
        key = hashlib.sha256(f"{url}?{query_json}".encode()).hexdigest()
        response_path = self.cache_dir / f"{key}.json"
        meta_path = self.cache_dir / f"{key}.meta.json"
        if response_path.exists() and meta_path.exists():
            try:
                raw = response_path.read_bytes()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                digest = hashlib.sha256(raw).hexdigest()
                valid_success = meta.get("status") == 200
                valid_empty = (meta.get("status") == 404 and b"no results found" in raw.lower())
                if (meta.get("query") == query and meta.get("endpoint_url") == url
                        and (valid_success or valid_empty)
                        and meta.get("sha256") == digest):
                    data = [] if valid_empty else json.loads(raw)
                    if isinstance(data, list) and all(isinstance(row, dict) for row in data):
                        self.raw_artifacts.append(str(response_path.resolve()))
                        self.provenance[key] = meta
                        return data
            except (OSError, ValueError, TypeError):
                pass  # Corrupt or mismatched cache is fetched again.

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            response = None
            try:
                self._pace()
                response = self.get(url, params=query, timeout=self.timeout)
                status = response.status_code
                if status == 404 and "no results found" in response.text.lower():
                    data: list[dict] = []
                    raw = response.content or b"[]"
                    self._save(response_path, meta_path, raw, query, status, data, url)
                    self.raw_artifacts.append(str(response_path.resolve()))
                    return data
                if status == 200:
                    raw = response.content
                    try:
                        data = response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        raise OpenF1AcquisitionError(f"Invalid JSON from OpenF1 {endpoint}: {exc}") from exc
                    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                        raise OpenF1AcquisitionError(f"OpenF1 {endpoint} response must be a JSON array of objects.")
                    self._save(response_path, meta_path, raw, query, status, data, url)
                    self.raw_artifacts.append(str(response_path.resolve()))
                    return data
                if status == 429 or 500 <= status <= 599:
                    last_error = OpenF1AcquisitionError(
                        f"OpenF1 {endpoint} returned HTTP {status} for query {query!r}."
                    )
                    if attempt + 1 < self.max_attempts:
                        retry_after = self._retry_after(response)
                        self.sleep(retry_after if retry_after is not None else min(2 ** attempt, 8))
                        continue
                else:
                    raise OpenF1AcquisitionError(
                        f"OpenF1 {endpoint} returned HTTP {status} for query {query!r}: {response.text[:300]}"
                    )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    self.sleep(min(2 ** attempt, 8))
                    continue
            except OpenF1AcquisitionError:
                raise
            except requests.RequestException as exc:
                raise OpenF1AcquisitionError(f"OpenF1 request failed for {endpoint}: {exc}") from exc
            break
        raise OpenF1UnavailableError(
            f"OpenF1 request failed for {endpoint} query {query!r} after "
            f"{self.max_attempts} attempts: {last_error}"
        )

    def _save(self, response_path: Path, meta_path: Path, raw: bytes, query: dict,
              status: int, data: list[dict], endpoint_url: str) -> None:
        digest = hashlib.sha256(raw).hexdigest()
        meta = {"query": query, "endpoint_url": endpoint_url, "status": status, "sha256": digest,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "rowcount": len(data)}
        self._atomic_write(response_path, raw)
        self._atomic_write(meta_path, json.dumps(meta, sort_keys=True).encode("utf-8"))
        self.provenance[response_path.stem] = meta

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def _single(rows: list[dict], label: str) -> dict:
    if not rows:
        raise OpenF1AcquisitionError(f"No OpenF1 {label} metadata matched the configured session.")
    if len(rows) != 1:
        raise OpenF1AcquisitionError(f"Expected one OpenF1 {label} metadata row, received {len(rows)}.")
    return rows[0]


def _assert_session_rows(rows: list[dict], endpoint: str, expected_key: str) -> None:
    for index, row in enumerate(rows):
        if "session_key" in row and str(row["session_key"]) != expected_key:
            raise OpenF1AcquisitionError(
                f"OpenF1 {endpoint} row {index} has session_key={row['session_key']!r}; "
                f"expected pinned session_key={expected_key}."
            )


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def load_openf1_tables(config: dict) -> dict:
    """Load a configured historical session and its required OpenF1 tables.

    ``config`` uses the existing ``session`` mapping and an ``openf1`` mapping.
    All responses are cached under ``session.cache_dir/openf1``.
    """
    try:
        session_cfg = config["session"]
        section = config.get("openf1", {})
        session_key = session_cfg.get("session_key")
        if session_key is None:
            raise ValueError("session.session_key is required for OpenF1 acquisition")
        for name, value, minimum in (
            ("timeout_sec", section.get("timeout_sec", 30), 0),
            ("max_attempts", section.get("max_attempts", 3), 1),
            ("chunk_seconds", section.get("chunk_seconds", 300), 1),
            ("request_interval_sec", section.get("request_interval_sec", 2.1), 0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
                raise ValueError(f"openf1.{name} must be a number >= {minimum}")
        if not isinstance(section.get("fetch_location", True), bool):
            raise ValueError("openf1.fetch_location must be boolean")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Configuration requires session mapping and OpenF1 settings.") from exc

    # The configured session identity is pinned before any network access.
    from config_validation import validate_session_source
    validate_session_source(session_cfg, section)

    session_cache = Path(session_cfg["cache_dir"]).expanduser()
    if not session_cache.is_absolute():
        session_cache = Path.cwd() / session_cache
    client = _OpenF1Client(section, session_cache / "openf1", get=requests.get,
                           sleeper=time.sleep, clock=time.monotonic)
    key = str(session_key)
    session_rows = client.fetch("sessions", {"session_key": key})
    _assert_session_rows(session_rows, "sessions", key)
    session_meta = _single(session_rows, "session")
    response_key = session_meta.get("session_key")
    if str(response_key) != key:
        raise OpenF1AcquisitionError(f"OpenF1 returned session_key={response_key!r} for pinned session_key={key}.")
    if str(session_meta.get("year")) != str(session_cfg["year"]):
        raise OpenF1AcquisitionError(
            f"OpenF1 session_key {key} has year={session_meta.get('year')}, expected {session_cfg['year']}."
        )
    if str(session_meta.get("location", "")).casefold() != str(session_cfg["gp"]).casefold():
        raise OpenF1AcquisitionError(
            f"OpenF1 session_key {key} has location={session_meta.get('location')!r}, expected {session_cfg['gp']!r}."
        )
    expected_type = {"R": "Race", "Q": "Qualifying", "S": "Sprint", "SQ": "Sprint Qualifying",
                     "SS": "Sprint Shootout"}.get(str(session_cfg["session_type"]).upper(),
                                                    str(session_cfg["session_type"]))
    if str(session_meta.get("session_name", "")).casefold() != expected_type.casefold():
        raise OpenF1AcquisitionError(
            f"OpenF1 session_key {key} has session_name={session_meta.get('session_name')!r}, "
            f"expected {expected_type!r}."
        )

    tables: dict[str, pd.DataFrame] = {"session": pd.DataFrame(session_rows)}
    optional_failures: list[dict] = []
    endpoint_params = {
        "drivers": {"session_key": key}, "laps": {"session_key": key},
        "stints": {"session_key": key}, "pit": {"session_key": key},
        "race_control": {"session_key": key},
    }
    for endpoint, params in endpoint_params.items():
        try:
            rows = client.fetch(endpoint, params)
        except OpenF1AcquisitionError as exc:
            if endpoint == "pit" and "HTTP 404" in str(exc):
                rows = []
            else:
                raise
        _assert_session_rows(rows, endpoint, key)
        tables[endpoint] = pd.DataFrame(rows)

    laps = tables["laps"]
    drivers = tables["drivers"]
    required = ("session", "drivers", "laps", "stints", "race_control")
    empty_required = [name for name in required if tables[name].empty]
    if empty_required:
        raise OpenF1AcquisitionError(f"Required OpenF1 tables are empty: {', '.join(empty_required)}.")
    missing_lap_fields = {"driver_number", "lap_duration", "date_start"} - set(laps.columns)
    if missing_lap_fields:
        raise OpenF1AcquisitionError(
            f"OpenF1 laps data is missing fields needed to identify timed laps: {', '.join(sorted(missing_lap_fields))}."
        )
    lap_duration = pd.to_numeric(laps["lap_duration"], errors="coerce")
    lap_start = pd.to_datetime(laps["date_start"], errors="coerce", utc=True)
    timed_lap_mask = (laps["driver_number"].notna() & lap_duration.gt(0)
                      & np.isfinite(lap_duration) & lap_start.notna())
    lap_drivers = sorted({str(value) for value in laps.loc[timed_lap_mask, "driver_number"]})
    if not lap_drivers:
        raise OpenF1AcquisitionError("OpenF1 laps contain no finite, positive timed laps with valid date_start.")
    if "driver_number" not in drivers.columns:
        raise OpenF1AcquisitionError("OpenF1 drivers data has no driver_number field.")
    known_drivers = set(drivers["driver_number"].dropna().astype(str))
    missing_drivers = [number for number in lap_drivers if number not in known_drivers]
    if missing_drivers:
        raise OpenF1AcquisitionError(
            f"OpenF1 driver metadata is missing drivers with laps: {', '.join(missing_drivers)}."
        )
    start = _as_utc(session_meta["date_start"])
    end = _as_utc(session_meta["date_end"])
    if end <= start:
        raise OpenF1AcquisitionError("OpenF1 session date_end must follow date_start.")
    chunk = timedelta(seconds=section.get("chunk_seconds", 300))
    car_rows: list[dict] = []
    location_rows: list[dict] = []
    for driver_number in lap_drivers:
        cursor = start
        while cursor < end:
            upper = min(cursor + chunk, end)
            params = {"session_key": key, "driver_number": driver_number,
                      "date>": _utc_text(cursor - timedelta(microseconds=1)),
                      "date<": _utc_text(upper)}
            rows = client.fetch("car_data", params)
            _assert_session_rows(rows, "car_data", key)
            car_rows.extend(rows)
            cursor = upper
    tables["car_data"] = pd.DataFrame(car_rows)

    if section.get("fetch_weather", False):
        params = {"session_key": key}
        try:
            rows = client.fetch("weather", params)
            _assert_session_rows(rows, "weather", key)
        except OpenF1UnavailableError as exc:
            rows = []
            optional_failures.append({"endpoint": "weather", "query": params, "exception": str(exc)})
        tables["weather"] = pd.DataFrame(rows)
    else:
        tables["weather"] = pd.DataFrame()
    if section.get("fetch_location", True):
        location_unavailable = False
        for driver_number in lap_drivers:
            cursor = start
            while cursor < end:
                upper = min(cursor + chunk, end)
                params = {"session_key": key, "driver_number": driver_number,
                          "date>": _utc_text(cursor - timedelta(microseconds=1)),
                          "date<": _utc_text(upper)}
                try:
                    rows = client.fetch("location", params)
                    _assert_session_rows(rows, "location", key)
                except OpenF1UnavailableError as exc:
                    optional_failures.append({"endpoint": "location", "query": params,
                                              "exception": str(exc)})
                    location_unavailable = True
                    break
                location_rows.extend(rows)
                cursor = upper
            if location_unavailable:
                break
    tables["location"] = pd.DataFrame(location_rows)

    missing_car = [number for number in lap_drivers if tables["car_data"].empty or
                   not tables["car_data"]["driver_number"].astype(str).eq(number).any()]
    if missing_car:
        raise OpenF1AcquisitionError(f"OpenF1 car_data is empty for drivers with laps: {', '.join(missing_car)}.")
    provenance = {**client.provenance, "optional_failures": optional_failures}
    return {"session": session_meta, "tables": tables, "raw_artifacts": client.raw_artifacts,
            "provenance": provenance}
