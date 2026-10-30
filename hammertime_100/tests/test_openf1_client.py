"""Small mocked contract checks for OpenF1 acquisition and caching."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import sys
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openf1_client import OpenF1AcquisitionError, OpenF1UnavailableError, load_openf1_tables


class FakeResponse:
    def __init__(self, data, status=200, headers=None, raw=None, text=None):
        self.status_code = status
        self.data = data
        self.content = raw if raw is not None else json.dumps(data).encode()
        self.text = text if text is not None else self.content.decode(errors="replace")
        self.headers = headers or {}

    def json(self):
        return self.data


class OpenF1ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = {
            "session": {"year": 2023, "gp": "Monza", "session_type": "R", "source": "openf1",
                        "session_key": 42, "cache_dir": self.temp.name},
            "openf1": {"base_url": "https://api.openf1.org/v1", "timeout_sec": 30,
                       "max_attempts": 2, "chunk_seconds": 300, "request_interval_sec": 0,
                       "fetch_location": False},
        }

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def payload(url, params):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "sessions":
            return [{"session_key": 42, "year": 2023, "location": "Monza", "session_name": "Race",
                     "date_start": "2023-09-03T12:00:00+00:00",
                     "date_end": "2023-09-03T12:00:30Z"}]
        if endpoint == "drivers":
            return [{"driver_number": 1, "full_name": "Driver One"},
                    {"driver_number": 22, "full_name": "DNS Driver"},
                    {"driver_number": 23, "full_name": "Invalid Timestamp Driver"}]
        if endpoint == "laps":
            return [{"driver_number": 1, "lap_number": 1, "lap_duration": 90.0,
                     "date_start": "2023-09-03T12:00:00+00:00"},
                    {"driver_number": 22, "lap_number": 1, "lap_duration": None,
                     "date_start": "2023-09-03T12:00:00+00:00"},
                    {"driver_number": 23, "lap_number": 1, "lap_duration": 90.0,
                     "date_start": "not-a-date"}]
        if endpoint in ("stints", "race_control"):
            return [{"session_key": 42, "event": "present"}]
        if endpoint == "pit":
            return []
        if endpoint == "car_data":
            return [{"driver_number": int(params["driver_number"]), "date": params["date>"] , "speed": 200}]
        return []

    def test_fetches_required_data_and_returns_cache_provenance(self):
        car_drivers = []

        def get(url, params, timeout):
            if url.endswith("/pit"):
                return FakeResponse([], status=404, raw=b'{"detail":"No results found"}',
                                    text='{"detail":"No results found"}')
            if url.endswith("/car_data"):
                car_drivers.append(params["driver_number"])
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            result = load_openf1_tables(self.config)
        self.assertEqual(result["session"]["session_key"], 42)
        self.assertTrue({"drivers", "laps", "stints", "race_control", "pit", "car_data"}
                        <= set(result["tables"]))
        self.assertEqual(len(result["tables"]["car_data"]), 1)
        self.assertEqual(car_drivers, ["1"])
        self.assertTrue(result["raw_artifacts"])
        self.assertTrue(all(Path(path).is_absolute() for path in result["raw_artifacts"]))
        self.assertTrue(result["provenance"])
        with patch("openf1_client.requests.get", side_effect=AssertionError("404 empty result should be cached")):
            cached = load_openf1_tables(self.config)
        self.assertTrue(cached["tables"]["pit"].empty)

    def test_retries_timeout_then_caches_success(self):
        calls = []

        def get(url, params, timeout):
            calls.append(url)
            if len(calls) == 1:
                raise requests.Timeout("temporary")
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch("openf1_client.time.sleep"):
            result = load_openf1_tables(self.config)
        self.assertEqual(len(calls), 8)  # seven requests plus one timeout retry
        self.assertEqual(len(result["tables"]["car_data"]), 1)
        with patch("openf1_client.requests.get", side_effect=AssertionError("cache should satisfy query")):
            cached = load_openf1_tables(self.config)
        self.assertTrue(cached["tables"]["car_data"].equals(result["tables"]["car_data"]))

    def test_does_not_cache_server_error_as_complete(self):
        call_count = 0

        def get(url, params, timeout):
            nonlocal call_count
            call_count += 1
            if url.endswith("/sessions"):
                return FakeResponse([], status=503)
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch("openf1_client.time.sleep"):
            with self.assertRaisesRegex(OpenF1AcquisitionError, "503"):
                load_openf1_tables(self.config)
        self.assertFalse(list((Path(self.temp.name) / "openf1").glob("*.json")))
        self.assertEqual(call_count, 2)

    def test_retries_429_using_retry_after(self):
        calls = 0
        waits = []

        def get(url, params, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return FakeResponse([], status=429, headers={"Retry-After": "7"})
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch(
            "openf1_client.time.sleep", side_effect=waits.append
        ):
            load_openf1_tables(self.config)
        self.assertEqual(waits[0], 7.0)

    def test_corrupt_cache_is_refetched(self):
        def get(url, params, timeout):
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            load_openf1_tables(self.config)
        response_file = next((Path(self.temp.name) / "openf1").glob("*.json"))
        response_file.write_text("broken", encoding="utf-8")
        calls = []
        with patch("openf1_client.requests.get", side_effect=lambda *args, **kwargs: (
            calls.append(args[0]) or FakeResponse(self.payload(args[0], kwargs["params"]))
        )):
            load_openf1_tables(self.config)
        self.assertTrue(calls)

    def test_rejects_mismatched_session_identity(self):
        for field, value, message in (("session_key", 41, "session_key"), ("year", 2024, "year"),
                                      ("location", "Spa", "location"),
                                      ("session_name", "Qualifying", "session_name")):
            with self.subTest(field=field):
                isolated = tempfile.TemporaryDirectory()
                config = {**self.config, "session": {**self.config["session"],
                                                      "cache_dir": isolated.name}}

                def get(url, params, timeout):
                    data = self.payload(url, params)
                    if url.endswith("/sessions"):
                        data[0][field] = value
                    return FakeResponse(data)

                with patch("openf1_client.requests.get", side_effect=get):
                    with self.assertRaisesRegex(OpenF1AcquisitionError, message):
                        load_openf1_tables(config)
                isolated.cleanup()

    def test_rejects_wrong_session_key_rows_from_fetched_and_cached_endpoints(self):
        for endpoint in ("drivers", "laps", "stints", "pit", "race_control", "car_data",
                         "location", "weather"):
            with self.subTest(endpoint=endpoint):
                isolated = tempfile.TemporaryDirectory()
                config = {**self.config, "session": {**self.config["session"],
                                                      "cache_dir": isolated.name},
                          "openf1": {**self.config["openf1"],
                                     "fetch_location": endpoint == "location",
                                     "fetch_weather": endpoint == "weather"}}
                hits = 0

                def get(url, params, timeout):
                    nonlocal hits
                    name = url.rsplit("/", 1)[-1]
                    rows = self.payload(url, params)
                    if name == endpoint:
                        hits += 1
                        if rows:
                            rows[0]["session_key"] = 99
                        else:
                            rows = [{"session_key": 99, "driver_number": 1}]
                    return FakeResponse(rows)

                with patch("openf1_client.requests.get", side_effect=get):
                    for _ in range(2):
                        with self.assertRaisesRegex(OpenF1AcquisitionError, "pinned session_key=42"):
                            load_openf1_tables(config)
                self.assertEqual(hits, 1, "second failure should come from the hash-verified cache")
                isolated.cleanup()

    def test_required_data_and_driver_membership_fail_before_telemetry(self):
        for failure in ("empty_stints", "missing_driver"):
            with self.subTest(failure=failure):
                isolated = tempfile.TemporaryDirectory()
                config = {**self.config, "session": {**self.config["session"],
                                                      "cache_dir": isolated.name}}
                telemetry_calls = []

                def get(url, params, timeout):
                    endpoint = url.rsplit("/", 1)[-1]
                    if endpoint in ("car_data", "location", "weather"):
                        telemetry_calls.append(endpoint)
                    if failure == "empty_stints" and endpoint == "stints":
                        return FakeResponse([])
                    if failure == "missing_driver" and endpoint == "laps":
                        return FakeResponse([{"driver_number": 2, "lap_number": 1,
                                              "lap_duration": 90, "date_start": "2023-09-03T12:00:00+00:00"}])
                    return FakeResponse(self.payload(url, params))

                expected_error = "Required OpenF1 tables are empty" if failure == "empty_stints" else "missing drivers"
                with patch("openf1_client.requests.get", side_effect=get):
                    with self.assertRaisesRegex(OpenF1AcquisitionError, expected_error):
                        load_openf1_tables(config)
                self.assertEqual(telemetry_calls, [])
                isolated.cleanup()

    def test_duplicate_session_metadata_is_rejected(self):
        def get(url, params, timeout):
            rows = self.payload(url, params)
            return FakeResponse(rows + rows if url.endswith("/sessions") else rows)

        with patch("openf1_client.requests.get", side_effect=get):
            with self.assertRaisesRegex(OpenF1AcquisitionError, "Expected one OpenF1 session"):
                load_openf1_tables(self.config)

    def test_uses_only_session_section_session_key(self):
        config = {**self.config, "openf1": {**self.config["openf1"], "session_key": 99}}
        seen = []

        def get(url, params, timeout):
            if url.endswith("/sessions"):
                seen.append(params["session_key"])
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            load_openf1_tables(config)
        self.assertEqual(seen, ["42"])

    def test_car_windows_are_half_open_with_one_microsecond_lower_overlap(self):
        config = {**self.config, "openf1": {**self.config["openf1"], "chunk_seconds": 300}}
        requests_seen = []

        def get(url, params, timeout):
            if url.endswith("/sessions"):
                data = self.payload(url, params)
                data[0]["date_end"] = "2023-09-03T12:05:01+00:00"
                return FakeResponse(data)
            if url.endswith("/car_data"):
                requests_seen.append(params)
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            load_openf1_tables(config)
        self.assertEqual(len(requests_seen), 2)
        self.assertEqual(requests_seen[0]["date>"], "2023-09-03T11:59:59.999999+00:00")
        self.assertEqual(requests_seen[0]["date<"], "2023-09-03T12:05:00.000000+00:00")
        self.assertEqual(requests_seen[1]["date>"], "2023-09-03T12:04:59.999999+00:00")

    def test_invalid_json_fails_without_cache(self):
        class InvalidJsonResponse(FakeResponse):
            def json(self):
                raise ValueError("invalid document")

        def get(url, params, timeout):
            return InvalidJsonResponse(None, raw=b"not-json")

        with patch("openf1_client.requests.get", side_effect=get):
            with self.assertRaisesRegex(OpenF1AcquisitionError, "Invalid JSON"):
                load_openf1_tables(self.config)
        self.assertFalse(list((Path(self.temp.name) / "openf1").glob("*.json")))

    def test_empty_car_data_for_lapped_driver_fails(self):
        def get(url, params, timeout):
            if url.endswith("/car_data"):
                return FakeResponse([])
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            with self.assertRaisesRegex(OpenF1AcquisitionError, "car_data is empty"):
                load_openf1_tables(self.config)

    def test_weather_transport_failure_is_recorded_and_required_data_returns(self):
        config = {**self.config, "openf1": {**self.config["openf1"], "fetch_weather": True}}
        weather_calls = 0

        def get(url, params, timeout):
            nonlocal weather_calls
            if url.endswith("/weather"):
                weather_calls += 1
                return FakeResponse([], status=500)
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch("openf1_client.time.sleep"):
            result = load_openf1_tables(config)
        self.assertEqual(weather_calls, 2)
        self.assertTrue(result["tables"]["weather"].empty)
        self.assertFalse(result["tables"]["laps"].empty)
        self.assertEqual(result["provenance"]["optional_failures"][0]["endpoint"], "weather")
        self.assertEqual(result["provenance"]["optional_failures"][0]["query"], {"session_key": "42"})
        self.assertIn("500", result["provenance"]["optional_failures"][0]["exception"])

    def test_location_failure_keeps_observed_rows_and_stops_later_windows(self):
        config = {**self.config,
                  "openf1": {**self.config["openf1"], "fetch_location": True, "chunk_seconds": 300}}
        location_requests = []

        def get(url, params, timeout):
            if url.endswith("/sessions"):
                rows = self.payload(url, params)
                rows[0]["date_end"] = "2023-09-03T12:10:01+00:00"
                return FakeResponse(rows)
            if url.endswith("/location"):
                location_requests.append(dict(params))
                if len(location_requests) == 1:
                    return FakeResponse([{"session_key": 42, "driver_number": 1, "x": 12}])
                return FakeResponse([], status=500)
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch("openf1_client.time.sleep"):
            result = load_openf1_tables(config)
        self.assertEqual(len(location_requests), 3)  # first window; second window retries twice
        self.assertEqual(len(result["tables"]["location"]), 1)
        failure = result["provenance"]["optional_failures"][0]
        self.assertEqual(failure["endpoint"], "location")
        self.assertEqual(failure["query"], location_requests[1])
        self.assertIn("500", failure["exception"])

    def test_required_car_data_transport_failure_still_aborts(self):
        def get(url, params, timeout):
            if url.endswith("/car_data"):
                return FakeResponse([], status=500)
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get), patch("openf1_client.time.sleep"):
            with self.assertRaises(OpenF1UnavailableError):
                load_openf1_tables(self.config)

    def test_optional_session_key_mismatch_remains_fatal(self):
        config = {**self.config,
                  "openf1": {**self.config["openf1"], "fetch_location": True}}

        def get(url, params, timeout):
            if url.endswith("/location"):
                return FakeResponse([{"session_key": 99, "driver_number": 1, "x": 12}])
            return FakeResponse(self.payload(url, params))

        with patch("openf1_client.requests.get", side_effect=get):
            with self.assertRaisesRegex(OpenF1AcquisitionError, "pinned session_key=42") as raised:
                load_openf1_tables(config)
        self.assertNotIsInstance(raised.exception, OpenF1UnavailableError)


if __name__ == "__main__":
    unittest.main()
