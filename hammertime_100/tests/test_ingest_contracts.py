"""No-network ingestion contracts for FastF1's partial-load behavior."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
from ingest import load_race


def loaded_session():
    return SimpleNamespace(
        load=Mock(), event={"EventName": "Observed event"},
        laps=pd.DataFrame({"DriverNumber": ["1"], "LapNumber": [1]}),
        car_data={"1": pd.DataFrame({"Speed": [180.0]})},
    )


class IngestionContractTests(unittest.TestCase):
    def load(self, session):
        fastf1 = SimpleNamespace(get_session=Mock(return_value=session))
        with patch.dict(sys.modules, {"fastf1": fastf1}), patch("ingest.init_cache"), \
                patch("ingest.logger") as logger:
            result = load_race(2023, "Monza", "R")
        return result, logger

    def test_normal_fastf1_return_without_loaded_laps_is_an_error(self):
        source_error = RuntimeError("laps were not loaded")

        class UnloadedSession:
            load = Mock()
            event = {"EventName": "Observed event"}

            @property
            def laps(self):
                raise source_error

        with self.assertRaisesRegex(RuntimeError, "2023.*Monza.*R") as raised:
            self.load(UnloadedSession())
        self.assertIs(raised.exception.__cause__, source_error)

    def test_empty_laps_are_not_success(self):
        session = loaded_session()
        session.laps = pd.DataFrame()
        with self.assertRaisesRegex(RuntimeError, "laps"):
            self.load(session)

    def test_missing_or_empty_car_telemetry_is_not_success(self):
        for telemetry in (None, {}, {"1": pd.DataFrame()}):
            with self.subTest(telemetry=telemetry):
                session = loaded_session()
                session.car_data = telemetry
                with self.assertRaisesRegex(RuntimeError, "telemetry"):
                    self.load(session)

    def test_normal_fastf1_return_without_loaded_telemetry_is_an_error(self):
        session = loaded_session()
        del session.car_data
        with self.assertRaisesRegex(RuntimeError, "telemetry"):
            self.load(session)

    def test_loaded_nonempty_laps_and_telemetry_return_original_session(self):
        session = loaded_session()
        session.car_data["2"] = pd.DataFrame()
        result, logger = self.load(session)
        self.assertIs(result, session)
        session.load.assert_called_once_with(telemetry=True, laps=True, weather=True)
        self.assertTrue(any("Successfully loaded session" in str(call)
                            for call in logger.info.call_args_list))

    def test_explicit_fastf1_failure_is_preserved(self):
        session = loaded_session()
        source_error = OSError("source unavailable")
        session.load.side_effect = source_error
        with self.assertRaises(OSError) as raised:
            self.load(session)
        self.assertIs(raised.exception, source_error)


if __name__ == "__main__":
    unittest.main()
