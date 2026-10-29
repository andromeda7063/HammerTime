"""Unknown observed tyre metadata must not become active invented features."""

import unittest

import numpy as np
import pandas as pd

from test_preprocess_measurements import lap_rows, telemetry
from preprocess import align_driver_telemetry


class TyreMetadataTests(unittest.TestCase):
    def align(self, laps):
        return align_driver_telemetry("AA", telemetry(), laps, sample_hz=1,
                                       lap_length_m=100, mini_sectors=2)

    def test_missing_and_unknown_compound_are_invalid_without_medium_imputation(self):
        for value in (None, np.nan, "UNKNOWN", ""):
            with self.subTest(compound=value):
                laps = lap_rows()
                laps["Compound"] = value
                aligned = self.align(laps)
                self.assertFalse(aligned["IsValid"].any())
                self.assertFalse(aligned["Compound"].eq("MEDIUM").any())

    def test_absent_compound_column_is_not_assumed_medium(self):
        aligned = self.align(lap_rows().drop(columns="Compound"))
        self.assertFalse(aligned["IsValid"].any())
        self.assertTrue(aligned["Compound"].isna().all())

    def test_missing_nonfinite_and_negative_tyre_age_are_invalid(self):
        for value in (None, np.nan, np.inf, -1.0, "unavailable"):
            with self.subTest(tyre_age=value):
                laps = lap_rows()
                laps["TyreLife"] = value
                aligned = self.align(laps)
                self.assertFalse(aligned["IsValid"].any())

    def test_absent_tyre_age_column_is_not_assumed_one(self):
        aligned = self.align(lap_rows().drop(columns="TyreLife"))
        self.assertFalse(aligned["IsValid"].any())
        self.assertTrue(aligned["TyreLife"].isna().all())

    def test_observed_finite_tyre_age_and_known_compound_stay_valid(self):
        laps = lap_rows()
        laps["Compound"] = "soft"
        laps["TyreLife"] = 0.0
        aligned = self.align(laps)
        self.assertTrue(aligned.loc[:1, "IsValid"].all())
        self.assertEqual(aligned.loc[:1, "Compound"].tolist(), ["SOFT", "SOFT"])
        np.testing.assert_allclose(aligned.loc[:1, "TyreLife"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
