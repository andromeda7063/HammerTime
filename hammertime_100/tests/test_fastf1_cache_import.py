import tempfile
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from fastf1_cache_import import import_fastf1_cache


class FastF1CacheImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        session_file = self.source / "2023" / "event" / "Race" / "car_data.ff1pkl"
        session_file.parent.mkdir(parents=True)
        session_file.write_bytes(b"native cache fixture")
        self.http_cache = self.root / "source-http.sqlite"
        self.http_cache.write_bytes(b"http cache fixture")
        self.destination = self.root / "destination"

    def tearDown(self):
        self.temp.cleanup()

    def test_import_is_idempotent_and_preserves_hashes(self):
        first = import_fastf1_cache(self.source, self.http_cache, self.destination)
        second = import_fastf1_cache(self.source, self.http_cache, self.destination)
        self.assertEqual(len(first["copied"]), 2)
        self.assertEqual(len(second["copied"]), 0)
        self.assertEqual(len(second["skipped_identical"]), 2)
        self.assertEqual(
            (self.destination / "2023/event/Race/car_data.ff1pkl").read_bytes(),
            b"native cache fixture",
        )
        self.assertEqual(
            (self.destination / "fastf1_http_cache.sqlite").read_bytes(),
            b"http cache fixture",
        )

    def test_conflicting_file_aborts_before_other_files_are_copied(self):
        self.destination.mkdir()
        existing = self.destination / "2023/event/Race/car_data.ff1pkl"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"user cache")

        with self.assertRaisesRegex(FileExistsError, "no files copied"):
            import_fastf1_cache(self.source, self.http_cache, self.destination)

        self.assertEqual(existing.read_bytes(), b"user cache")
        self.assertFalse((self.destination / "fastf1_http_cache.sqlite").exists())

    def test_identical_existing_file_is_never_replaced(self):
        self.destination.mkdir()
        existing = self.destination / "2023/event/Race/car_data.ff1pkl"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"native cache fixture")
        before = existing.stat().st_ino

        result = import_fastf1_cache(self.source, self.http_cache, self.destination)

        self.assertEqual(existing.stat().st_ino, before)
        self.assertIn("2023/event/Race/car_data.ff1pkl", result["skipped_identical"])

    def test_sqlite_wal_sidecar_aborts_before_copy(self):
        Path(str(self.http_cache) + "-wal").write_bytes(b"active WAL")
        with self.assertRaisesRegex(RuntimeError, "WAL/SHM sidecars"):
            import_fastf1_cache(self.source, self.http_cache, self.destination)
        self.assertFalse(self.destination.exists())

    def test_destination_symlink_directory_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.destination.mkdir()
        (self.destination / "2023").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "contains a symlink"):
            import_fastf1_cache(self.source, self.http_cache, self.destination)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
