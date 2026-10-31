"""Safely import a supplied FastF1 cache into an isolated cache root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_fastf1_cache(source_root: str | Path, http_cache: str | Path,
                        destination_root: str | Path) -> dict:
    """Copy cache files without overwriting any existing file.

    ``source_root`` must contain the FastF1 session hierarchy (for example,
    ``2023/2023-09-03_Italian_Grand_Prix/2023-09-03_Race``). The supplied
    requests-cache database is installed under FastF1's expected filename.
    Existing identical files are skipped. Any differing path aborts before
    any cache content is copied.
    """
    source_root = Path(source_root).resolve()
    http_cache = Path(http_cache).resolve()
    destination_root = Path(destination_root).resolve()
    if not source_root.is_dir() or not http_cache.is_file():
        raise FileNotFoundError("Source cache directory and HTTP cache database must exist.")
    wal_sidecars = [Path(str(http_cache) + suffix) for suffix in ("-wal", "-shm")]
    if any(path.exists() for path in wal_sidecars):
        raise RuntimeError("HTTP cache has SQLite WAL/SHM sidecars; close it and checkpoint before import.")

    sources: dict[Path, Path] = {}
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source cache must not contain symlinks: {path}")
        if path.is_file():
            sources[path.relative_to(source_root)] = path
    if not sources:
        raise ValueError(f"Source cache directory is empty: {source_root}")
    sources[Path("fastf1_http_cache.sqlite")] = http_cache

    hashes = {relative: _sha256(path) for relative, path in sources.items()}
    for relative in sources:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative cache path: {relative}")
        target = destination_root / relative
        parent = target.parent
        while parent != destination_root and parent.is_relative_to(destination_root):
            if parent.is_symlink():
                raise ValueError(f"Destination cache path contains a symlink: {parent}")
            parent = parent.parent
        if target.is_symlink():
            raise ValueError(f"Destination cache file is a symlink: {target}")
    conflicts = [str(relative) for relative, digest in hashes.items()
                 if (destination_root / relative).exists()
                 and (not (destination_root / relative).is_file()
                      or _sha256(destination_root / relative) != digest)]
    if conflicts:
        raise FileExistsError("Cache import conflicts; no files copied: " + ", ".join(conflicts))

    missing = [relative for relative in sources if not (destination_root / relative).exists()]
    if missing:
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fastf1-cache-import-",
                                         dir=destination_root.parent) as stage_name:
            stage = Path(stage_name)
            for relative in missing:
                staged = stage / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sources[relative], staged)
                if _sha256(staged) != hashes[relative]:
                    raise IOError(f"Hash mismatch while staging cache file: {relative}")
            destination_root.mkdir(parents=True, exist_ok=True)
            for relative in missing:
                destination = destination_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(stage / relative, destination)
                except FileExistsError:
                    if not destination.is_file() or _sha256(destination) != hashes[relative]:
                        raise FileExistsError(f"Cache file changed during import: {relative}")

    return {
        "destination": str(destination_root),
        "copied": [str(path) for path in missing],
        "skipped_identical": [str(path) for path in sources if path not in missing],
        "sha256": {str(path): digest for path, digest in hashes.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Extracted FastF1 cache root")
    parser.add_argument("http_cache", type=Path, help="Supplied fastf1_http_cache.sqlite")
    parser.add_argument("destination_root", type=Path, help="Isolated FastF1 cache directory")
    args = parser.parse_args()
    print(json.dumps(import_fastf1_cache(args.source_root, args.http_cache,
                                         args.destination_root), indent=2))


if __name__ == "__main__":
    main()
