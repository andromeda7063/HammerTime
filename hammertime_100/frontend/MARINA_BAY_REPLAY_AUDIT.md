# Marina Bay archived replay audit

**Date:** 2026-11-02
**Scope:** Independent provenance review and focused headless-Chromium QA of the 2023 Singapore replay. The replay adds archived telemetry context; it does not alter season metrics, split membership, model thresholds, or inference.

## Source evidence

The ignored FastF1 archive contains raw Singapore race telemetry and laps under `.local/research-history/2023-fastf1-season/race_cache/raw_sessions/2023-Singapore-R/`, plus aligned telemetry under `.../sessions/2023-Singapore-R/`. The source manifests identify the 2023 Singapore Grand Prix, round 15, circuit key `singapore`, and 4,940 m lap length. The aligned Parquet hash recorded in the stage manifest is `1dbc689b8b971651d2765538ac7c80949f6d48d2f77ae4789f9d43ed5494ef08`.

The archived aligned table has 492,898 rows: 19 driver slots on one shared 25,942-timestamp grid at 4 Hz. X/Y are finite in 453,289 rows; missing position fields must remain absent in the replay. The exported browser asset takes every other aligned row, yielding 12,971 frames at 2 Hz. Each frame has one stable position slot per driver; null slots are retained, and the renderer omits them without carrying a previous position forward. The payload is 15,665,281 bytes.

The track trace uses 374 raw `Source=pos` samples from an accurate, non-pit Sainz (#55) lap 46, sorted within that single lap. Its start, midpoint, and end coordinates match the selected raw Parquet samples. FastF1 3.8.3’s local `_api.py` documents position X/Y/Z as tenths of a meter; the exporter converts X/Y by 0.1 and the renderer uses equal X/Y scale. This is a trace from recorded positions, not a surveyed road boundary or official projected map. The 2023 Singapore layout metadata is attributed to the official event and Singapore GP circuit-change references in the JSON.

Singapore appears in the saved model run’s training race list and is absent from its held-out test list. The held-out test remains Abu Dhabi, Las Vegas, Mexico, and São Paulo. The replay labels the 2023 archive and training role, separates race-relative elapsed time from raw FastF1 session seconds, and states that it is neither live timing nor browser-side model inference.

The legacy frontend assets are separate: `telemetry.json` contains Italy race data, `race_timeline.json` is a driver lap/gap summary, and the previous Git `TrackMap` implementation drew a hand-authored Monza polyline with generated car offsets and speed animation. Those assets are not used as Singapore positions.

## Verification

Run from the repository root:

```sh
uv run --frozen python hammertime_100/frontend/tools/marina_bay_smoke.py
```

The focused smoke reads the Singapore source Parquet and verifies SHA-256 against the source manifests. It compares 28 deterministic endpoint/random/null-containing frames (467 present positions and 65 null slots) to exact aligned rows, including position coordinates, lap and speed. It checks the one-lap outline against raw position samples at the start, midpoint, and end.

Headless Chromium confirmed:

- The 15.7 MB replay loaded in 131 ms over the local static server. Archived season metrics loaded independently and remained MAE `0.09046 s`, 37,956 held-out targets.
- Frame 1000’s accessible table matched the selected 2 Hz frame for all 19 drivers; 18 drivers had coordinates at that sample.
- Lap jumps followed Sainz #55 when all cars were shown, then followed the selected driver #20 after changing focus.
- Sainz #55 had a real end-of-coverage transition from a recorded coordinate at frame 12,795 to a null slot at frame 12,796. The marker disappeared and the focus/table reported no coordinates; no prior position was carried forward.
- The first playback implementation advanced the button/status but failed to accumulate elapsed time because each animation tick discarded sub-frame deltas. The main UI lead fixed this with a separate replay-time accumulator while keeping positions snapped to recorded 0.5-second samples. Regression checks observed 1.0 replay-second at 1× and 4.5 seconds at 4× over comparable 1.2-second wall-time intervals. Pause held the selected frame; scrub and restart returned to the expected archived samples.
- At 1440 px, 390 px/DPR2, and 320 px, the document had no horizontal overflow and the canvas fit its wrapper. At 390 px the lap-trace start label initially clipped at the right edge; the main UI lead clamped its placement, and the refreshed mobile screenshot shows the full label.
- Blocking only `marina_bay_replay.json` displayed an archived-replay unavailable message with controls disabled while preserving the independent archived evaluation metrics. No synthetic position fallback appeared.

These checks cover the static archive, selected controls, data mapping, and measured widths only. They do not verify every browser or assistive-technology combination, nor do they establish model quality from this training-race replay.

## Screenshots

- Desktop 1440 × 1000: `.local/research-history/2026-11-02/marina-bay-qa/marina-bay-wide.png`
- Mobile 390 CSS px at DPR2: `.local/research-history/2026-11-02/marina-bay-qa/marina-bay-mobile.png`
