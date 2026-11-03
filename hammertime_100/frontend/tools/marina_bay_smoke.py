#!/usr/bin/env python3
"""Focused browser and source checks for the archived Singapore race replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import browser_smoke as smoke


ROOT = Path(__file__).resolve().parents[3]
REPLAY_JSON = smoke.FRONTEND / "data/marina_bay_replay.json"
CACHE = ROOT / ".local/research-history/2023-fastf1-season/race_cache"
MODEL_JSON = smoke.FRONTEND / "data/season_model.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_check(payload: dict) -> dict:
    """Independently compare sampled output frames and lap outline to cached Parquet."""
    aligned_path = CACHE / "sessions/2023-Singapore-R/aligned_telemetry.parquet"
    stage_manifest_path = CACHE / "sessions/2023-Singapore-R/stage_manifest.json"
    raw_manifest_path = CACHE / "raw_sessions/2023-Singapore-R/raw_manifest.json"
    raw_laps_path = CACHE / "raw_sessions/2023-Singapore-R/raw_laps.parquet"
    raw_telemetry_path = CACHE / "raw_sessions/2023-Singapore-R/raw_telemetry.parquet"
    for path in (aligned_path, stage_manifest_path, raw_manifest_path, raw_laps_path, raw_telemetry_path):
        if not path.is_file():
            raise FileNotFoundError(f"Singapore source-check input is missing: {path}")
    stage = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    hashes = payload["provenance"]["source_sha256"]
    assert sha256(stage_manifest_path) == hashes["stage_manifest_sha256"]
    assert sha256(raw_manifest_path) == hashes["raw_manifest_sha256"]
    assert sha256(aligned_path) == hashes["aligned_telemetry_sha256"] == stage["files"]["aligned_telemetry.parquet"]["sha256"]
    assert sha256(raw_laps_path) == hashes["raw_laps_sha256"] == raw_manifest["files"]["raw_laps.parquet"]["sha256"]
    assert sha256(raw_telemetry_path) == hashes["raw_telemetry_sha256"] == raw_manifest["files"]["raw_telemetry.parquet"]["sha256"]
    assert payload["evaluation_context"]["split_role"] == "training race"
    assert payload["evaluation_context"]["held_out_test"] is False

    frames = payload["replay"]["frames"]
    drivers = [driver["number"] for driver in payload["drivers"]]
    rng = np.random.default_rng(20261102)
    selected = {0, 1, 2, 100, len(frames) // 2, len(frames) - 2, len(frames) - 1}
    selected.update(rng.choice(len(frames), size=20, replace=False).tolist())
    first_null = next(index for index, frame in enumerate(frames)
                      if any(position is None for position in frame["positions"]))
    selected.add(first_null)
    selected_times = {round(frames[index]["session_time_sec"], 3) for index in selected}

    aligned = pd.read_parquet(aligned_path, columns=["TimeSec", "Driver", "X", "Y", "LapNumber", "Speed"])
    aligned["TimeKey"] = aligned["TimeSec"].round(3)
    aligned["Driver"] = aligned["Driver"].astype(str)
    sampled = aligned[aligned["TimeKey"].isin(selected_times)]
    source_rows = {(float(row.TimeKey), row.Driver): row for row in sampled.itertuples(index=False)}
    position_checks = null_checks = 0
    for index in sorted(selected):
        frame = frames[index]
        time_key = round(frame["session_time_sec"], 3)
        assert math.isclose(frame["race_elapsed_sec"], frame["session_time_sec"]
                            - payload["time_axis"]["race_start_session_sec"], abs_tol=1e-6)
        for driver_index, driver_id in enumerate(drivers):
            source = source_rows[(time_key, driver_id)]
            point = frame["positions"][driver_index]
            finite_xy = math.isfinite(float(source.X)) and math.isfinite(float(source.Y))
            if not finite_xy:
                assert point is None, (index, driver_id, point)
                null_checks += 1
                continue
            assert point is not None, (index, driver_id)
            assert point["x_m"] == round(float(source.X) * 0.1, 3)
            assert point["y_m"] == round(float(source.Y) * 0.1, 3)
            assert point["lap_number"] == (int(source.LapNumber) if pd.notna(source.LapNumber) else None)
            assert point["speed_kph"] == (round(float(source.Speed), 1) if pd.notna(source.Speed) else None)
            position_checks += 1

    outline = payload["track_outline"]
    laps = pd.read_parquet(raw_laps_path)
    selected_lap = laps[
        (laps["DriverNumber"].astype(str) == outline["driver_number"])
        & (pd.to_numeric(laps["LapNumber"], errors="coerce") == outline["lap_number"])
        & laps["IsAccurate"].fillna(False).astype(bool)
        & (laps["TrackStatus"].astype(str) == "1")
        & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    ]
    assert len(selected_lap) == 1
    lap = selected_lap.iloc[0]
    lap_start = lap["LapStartTime"].total_seconds()
    duration = lap["LapTime"].total_seconds()
    raw = pd.read_parquet(raw_telemetry_path, columns=["Driver", "Source", "SessionTime", "X", "Y"])
    trace = raw[
        (raw["Driver"].astype(str) == outline["driver_number"])
        & (raw["Source"].astype(str) == "pos")
        & (raw["SessionTime"].dt.total_seconds() >= lap_start)
        & (raw["SessionTime"].dt.total_seconds() < lap_start + duration)
    ].dropna(subset=["X", "Y", "SessionTime"]).sort_values("SessionTime", kind="stable")
    points = outline["points"]
    assert len(trace) == outline["source_sample_count"] == len(points)
    for index in (0, len(trace) // 2, len(trace) - 1):
        source = trace.iloc[index]
        point = points[index]
        assert math.isclose(point["session_time_sec"], source["SessionTime"].total_seconds(), abs_tol=1e-6)
        assert point["x_m"] == round(float(source["X"]) * 0.1, 3)
        assert point["y_m"] == round(float(source["Y"]) * 0.1, 3)
    return {"source_frames": len(selected), "position_checks": position_checks,
            "null_checks": null_checks, "outline_points": len(points),
            "outline_driver": outline["driver_acronym"], "outline_lap": outline["lap_number"]}


def evaluate_json(cdp: smoke.DevTools, expression: str) -> dict:
    return json.loads(cdp.evaluate(f"JSON.stringify({expression})"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", default=None, help="Chromium executable (default: chromium)")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=smoke.DEFAULT_ARTIFACTS / "marina-bay-qa",
                        help="directory for desktop/mobile review screenshots")
    args = parser.parse_args()
    executable = shutil.which(args.chromium or "chromium")
    if not executable:
        parser.error("Chromium executable not found")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    replay_payload = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    model_payload = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    expected_mae = f"{model_payload['metrics']['mae_sec']:.5f} s"
    expected_targets = f"{model_payload['metrics']['sample_count']:,}"
    provenance = source_check(replay_payload)
    print(f"PASS independent Singapore parquet reconciliation: {provenance}")

    http_port, debug_port = smoke.free_port(), smoke.free_port()
    base_url = f"http://127.0.0.1:{http_port}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-marina-bay-smoke-") as temp:
        profile = Path(temp) / "profile"
        profile.mkdir()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(http_port), "--bind", "127.0.0.1",
             "--directory", str(smoke.FRONTEND)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        chrome = subprocess.Popen(
            [executable, "--headless", "--no-sandbox", "--disable-gpu", "--disable-extensions",
             "--disable-dev-shm-usage", f"--remote-debugging-port={debug_port}",
             "--remote-allow-origins=*", f"--user-data-dir={profile}", base_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        cdp = None
        try:
            endpoint = f"http://127.0.0.1:{debug_port}/json/list"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    pages = json.load(urllib.request.urlopen(endpoint, timeout=1))
                    page = next(p for p in pages if p.get("type") == "page"
                                and p.get("url", "").startswith(base_url))
                    cdp = smoke.DevTools(page["webSocketDebuggerUrl"])
                    break
                except (OSError, ValueError, StopIteration):
                    time.sleep(0.2)
            if cdp is None:
                raise TimeoutError("Chromium did not expose the dashboard via DevTools")
            cdp.command("Runtime.enable")
            cdp.command("Page.enable")
            smoke.wait_for_page(cdp,
                "document.querySelector('#loadStatus')?.textContent==='Run loaded · archived metrics'")
            print("PASS Singapore in-sample metrics loaded while the independent archived replay request was running")
            smoke.wait_for_page(cdp,
                "document.querySelector('#trackReplayStatus')?.textContent.startsWith('Archived replay ready')",
                timeout=40)

            initial = evaluate_json(cdp,
                "(() => ({status:trackReplayStatus.textContent,error:trackReplayError.hidden,"
                "resource:(()=>{const e=performance.getEntriesByType('resource').find(x=>x.name.endsWith('/data/marina_bay_replay.json'));"
                "return e?{size:e.encodedBodySize,duration:e.duration,transfer:e.transferSize}:null})(),"
                "modelStatus:loadStatus.textContent,mae:metricMae.textContent,targets:metricN.textContent,"
                "facts:trackCircuitFacts.textContent,context:document.querySelector('.track-replay-context').textContent,"
                "outline:trackOutlineSource.textContent,lap:trackCurrentLap.textContent,"
                "clock:trackReplayClock.textContent,coverage:trackPositionCoverage.textContent,"
                "aria:marinaBayCanvas.getAttribute('aria-label'),controls:[replayDriverFilter,replayLapFilter,replayPlayPause,replayReset,replayScrubber].map(x=>x.disabled),"
                "rows:trackFrameRows.querySelectorAll('tr').length,bodyWidth:document.body.scrollWidth,viewport:innerWidth}))()")
            assert initial["status"].startswith("Archived replay ready") and initial["error"]
            assert initial["modelStatus"] == "Run loaded · archived metrics" and initial["mae"] == expected_mae
            assert initial["targets"] == expected_targets and "4.94 km" in initial["facts"] and "19 turns" in initial["facts"]
            assert "training split" in initial["context"] and "not its held-out evaluation" in initial["context"]
            assert "does not run the model" in initial["context"] and "represent live timing" in initial["context"]
            assert "Source=pos" in initial["outline"] and "not a surveyed boundary" in initial["outline"]
            assert initial["clock"] == "Race elapsed 00:00:00" and initial["coverage"] in ("Positions 19/19", "Markers 19/19")
            assert initial["rows"] == 19 and initial["bodyWidth"] <= initial["viewport"] + 1
            assert initial["controls"] == [False] * 5
            assert "archived 2023 singapore" in initial["aria"].lower()
            assert initial["resource"] is not None, initial
            if initial["resource"]["size"]:
                assert initial["resource"]["size"] == REPLAY_JSON.stat().st_size
            print(f"PASS 2023 archived replay ready: 2 Hz, asset {initial['resource']['size']:,} bytes, "
                  f"fetch {initial['resource']['duration']:.0f} ms; "
                  f"in-sample metrics remain available ({initial['mae']}, N={initial['targets']})")

            # The chart uses 19 stable driver slots; every table row must match
            # the selected recorded frame, including explicit missing positions.
            frame_case = evaluate_json(cdp,
                "(() => { const i=1000,f=replayState.data.replay.frames[i]; setReplayFrame(i); "
                "return {index:replayState.frameIndex,elapsed:f.race_elapsed_sec,source:f.session_time_sec,"
                "positions:f.positions,drivers:replayState.data.drivers.map(d=>d.number),"
                "coverage:trackPositionCoverage.textContent,caption:trackFrameCaption.textContent}; })()")
            expected_count = sum(position is not None for position in frame_case["positions"])
            assert frame_case["index"] == 1000 and frame_case["coverage"] in (f"Positions {expected_count}/19", f"Markers {expected_count}/19")
            assert "source session timestamp" in frame_case["caption"]
            # Compare visible table rows in browser memory against selected frame values.
            table_case = evaluate_json(cdp,
                "([...trackFrameRows.querySelectorAll('tr')].map(row=>({driver:row.cells[0].textContent,"
                "lap:row.cells[1].textContent,x:row.cells[2].textContent,y:row.cells[3].textContent,"
                "speed:row.cells[4].textContent,status:row.cells[5].textContent})))")
            assert len(table_case) == 19
            for i, (driver, position, row) in enumerate(zip(replay_payload["drivers"], frame_case["positions"], table_case)):
                assert row["driver"].endswith(f"#{driver['number']}")
                if position is None:
                    assert row["status"] == "No coordinates at sample" and row["x"] == "—" and row["y"] == "—"
                else:
                    assert row["status"] == "Coordinates recorded"
                    assert row["x"] == f"{position['x_m']:.3f}" and row["y"] == f"{position['y_m']:.3f}"
                    assert row["lap"] == (str(position["lap_number"]) if position["lap_number"] is not None else "—")
            print(f"PASS frame 1000 table rows match recorded frame ({expected_count}/19 coordinates)")

            # Lap jumps use the stated reference car for All, then the selected driver.
            ref = json.loads(cdp.evaluate("JSON.stringify({hint:replayLapReference.textContent,title:replayLapFilter.title,"
                                             "options:[...replayLapFilter.options].map(o=>({label:o.textContent,value:o.value}))})"))
            assert "#55" in ref["hint"] and "#55" in ref["title"]
            lap55 = next(option for option in ref["options"] if option["label"] == "Lap 46")
            cdp.evaluate(f"(() => {{ replayLapFilter.value={json.dumps(lap55['value'])}; replayLapFilter.dispatchEvent(new Event('change')); }})()")
            assert cdp.evaluate("replayState.data.replay.frames[replayState.frameIndex].positions[replayState.data.drivers.findIndex(d=>d.number==='55')].lap_number===46")
            cdp.evaluate("(() => { replayDriverFilter.value='20'; replayDriverFilter.dispatchEvent(new Event('change')); })()")
            wait = smoke.wait_for_page(cdp, "replayState.focusDriver==='20'&&replayLapReference.textContent.includes('#20')")
            driver_laps = json.loads(cdp.evaluate("JSON.stringify([...replayLapFilter.options].map(o=>({label:o.textContent,value:o.value})))"))
            lap20 = next(option for option in driver_laps if option["label"] == "Lap 36")
            cdp.evaluate(f"(() => {{ replayLapFilter.value={json.dumps(lap20['value'])}; replayLapFilter.dispatchEvent(new Event('change')); }})()")
            assert cdp.evaluate("replayState.data.replay.frames[replayState.frameIndex].positions[replayState.data.drivers.findIndex(d=>d.number==='20')].lap_number===36")
            print("PASS lap jump follows #55 with all cars, then selected driver #20")

            # Driver 55 has a real telemetry coverage end: position at frame 12795,
            # then null at 12796. The UI must not carry that marker forward.
            cdp.evaluate("(() => { replayDriverFilter.value='55'; replayDriverFilter.dispatchEvent(new Event('change')); })()")
            cdp.evaluate("(() => { replayScrubber.value='6397.5'; replayScrubber.dispatchEvent(new Event('input')); })()")
            prior = evaluate_json(cdp,
                "({frame:replayState.frameIndex,positionStatus:replayFocusStatus.textContent,coverage:trackPositionCoverage.textContent})")
            cdp.evaluate("(() => { replayScrubber.value='6398'; replayScrubber.dispatchEvent(new Event('input')); })()")
            missing = evaluate_json(cdp,
                "(() => { const row=[...trackFrameRows.rows].find(r=>r.cells[0].textContent.endsWith('#55')); "
                "return {frame:replayState.frameIndex,status:replayFocusStatus.textContent,coverage:trackPositionCoverage.textContent,"
                "rowStatus:row.cells[5].textContent,x:row.cells[2].textContent,y:row.cells[3].textContent,lap:trackCurrentLap.textContent}; })()")
            assert prior["positionStatus"].startswith("Recorded position")
            assert missing["status"] == "No recorded coordinates at this sample"
            assert missing["rowStatus"] == "No coordinates at sample" and missing["x"] == "—" and missing["y"] == "—"
            expected_missing_count = sum(position is not None for position in replay_payload["replay"]["frames"][12796]["positions"])
            assert missing["coverage"] in (f"Positions {expected_missing_count}/19", f"Markers {expected_missing_count}/19")
            print(f"PASS missing-position suppression: #55 frame {prior['frame']} recorded → frame {missing['frame']} absent")

            # Transport, speed controls, scrubber and restart.
            cdp.evaluate("document.getElementById('replayReset').click()")
            assert cdp.evaluate("replayState.frameIndex===0&&!replayState.playing")
            cdp.evaluate("document.querySelector('[data-replay-speed=\"4\"]').click()")
            assert cdp.evaluate("document.querySelector('[data-replay-speed=\"4\"]').getAttribute('aria-pressed')==='true'")
            cdp.evaluate("document.querySelector('[data-replay-speed=\"12\"]').click()")
            assert cdp.evaluate("document.querySelector('[data-replay-speed=\"12\"]').getAttribute('aria-pressed')==='true'")
            cdp.evaluate("document.querySelector('[data-replay-speed=\"1\"]').click()")
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            assert cdp.evaluate("replayState.playing&&replayPlayPause.textContent==='Pause'")
            cdp.evaluate("new Promise(resolve=>setTimeout(resolve,1200))")
            elapsed_1x = float(cdp.evaluate("replayState.data.replay.frames[replayState.frameIndex].race_elapsed_sec"))
            assert elapsed_1x > 0, elapsed_1x
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            assert cdp.evaluate("!replayState.playing&&replayPlayPause.textContent==='Play'")
            paused_at = int(cdp.evaluate("replayState.frameIndex"))
            cdp.evaluate("new Promise(resolve=>setTimeout(resolve,350))")
            assert int(cdp.evaluate("replayState.frameIndex")) == paused_at
            cdp.evaluate("document.getElementById('replayReset').click()")
            cdp.evaluate("document.querySelector('[data-replay-speed=\"4\"]').click()")
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            cdp.evaluate("new Promise(resolve=>setTimeout(resolve,1200))")
            elapsed_4x = float(cdp.evaluate("replayState.data.replay.frames[replayState.frameIndex].race_elapsed_sec"))
            assert elapsed_4x > elapsed_1x, (elapsed_1x, elapsed_4x)
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            assert not cdp.evaluate("replayState.playing")
            cdp.evaluate("(() => { replayScrubber.value='1000.5'; replayScrubber.dispatchEvent(new Event('input')); })()")
            scrubbed = evaluate_json(cdp,
                "({index:replayState.frameIndex,elapsed:replayState.data.replay.frames[replayState.frameIndex].race_elapsed_sec,"
                "clock:trackReplayClock.textContent,playing:replayState.playing})")
            assert scrubbed["elapsed"] == 1000.5 and not scrubbed["playing"]
            assert scrubbed["clock"] == "Race elapsed 00:16:40"
            cdp.evaluate("document.getElementById('replayReset').click()")
            assert cdp.evaluate("replayState.frameIndex===0&&!replayState.playing&&trackReplayClock.textContent==='Race elapsed 00:00:00'")
            print(f"PASS play/pause, speed pressed states, scrub and restart; 1×={elapsed_1x:.1f}s, 4×={elapsed_4x:.1f}s; scrub={scrubbed}")

            # Responsive layout and screenshots. Keep all cars at first frame.
            cdp.evaluate("(() => { replayDriverFilter.value='all'; replayDriverFilter.dispatchEvent(new Event('change')); document.getElementById('replayReset').click(); })()")
            for width, height, dpr, mobile, name in [
                (1440, 1000, 1, False, "marina-bay-wide.png"),
                (390, 844, 2, True, "marina-bay-mobile.png"),
                (320, 740, 1, True, None),
            ]:
                cdp.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": dpr, "mobile": mobile,
                })
                cdp.evaluate("(() => { document.documentElement.style.scrollBehavior='auto'; const e=document.getElementById('track-replay'); "
                             "window.scrollTo(0,Math.max(0,window.scrollY+e.getBoundingClientRect().top-90)); })()")
                cdp.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
                layout = evaluate_json(cdp,
                    "(() => { const c=marinaBayCanvas.getBoundingClientRect(),w=document.body.scrollWidth,v=innerWidth; "
                    "return {body:w,viewport:v,canvasWidth:c.width,canvasHeight:c.height,canvasRight:c.right,"
                    "wrapWidth:document.querySelector('.track-canvas-wrap').clientWidth,"
                    "status:trackReplayStatus.textContent,clock:trackReplayClock.textContent,coverage:trackPositionCoverage.textContent}; })()")
                assert layout["body"] <= layout["viewport"] + 1, (width, layout)
                assert layout["canvasWidth"] <= layout["viewport"] and layout["canvasHeight"] >= 180, (width, layout)
                if width == 320:
                    assert layout["canvasWidth"] > 0 and layout["wrapWidth"] <= width
                if name:
                    cdp.screenshot(args.artifacts_dir / name)
                print(f"PASS responsive replay layout {width}px DPR{dpr}: {layout}")

            # Blocking only replay JSON must preserve the current season metrics
            # and show unavailable instead of any synthetic position fallback.
            cdp.command("Network.enable")
            cdp.command("Network.setBlockedURLs", {"urls": ["*marina_bay_replay.json*"]})
            cdp.command("Page.reload", {"ignoreCache": True})
            smoke.wait_for_page(cdp,
                "loadStatus.textContent==='Run loaded · archived metrics'&&trackReplayStatus.textContent==='Archived replay unavailable'",
                timeout=20)
            failure = evaluate_json(cdp,
                "({metric:metricMae.textContent,n:metricN.textContent,status:trackReplayStatus.textContent,"
                "error:trackReplayError.textContent,disabled:[replayDriverFilter.disabled,replayLapFilter.disabled,replayPlayPause.disabled,replayScrubber.disabled]})")
            assert failure["metric"] == expected_mae and failure["n"] == expected_targets
            assert "No synthetic position fallback" in failure["error"] and all(failure["disabled"])
            print("PASS blocked Singapore replay remains unavailable while archived evaluation metrics load")
            print("PASS Marina Bay replay smoke complete")
        finally:
            if cdp:
                try:
                    cdp.close()
                except OSError:
                    pass
            smoke.stop_process_group(chrome)
            smoke.stop_process_group(server)


if __name__ == "__main__":
    raise SystemExit(main())
