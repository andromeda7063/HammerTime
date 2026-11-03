#!/usr/bin/env python3
"""Focused verification of Singapore replay visual interpolation and team colors."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

import browser_smoke as smoke


FRONTEND = smoke.FRONTEND
REPLAY_PATH = FRONTEND / "data/marina_bay_replay.json"
ASSET_PATH = FRONTEND / "data/marina_bay_driver_assets.json"
OUTPUT_DIR = smoke.DEFAULT_ARTIFACTS / "marina-bay-smoothing-qa"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker_pairs(data: dict) -> dict:
    frames = data["replay"]["frames"]
    drivers = data["drivers"]
    by_number = {d["number"]: i for i, d in enumerate(drivers)}

    def next_valid_pair(driver_number, predicate):
        index = by_number[driver_number]
        for i in range(len(frames) - 1):
            a, b = frames[i]["positions"][index], frames[i + 1]["positions"][index]
            if a is None or b is None or not predicate(a, b, 0.5):
                continue
            return {"driver": driver_number, "i": i, "a": a, "b": b,
                    "t0": frames[i]["race_elapsed_sec"],
                    "t1": frames[i + 1]["race_elapsed_sec"]}
        raise AssertionError(f"No suitable source pair for #{driver_number}")

    def same_lap(a, b, dt):
        return a["lap_number"] == b["lap_number"] and not a["is_pit_lap"] and not b["is_pit_lap"]

    def moving_same_lap(a, b, dt):
        distance = math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
        return same_lap(a, b, dt) and distance > 14

    def lap_wrap_plausible(a, b, dt):
        distance = math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
        speed = max(a["speed_kph"], b["speed_kph"]) / 3.6
        return (isinstance(a["lap_number"], int) and isinstance(b["lap_number"], int)
                and b["lap_number"] == a["lap_number"] + 1 and not a["is_pit_lap"]
                and not b["is_pit_lap"] and distance > 14 and distance <= speed * dt + 0.1)

    def lap_wrap_discontinuous(a, b, dt):
        distance = math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
        speed = max(a["speed_kph"], b["speed_kph"]) / 3.6
        return (isinstance(a["lap_number"], int) and isinstance(b["lap_number"], int)
                and b["lap_number"] == a["lap_number"] + 1 and not a["is_pit_lap"]
                and not b["is_pit_lap"] and distance > speed * dt + 0.1)

    same = next_valid_pair("55", moving_same_lap)
    lap_smooth = next_valid_pair("4", lap_wrap_plausible)
    lap_hold = next_valid_pair("1", lap_wrap_discontinuous)
    pit = next_valid_pair("1", lambda a, b, _dt: a["is_pit_lap"] != b["is_pit_lap"])

    center = len(frames) // 2
    midrace_candidates = []
    for driver in drivers:
        index = by_number[driver["number"]]
        for i in range(max(0, center - 300), min(len(frames) - 1, center + 300)):
            a, b = frames[i]["positions"][index], frames[i + 1]["positions"][index]
            if a is None or b is None or not moving_same_lap(a, b, 0.5):
                continue
            speed = max(a["speed_kph"], b["speed_kph"]) / 3.6
            delta = math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"])
            if delta <= speed * 0.5 + 0.1:
                midrace_candidates.append((abs(i - center), driver["number"], i, a, b))
    if not midrace_candidates:
        raise AssertionError("No plausible same-lap midpoint pair near the race midpoint")
    _, number, i, a, b = min(midrace_candidates)
    midrace = {"driver": number, "i": i, "a": a, "b": b,
               "t0": frames[i]["race_elapsed_sec"],
               "t1": frames[i + 1]["race_elapsed_sec"]}

    # The archived #22 run ends in a true null transition; no later samples exist.
    tsu_index = by_number["22"]
    tsu_valid = [i for i, frame in enumerate(frames) if frame["positions"][tsu_index] is not None]
    end = tsu_valid[-1]
    assert frames[end + 1]["positions"][tsu_index] is None
    missing = {"driver": "22", "i": end, "a": frames[end]["positions"][tsu_index],
               "b": None, "t0": frames[end]["race_elapsed_sec"],
               "t1": frames[end + 1]["race_elapsed_sec"]}
    return {"same_lap": same, "midrace": midrace, "lap_wrap_smooth": lap_smooth,
            "lap_wrap_hold": lap_hold, "pit_transition": pit,
            "null_transition": missing}


def js_json(cdp: smoke.DevTools, expression: str):
    value = cdp.evaluate(f"JSON.stringify({expression})")
    return json.loads(value)


def main() -> int:
    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    assets = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    assert sha(REPLAY_PATH) == assets["provenance"]["replay_json_sha256"]
    assert replay["replay"]["output_sample_hz"] == 2
    assert all(pos is None or isinstance(pos.get("is_pit_lap"), bool)
               for frame in replay["replay"]["frames"] for pos in frame["positions"])
    pairs = marker_pairs(replay)
    palette = assets["team_colors_2023"]
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("chromium")
    if not executable:
        raise RuntimeError("Chromium is required for the focused replay visual smoke.")
    port, debug = smoke.free_port(), smoke.free_port()
    page_url = f"http://127.0.0.1:{port}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-marina-visual-") as temp:
        profile = Path(temp) / "profile"
        profile.mkdir()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
             "--directory", str(FRONTEND)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        chrome = subprocess.Popen(
            [executable, "--headless", "--no-sandbox", "--disable-gpu",
             "--disable-extensions", "--disable-dev-shm-usage",
             f"--remote-debugging-port={debug}", "--remote-allow-origins=*",
             f"--user-data-dir={profile}", page_url], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        cdp = None
        try:
            endpoint = f"http://127.0.0.1:{debug}/json/list"
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    pages = json.load(urllib.request.urlopen(endpoint, timeout=1))
                    page = next(p for p in pages if p.get("type") == "page"
                                and p.get("url", "").startswith(page_url))
                    cdp = smoke.DevTools(page["webSocketDebuggerUrl"])
                    break
                except (OSError, ValueError, StopIteration):
                    time.sleep(0.2)
            if cdp is None:
                raise TimeoutError("Chromium did not expose the replay page via DevTools")
            cdp.command("Runtime.enable")
            cdp.command("Page.enable")
            cdp.command("Emulation.setDeviceMetricsOverride", {
                "width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False,
            })
            smoke.wait_for_page(cdp, "document.getElementById('trackReplayStatus')?.textContent.startsWith('Archived replay ready')", 40)
            smoke.wait_for_page(cdp, "document.getElementById('marinaInferenceStatus')?.textContent.startsWith('Checkpoint outputs loaded')", 20)
            smoke.wait_for_page(cdp, "window.HammerTimeDriverAssets !== undefined", 10)

            instrument = """
              (() => {
                const c = document.getElementById('marinaBayCanvas').getContext('2d');
                if (!c.__visualSmokeInstalled) {
                  c.__visualSmokeInstalled = true;
                  c.__visualSmoke = {arcs:[], active:null};
                  const arc = c.arc.bind(c), fill = c.fill.bind(c), stroke = c.stroke.bind(c);
                  c.arc = function(x,y,r,...rest) { this.__visualSmoke.active={x,y,r}; return arc(x,y,r,...rest); };
                  c.fill = function(...args) { if (this.__visualSmoke.active) this.__visualSmoke.arcs.push({...this.__visualSmoke.active,fill:this.fillStyle,fillRule:args[0]||'nonzero'}); return fill(...args); };
                  c.stroke = function(...args) { if (this.__visualSmoke.active) { const last=this.__visualSmoke.arcs.at(-1); if (last) {last.stroke=this.strokeStyle;last.lineWidth=this.lineWidth;} this.__visualSmoke.active=null; } return stroke(...args); };
                }
                return true;
              })()
            """
            assert cdp.evaluate(instrument)
            test_setup = """
              (driver, elapsed) => {
                const select=document.getElementById('replayDriverFilter');
                select.value=String(driver); select.dispatchEvent(new Event('change',{bubbles:true}));
                const c=document.getElementById('marinaBayCanvas').getContext('2d'); c.__visualSmoke.arcs=[]; c.__visualSmoke.active=null;
                setReplayElapsed(elapsed);
                return c.__visualSmoke.arcs;
              }
            """
            def draw(driver: str, elapsed: float):
                return js_json(cdp, f"({test_setup})({json.dumps(driver)},{elapsed!r})")

            # Map source XY to CSS canvas coordinates, independent of implementation output.
            bounds_json = json.dumps(replay["track_outline"]["bounds_m"])
            map_expr = f"""(x,y) => {{ const c=document.getElementById('marinaBayCanvas'),r=c.getBoundingClientRect(),b={bounds_json};
              const pad=Math.max(32,Math.min(56,r.width*0.06)),sx=(r.width-pad*2)/(b.max_x-b.min_x),sy=(r.height-pad*2)/(b.max_y-b.min_y),s=Math.min(sx,sy);
              const ox=(r.width-(b.max_x-b.min_x)*s)/2,oy=(r.height-(b.max_y-b.min_y)*s)/2;
              return {{x:ox+(x-b.min_x)*s,y:r.height-oy-(y-b.min_y)*s}}; }}
            """
            def expected_pair(pair, alpha):
                a, b = pair["a"], pair["b"]
                x = a["x_m"] + (b["x_m"] - a["x_m"]) * alpha
                y = a["y_m"] + (b["y_m"] - a["y_m"]) * alpha
                return js_json(cdp, f"({map_expr})({x!r},{y!r})")

            def focus_arc(arcs):
                found = [arc for arc in arcs if abs(arc["r"] - 7) < 0.01]
                assert len(found) == 1, arcs
                return found[0]

            def assert_pos(actual, expected, label):
                assert abs(actual["x"] - expected["x"]) < 0.75, (label, actual, expected)
                assert abs(actual["y"] - expected["y"]) < 0.75, (label, actual, expected)

            pair = pairs["same_lap"]
            arc0 = focus_arc(draw(pair["driver"], pair["t0"]))
            arc1 = focus_arc(draw(pair["driver"], pair["t1"]))
            mid = (pair["t0"] + pair["t1"]) / 2
            arcm = focus_arc(draw(pair["driver"], mid))
            assert_pos(arc0, expected_pair(pair, 0), "exact first endpoint")
            assert_pos(arc1, expected_pair(pair, 1), "exact second endpoint")
            assert_pos(arcm, expected_pair(pair, 0.5), "fractional midpoint")

            # Null, pit, and discontinuous lap intervals hold the current recorded sample;
            # when the current source tick itself is null, the marker disappears.
            null_pair = pairs["null_transition"]
            null_mid = (null_pair["t0"] + null_pair["t1"]) / 2
            null_start_arc = focus_arc(draw("22", null_pair["t0"]))
            null_mid_arc = focus_arc(draw("22", null_mid))
            assert_pos(null_start_arc, js_json(cdp, f"({map_expr})({null_pair['a']['x_m']},{null_pair['a']['y_m']})"), "last valid null-gap endpoint")
            assert_pos(null_mid_arc, js_json(cdp, f"({map_expr})({null_pair['a']['x_m']},{null_pair['a']['y_m']})"), "hold before null tick")
            assert not [arc for arc in draw("22", null_pair["t1"]) if abs(arc["r"]-7)<0.01]

            pit = pairs["pit_transition"]
            pit_mid_arc = focus_arc(draw(pit["driver"], (pit["t0"]+pit["t1"])/2))
            assert_pos(pit_mid_arc, js_json(cdp, f"({map_expr})({pit['a']['x_m']},{pit['a']['y_m']})"), "pit transition source hold")
            lap_smooth = pairs["lap_wrap_smooth"]
            lap_mid_arc = focus_arc(draw(lap_smooth["driver"], (lap_smooth["t0"]+lap_smooth["t1"])/2))
            assert_pos(lap_mid_arc, expected_pair(lap_smooth, 0.5), "plausible lap wrap midpoint")
            lap_hold = pairs["lap_wrap_hold"]
            lap_hold_arc = focus_arc(draw(lap_hold["driver"], (lap_hold["t0"]+lap_hold["t1"])/2))
            assert_pos(lap_hold_arc, js_json(cdp, f"({map_expr})({lap_hold['a']['x_m']},{lap_hold['a']['y_m']})"), "implausible lap wrap holds source")
            midrace_pair = pairs["midrace"]
            midrace_time = (midrace_pair["t0"] + midrace_pair["t1"]) / 2
            midrace_marker = focus_arc(draw(midrace_pair["driver"], midrace_time))
            assert_pos(midrace_marker, expected_pair(midrace_pair, 0.5), "mid-race fractional point")

            # Independent team palette oracle: team identities and 2023 FastF1 official palette.
            colors = {}
            for number in ("55", "16", "1"):
                position = replay["replay"]["frames"][pair["i"]]["positions"][
                    next(i for i, d in enumerate(replay["drivers"]) if d["number"] == number)]
                arcs = draw(number, pair["t0"])
                marker = focus_arc(arcs)
                descriptor = next(d for d in replay["drivers"] if d["number"] == number)
                expected_color = palette[descriptor["team"]]["official_hex"]
                assert marker["fill"].lower() == expected_color.lower(), (number, marker, expected_color)
                assert marker.get("stroke", "").lower() == "#f5f5ef", (number, marker)
                colors[number] = marker["fill"]
            assert colors["55"] == colors["16"] and colors["55"] != colors["1"], colors

            # The actual range control seeks to a recorded source endpoint exactly.
            seek_driver = midrace_pair["driver"]
            seek_expr = f"""(() => {{
              const select=document.getElementById('replayDriverFilter'); select.value={json.dumps(seek_driver)}; select.dispatchEvent(new Event('change'));
              const c=document.getElementById('marinaBayCanvas').getContext('2d'); c.__visualSmoke.arcs=[]; c.__visualSmoke.active=null;
              const slider=document.getElementById('replayScrubber'); slider.value={midrace_pair['t0']!r}; slider.dispatchEvent(new Event('input',{{bubbles:true}}));
              return c.__visualSmoke.arcs;
            }})()"""
            seek_arc = focus_arc(js_json(cdp, seek_expr))
            assert_pos(seek_arc, expected_pair(midrace_pair, 0), "scrubber seek to recorded endpoint")

            # Current source tick and table remain distinct at a fractional playhead; pause holds time.
            playhead = mid
            cdp.evaluate(f"setReplayElapsed({playhead!r}); document.querySelector('[data-replay-speed=\"1\"]').click(); document.getElementById('replayPlayPause').click()")
            time.sleep(1.0)
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            one_x = float(cdp.evaluate("Number(document.getElementById('trackFrameCaption').textContent.match(/source session timestamp ([0-9.]+)/)[1])"))
            cdp.evaluate(f"setReplayElapsed({playhead!r}); document.querySelector('[data-replay-speed=\"4\"]').click(); document.getElementById('replayPlayPause').click()")
            time.sleep(1.0)
            cdp.evaluate("document.getElementById('replayPlayPause').click()")
            four_x = float(cdp.evaluate("Number(document.getElementById('trackFrameCaption').textContent.match(/source session timestamp ([0-9.]+)/)[1])"))
            assert four_x - one_x >= 1.0, (one_x, four_x, playhead)
            paused = float(cdp.evaluate("Number(document.getElementById('trackFrameCaption').textContent.match(/source session timestamp ([0-9.]+)/)[1])"))
            time.sleep(0.2)
            assert float(cdp.evaluate("Number(document.getElementById('trackFrameCaption').textContent.match(/source session timestamp ([0-9.]+)/)[1])")) == paused

            # Bounded checkpoint display uses recorded history only; the first target remains unavailable.
            inference = js_json(cdp, "({status:document.getElementById('marinaInferenceStatus').textContent,count:document.getElementById('marinaInferenceCount').textContent,rows:[...document.getElementById('marinaInferenceRows').rows].map(r=>[...r.cells].map(c=>c.textContent))})")
            assert inference["status"] == "Checkpoint outputs loaded · 1 unavailable", inference["status"]
            assert inference["count"] == "18" and len(inference["rows"]) == 19
            assert inference["rows"][0][1] == "Unavailable", "The target without recorded history must remain unavailable"
            assert all(row[1] != "Unavailable" for row in inference["rows"][1:]), "Remaining recorded-history sectors should have predictions"
            sector3 = inference["rows"][2]
            assert sector3[1] == "0.07711" and sector3[2] == "-2.00000", sector3

            # Review images use a valid fractional replay playhead and the existing lap-1 inference section.
            cdp.command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
            cdp.evaluate(f"document.getElementById('replayDriverFilter').value='all'; document.getElementById('replayDriverFilter').dispatchEvent(new Event('change')); setReplayElapsed({midrace_time!r}); window.scrollTo(0, document.getElementById('marinaBayCanvas').getBoundingClientRect().top - 130)")
            cdp.screenshot(output_dir / "marina-bay-team-colors-midpoint-wide.png")
            wide = js_json(cdp, "({dpr:devicePixelRatio,canvasWidth:document.getElementById('marinaBayCanvas').width,rect:document.getElementById('marinaBayCanvas').getBoundingClientRect().width,body:document.body.scrollWidth,viewport:innerWidth})")
            assert abs(wide["canvasWidth"] / wide["rect"] - 1) < 0.02
            cdp.command("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 2, "mobile": False})
            cdp.evaluate(f"(async()=>{{ setReplayElapsed({midrace_time!r}); await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))); window.scrollTo(0, document.getElementById('marinaBayCanvas').getBoundingClientRect().top - 120); }})()")
            mobile = js_json(cdp, "({dpr:devicePixelRatio,canvasWidth:document.getElementById('marinaBayCanvas').width,rect:document.getElementById('marinaBayCanvas').getBoundingClientRect().width,body:document.body.scrollWidth,viewport:innerWidth})")
            assert abs(mobile["canvasWidth"] / mobile["rect"] - 2) < 0.03
            assert mobile["body"] <= mobile["viewport"] + 1, mobile
            cdp.screenshot(output_dir / "marina-bay-team-colors-midpoint-mobile.png")
            cdp.command("Log.enable")
            runtime_errors = [e for e in cdp.events if e.get("method") == "Runtime.exceptionThrown"]
            assert not runtime_errors, runtime_errors
            print("PASS midpoint/endpoints/null-hold/pit-hold/lap-wrap/team palette/clock/DPR/inference")
            print(json.dumps({"same_lap_pair": {k: pair[k] for k in ("driver", "i", "t0", "t1")},
                              "midrace_pair": {k: midrace_pair[k] for k in ("driver", "i", "t0", "t1")},
                              "null_transition": {k: null_pair[k] for k in ("driver", "i", "t0", "t1")},
                              "pit_transition": {k: pit[k] for k in ("driver", "i", "t0", "t1")},
                              "lap_wrap_plausible": {k: lap_smooth[k] for k in ("driver", "i", "t0", "t1")},
                              "lap_wrap_fallback": {k: lap_hold[k] for k in ("driver", "i", "t0", "t1")},
                              "colors": colors, "clock_source_1x": one_x,
                              "clock_source_4x": four_x, "wide": wide, "mobile": mobile,
                              "inference_status": inference["status"],
                              "screenshot_dir": str(output_dir)}, indent=2))
        finally:
            if cdp is not None:
                cdp.close()
            for process in (chrome, server):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
