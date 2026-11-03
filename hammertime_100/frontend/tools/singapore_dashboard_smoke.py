#!/usr/bin/env python3
"""Verify the Singapore in-sample dashboard and its preserved run split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import browser_smoke as smoke

ROOT = smoke.ROOT
FRONTEND = smoke.FRONTEND
DATA_PATH = FRONTEND / "data/season_model.json"
RUN_DIR = ROOT / ".local/research-history/2023-fastf1-season/training/769a76aab164b514cc0a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    saved = json.loads((RUN_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    scope = data["inference_scope"]
    assert scope == {
        "race_id": "2023-Singapore-R",
        "split_role": "training",
        "held_out_test": False,
        "inference_window_count": scope["inference_window_count"],
        "note": "Full-session in-sample checkpoint inference on recorded Singapore features; not held-out evaluation.",
    }
    actual_split = data["split"]
    expected_split = saved["split"]
    assert actual_split == {
        "method": expected_split["method"],
        "train_window_count": expected_split["train"]["sample_count"],
        "validation_window_count": expected_split["validation"]["sample_count"],
        "test_window_count": expected_split["test"]["sample_count"],
        "test_races": expected_split["test"]["race_ids"],
    }
    assert scope["race_id"] not in actual_split["test_races"]
    assert data["metrics"]["sample_count"] == data["prediction_sample"]["source_rows"]
    assert all(point["race_id"] == scope["race_id"] for point in data["prediction_sample"]["points"])
    assert all(test["passed"] is None and "not assessed" in test["notes"].lower()
               for test in data["metrics"]["physics_tests"])
    assert sha256(RUN_DIR / "best.pt") == data["checkpoint"]["sha256"]
    assert sha256(RUN_DIR / "best.pt") == data["provenance"]["source_artifact_sha256"]["best.pt"]
    print(f"PASS inference is explicitly in-sample: {scope['race_id']} / {scope['split_role']}")
    print(f"PASS original held-out split preserved: {actual_split['test_window_count']:,} windows, {len(actual_split['test_races'])} races")
    print(f"PASS inference rows and unavailable physics gates: N={data['metrics']['sample_count']:,}; gates not assessed")

    browser = shutil.which("chromium")
    if not browser:
        raise RuntimeError("Chromium executable not found")
    port, debug_port = smoke.free_port(), smoke.free_port()
    url = f"http://127.0.0.1:{port}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-singapore-dashboard-") as temp:
        profile = Path(temp) / "profile"
        profile.mkdir()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
             "--directory", str(FRONTEND)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        chrome = subprocess.Popen(
            [browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-extensions",
             "--disable-dev-shm-usage", f"--remote-debugging-port={debug_port}",
             "--remote-allow-origins=*", f"--user-data-dir={profile}", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        cdp = None
        try:
            endpoint = f"http://127.0.0.1:{debug_port}/json/list"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    pages = json.load(urllib.request.urlopen(endpoint, timeout=1))
                    page = next(p for p in pages if p.get("type") == "page" and p.get("url", "").startswith(url))
                    cdp = smoke.DevTools(page["webSocketDebuggerUrl"])
                    break
                except (OSError, ValueError, StopIteration):
                    time.sleep(0.2)
            if cdp is None:
                raise TimeoutError("Chromium did not expose the dashboard via DevTools")
            cdp.command("Runtime.enable")
            cdp.command("Log.enable")
            cdp.command("Page.enable")
            status = smoke.wait_for_page(cdp, "(() => { const e=document.querySelector('#loadStatus'); return e?.textContent==='Run loaded · archived metrics' ? e.textContent : null; })()")
            state = json.loads(cdp.evaluate("JSON.stringify({valid:validRun(state.run), mae:metricMae.textContent, r2:metricR2.textContent, n:metricN.textContent, race:raceFilter.value, raceDisabled:raceFilter.disabled, split:splitStrip.innerText, metricGates:[maeGate.textContent,r2Gate.textContent], physics:[...physicsList.querySelectorAll('.gate-pill')].map(x=>x.textContent), error:dataError.textContent})"))
            assert status == "Run loaded · archived metrics"
            assert state["valid"] and state["race"] == "2023-Singapore-R" and state["raceDisabled"]
            assert state["n"] == f"{data['metrics']['sample_count']:,}"
            assert state["metricGates"] == ["NOT ASSESSED", "NOT ASSESSED"]
            assert all(value == "NOT ASSESSED" for value in state["physics"])
            assert not state["error"]
            print(f"PASS dashboard loaded in Chromium: MAE={state['mae']}, R²={state['r2']}, N={state['n']}")
            print(f"PASS no acceptance gate is claimed for the in-sample view: {state['metricGates'] + state['physics']}")
            for width, height in ((1440, 1000), (390, 1000), (320, 900)):
                cdp.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 500,
                })
                metrics = json.loads(cdp.evaluate("JSON.stringify({body:document.body.scrollWidth, viewport:innerWidth})"))
                assert metrics["body"] <= metrics["viewport"]
                assert metrics["viewport"] - metrics["body"] <= (16 if width >= 500 else 0), (width, metrics)
                if width == 1440:
                    out = ROOT / ".local/research-history/2026-11-03/singapore-dashboard-wide.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cdp.screenshot(out)
                print(f"PASS no page-level horizontal overflow at {width}px")
            errors = [event for event in cdp.events if event.get("method") in ("Runtime.exceptionThrown", "Network.loadingFailed")]
            assert not errors, errors
        finally:
            if cdp:
                cdp.close()
            chrome.terminate()
            server.terminate()
            chrome.wait(timeout=5)
            server.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
