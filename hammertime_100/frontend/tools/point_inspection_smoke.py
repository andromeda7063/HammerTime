#!/usr/bin/env python3
"""Focused browser checks for scatter-point tooltip and table inspection."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import browser_smoke as smoke


def evaluate_json(cdp: smoke.DevTools, expression: str) -> dict:
    return json.loads(cdp.evaluate(f"JSON.stringify({expression})"))


def scroll_chart_into_view(cdp: smoke.DevTools) -> None:
    cdp.evaluate(
        "(() => { document.documentElement.style.scrollBehavior='auto'; "
        "const c=document.querySelector('#predictionChart'); "
        "window.scrollTo(0,Math.max(0,window.scrollY+c.getBoundingClientRect().top-120)); })()"
    )
    cdp.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", default=None, help="Chromium executable (default: chromium)")
    parser.add_argument("--artifacts-dir", type=Path,
                        default=smoke.DEFAULT_ARTIFACTS / "point-inspection-qa",
                        help="directory for point inspection review screenshots")
    args = parser.parse_args()
    executable = shutil.which(args.chromium or "chromium")
    if not executable:
        parser.error("Chromium executable not found")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    frontend = smoke.FRONTEND
    http_port, debug_port = smoke.free_port(), smoke.free_port()
    base_url = f"http://127.0.0.1:{http_port}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-point-inspection-") as temp:
        profile = Path(temp) / "profile"
        downloads = Path(temp) / "downloads"
        profile.mkdir()
        downloads.mkdir()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(http_port), "--bind", "127.0.0.1",
             "--directory", str(frontend)],
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
            deadline = time.monotonic() + 15
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
            cdp.command("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(downloads)})
            smoke.wait_for_page(cdp,
                "(() => document.querySelector('#loadStatus')?.textContent === 'Run loaded · archived metrics')()")
            cdp.evaluate("window.__auditPoints=state.run.prediction_sample.points")

            # Hover the four data extrema at desktop, high-DPI mobile and narrow mobile sizes.
            # Pointer coordinates are rounded to CSS pixels before determining the expected hit.
            for width, height, dpr, mobile in [(1440, 900, 1, False), (390, 844, 2, True), (320, 740, 1, True)]:
                cdp.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": dpr, "mobile": mobile,
                })
                scroll_chart_into_view(cdp)
                edges = []
                for axis, direction in (("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")):
                    key = "actual" if axis == "x" else "predicted"
                    target = evaluate_json(cdp,
                        "(() => { const h=state.chartHitTest,c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(); "
                        f"const k='{key}',direction='{direction}'; "
                        "const p=h.points.reduce((best,v)=>!best||(direction==='min'?+v[k]<+best[k]:+v[k]>+best[k])?v:best,null); "
                        "const x=Math.round(r.left+h.x(+p.actual)),y=Math.round(r.top+h.y(+p.predicted)); "
                        "const hit=nearestScatterPoint({clientX:x,clientY:y}); return {p,x,y,expected:hit?.point,nearbyCount:hit?.nearbyCount}; })()")
                    cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": target["x"], "y": target["y"]})
                    tip = evaluate_json(cdp,
                        "(() => { const e=document.querySelector('#pointTooltip'),r=e.getBoundingClientRect(),w=e.parentElement.getBoundingClientRect(); "
                        "return {hidden:e.hidden,aria:e.getAttribute('aria-hidden'),text:e.textContent,"
                        "left:r.left-w.left,top:r.top-w.top,right:r.right-w.left,bottom:r.bottom-w.top,width:w.width,height:w.height}; })()")
                    point = target["expected"]
                    race = point["race_id"].removeprefix("2023-").removesuffix("-R").replace("-", " ")
                    expected_fragments = ["Nearest sampled prediction", race, f"driver {point['driver']}", f"lap {point['lap']}",
                                          f"mini-sector {int(point['mini_sector']) + 1}", "Actual residual",
                                          "predicted", "error"]
                    if target["nearbyCount"] > 1:
                        expected_fragments.append(f"{target['nearbyCount']} sampled points within 11 px")
                    assert not tip["hidden"] and tip["aria"] == "false", (width, axis, direction, tip)
                    assert all(fragment in tip["text"] for fragment in expected_fragments), (width, axis, direction, tip, point)
                    assert (tip["left"] >= -1 and tip["top"] >= -1
                            and tip["right"] <= tip["width"] + 1
                            and tip["bottom"] <= tip["height"] + 1), (width, axis, direction, tip)
                    edges.append(f"{axis}-{direction}→{point['race_id']}/#{point['driver']}/{point['lap']}")

                # Independent geometry oracle: choose a race/driver subset with a
                # well-separated point, then project raw JSON values using the
                # documented canvas margins/domain. Do not use chartHitTest or
                # nearestScatterPoint to produce the expected hit.
                group = evaluate_json(cdp,
                    "(() => { const all=state.run.prediction_sample.points,c=document.querySelector('#predictionChart'),w=c.getBoundingClientRect().width; "
                    "const groups=new Map(); for(const p of all){const k=p.race_id+'|'+p.driver;if(!groups.has(k))groups.set(k,[]);groups.get(k).push(p)} "
                    "let best=null; for(const [key,ps] of groups){if(ps.length<2)continue;const vals=ps.flatMap(p=>[+p.actual,+p.predicted]);let lo=Math.min(...vals),hi=Math.max(...vals),span=Math.max(hi-lo,.001);lo-=span*.05;hi+=span*.05; "
                    "const xy=p=>[65+((+p.actual-lo)/(hi-lo))*(w-65-22),25+390-25-58-((+p.predicted-lo)/(hi-lo))*(390-25-58)]; "
                    "for(let i=0;i<ps.length;i++){const a=xy(ps[i]);let nearest=Infinity;for(let j=0;j<ps.length;j++){if(i===j)continue;const b=xy(ps[j]);nearest=Math.min(nearest,Math.hypot(a[0]-b[0],a[1]-b[1]))}if(!best||nearest>best.separation)best={race:ps[i].race_id,driver:String(ps[i].driver),separation:nearest}}} "
                    "return best; })()")
                cdp.evaluate(
                    f"(() => {{ raceFilter.value={json.dumps(group['race'])}; driverFilter.value={json.dumps(group['driver'])}; "
                    "raceFilter.dispatchEvent(new Event('change')); })()"
                )
                oracle = evaluate_json(cdp,
                    "(() => { const ps=state.run.prediction_sample.points.filter(p=>p.race_id===raceFilter.value&&String(p.driver)===driverFilter.value); "
                    "const c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(),w=r.width,vals=ps.flatMap(p=>[+p.actual,+p.predicted]); "
                    "let lo=Math.min(...vals),hi=Math.max(...vals),span=Math.max(hi-lo,.001);lo-=span*.05;hi+=span*.05; "
                    "const xy=p=>[65+((+p.actual-lo)/(hi-lo))*(w-65-22),25+390-25-58-((+p.predicted-lo)/(hi-lo))*(390-25-58)]; "
                    "let best=null;for(const p of ps){const q=xy(p),x=Math.round(r.left+q[0]),y=Math.round(r.top+q[1]);let nearest=null,dist=Infinity,count=0; "
                    "for(const candidate of ps){const z=xy(candidate),d=Math.hypot(r.left+z[0]-x,r.top+z[1]-y);if(d<=11)count++;if(d<dist){dist=d;nearest=candidate}} "
                    "if(!best||dist>best.dist)best={point:p,nearest,x,y,nearbyCount:count,dist}}return best; })()")
                cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": oracle["x"], "y": oracle["y"]})
                oracle_tip = cdp.evaluate("pointTooltip.textContent")
                expected = oracle["nearest"]
                expected_race = expected["race_id"].removeprefix("2023-").removesuffix("-R").replace("-", " ")
                assert expected["race_id"] in oracle_tip or expected_race in oracle_tip, (width, dpr, oracle, oracle_tip)
                assert f"driver {expected['driver']}" in oracle_tip and f"lap {expected['lap']}" in oracle_tip
                if oracle["nearbyCount"] > 1:
                    assert f"{oracle['nearbyCount']} sampled points within 11 px" in oracle_tip
                print(f"PASS independent projected hit {width}px DPR{dpr}: {group['race']}/#{group['driver']}, "
                      f"isolated separation {group['separation']:.2f}px, expected nearest {expected['race_id']}/#{expected['driver']}/{expected['lap']}")

                cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 3, "y": 3})
                assert cdp.evaluate("document.querySelector('#pointTooltip').hidden")
                cdp.evaluate("(() => { raceFilter.value='2023-Mexico-R'; raceFilter.dispatchEvent(new Event('change')); })()")
                smoke.wait_for_page(cdp, "state.raceId==='2023-Mexico-R'")
                point_xy = evaluate_json(cdp,
                    "(() => { const h=state.chartHitTest,p=h.points[0],c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(); "
                    "return {x:Math.round(r.left+h.x(+p.actual)),y:Math.round(r.top+h.y(+p.predicted))}; })()")
                cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", **point_xy})
                assert not cdp.evaluate("document.querySelector('#pointTooltip').hidden")
                cdp.evaluate(
                    "(() => { state.run.prediction_sample.points=state.run.prediction_sample.points.filter(p=>p.race_id==='2023-AbuDhabi-R'); "
                    "raceFilter.value='2023-AbuDhabi-R'; driverFilter.value='4'; drawScatter(); })()"
                )
                empty = evaluate_json(cdp,
                    "({hidden:pointTooltip.hidden,aria:pointTooltip.getAttribute('aria-hidden'),"
                    "points:state.chartHitTest?.points.length??0,body:document.body.scrollWidth,viewport:innerWidth})")
                assert empty["hidden"] and empty["aria"] == "true" and empty["points"] == 0
                assert empty["body"] <= empty["viewport"] + 1, (width, empty)
                cdp.evaluate(
                    "(() => { state.run.prediction_sample.points=window.__auditPoints; raceFilter.value='all'; "
                    "driverFilter.value='all'; sampleOrderFilter.value='source'; raceFilter.dispatchEvent(new Event('change')); })()"
                )
                smoke.wait_for_page(cdp, "state.raceId==='all'&&state.driver==='all'&&state.sampleOrder==='source'")
                reset_card = evaluate_json(cdp,
                    "({title:selectedRaceTitle.textContent,n:raceN.textContent,expected:integer(state.run.metrics.sample_count)})")
                assert reset_card == {"title": "All held-out races", "n": reset_card["expected"],
                                      "expected": "37,956"}, reset_card
                print(f"PASS hover tooltip {width}px DPR{dpr}: {edges}; pointerleave/filter/empty clears; no page overflow")

            # A click follows the same nearest-hit result as the tooltip, then selects
            # and focuses that exact row/page in each of the three sort orders.
            cdp.command("Emulation.setDeviceMetricsOverride", {
                "width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False,
            })
            for order in ("source", "largest-error", "smallest-error"):
                cdp.evaluate(
                    f"(() => {{ predictionDataDetails.open=true; sampleOrderFilter.value='{order}'; "
                    "sampleOrderFilter.dispatchEvent(new Event('change')); })()"
                )
                scroll_chart_into_view(cdp)
                target = evaluate_json(cdp,
                    "(() => { const p=state.run.prediction_sample.points[500],h=state.chartHitTest,c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(); "
                    "const x=Math.round(r.left+h.x(+p.actual)),y=Math.round(r.top+h.y(+p.predicted)),hit=nearestScatterPoint({clientX:x,clientY:y}); "
                    "return {x,y,expected:hit.point,nearbyCount:hit.nearbyCount,index:state.run.prediction_sample.points.indexOf(hit.point)}; })()")
                cdp.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": target["x"], "y": target["y"]})
                tooltip = cdp.evaluate("document.querySelector('#pointTooltip').textContent")
                if order == "source":
                    cdp.screenshot(args.artifacts_dir / "point-tooltip-hover.png")
                for event in ("mousePressed", "mouseReleased"):
                    cdp.command("Input.dispatchMouseEvent", {
                        "type": event, "x": target["x"], "y": target["y"], "button": "left", "clickCount": 1,
                    })
                selected = json.loads(smoke.wait_for_page(cdp,
                    "(() => { const row=document.querySelector('#predictionTableBody [data-selected-point=\"true\"]'); "
                    "return row&&document.activeElement===row?JSON.stringify({order:state.sampleOrder,open:predictionDataDetails.open,"
                    "index:orderedSamplePoints().indexOf(state.selectedPoint),sourceIndex:state.run.prediction_sample.points.indexOf(state.selectedPoint),"
                    "focus:document.activeElement.tagName,cells:[...row.cells].map(c=>c.textContent),className:row.className}):null; })()"))
                point = target["expected"]
                race = point["race_id"].removeprefix("2023-").removesuffix("-R").replace("-", " ")
                expected_cells = [race, str(point["driver"]), str(point["lap"]), str(int(point["mini_sector"]) + 1),
                                  f"{point['actual']:.5f}", f"{point['predicted']:.5f}",
                                  f"{point['predicted'] - point['actual']:.5f}"]
                assert selected["order"] == order and selected["open"] and selected["focus"] == "TR"
                assert selected["sourceIndex"] == target["index"] and selected["cells"] == expected_cells
                assert selected["index"] // 40 >= 0 and "selected-point-row" in selected["className"]
                assert all(fragment in tooltip for fragment in
                           [race, f"driver {point['driver']}", f"lap {point['lap']}", f"mini-sector {int(point['mini_sector']) + 1}"])
                print(f"PASS click under {order}: tooltip and focused table row match nearest source row {selected['sourceIndex']}; "
                      f"ordered position {selected['index']}, page {selected['index'] // 40 + 1}")
                if order == "source":
                    cdp.screenshot(args.artifacts_dir / "point-selected-table-row.png")

            # Exercise race-card scope through real selector changes and browser
            # history: race-wide metrics must follow the restored race filter.
            cdp.evaluate(
                "(() => { raceFilter.value='2023-Mexico-R'; driverFilter.value='44'; "
                "raceFilter.dispatchEvent(new Event('change')); })()"
            )
            smoke.wait_for_page(cdp, "state.raceId==='2023-Mexico-R'&&state.driver==='44'")
            mexico_card = evaluate_json(cdp,
                "({title:selectedRaceTitle.textContent,n:raceN.textContent,expected:integer(state.run.race_metrics.find(x=>x.race_id===state.raceId).sample_count)})")
            assert mexico_card == {"title": "Mexico", "n": mexico_card["expected"], "expected": "10,759"}, mexico_card
            cdp.evaluate(
                "(() => { raceFilter.value='2023-LasVegas-R'; driverFilter.value='4'; "
                "raceFilter.dispatchEvent(new Event('change')); })()"
            )
            smoke.wait_for_page(cdp, "state.raceId==='2023-LasVegas-R'&&state.driver==='4'")
            vegas_card = evaluate_json(cdp,
                "({title:selectedRaceTitle.textContent,n:raceN.textContent,expected:integer(state.run.race_metrics.find(x=>x.race_id===state.raceId).sample_count)})")
            assert vegas_card["title"] == "LasVegas" and vegas_card["n"] == vegas_card["expected"], vegas_card
            cdp.evaluate("history.back()")
            smoke.wait_for_page(cdp, "state.raceId==='2023-Mexico-R'&&state.driver==='44'")
            back_card = evaluate_json(cdp,
                "({title:selectedRaceTitle.textContent,n:raceN.textContent,expected:integer(state.run.race_metrics.find(x=>x.race_id===state.raceId).sample_count)})")
            assert back_card == mexico_card, (mexico_card, back_card)
            cdp.evaluate(
                "(() => { raceFilter.value='all'; driverFilter.value='all'; "
                "raceFilter.dispatchEvent(new Event('change')); })()"
            )
            smoke.wait_for_page(cdp, "state.raceId==='all'&&state.driver==='all'")
            final_reset_card = evaluate_json(cdp,
                "({title:selectedRaceTitle.textContent,n:raceN.textContent,expected:integer(state.run.metrics.sample_count)})")
            assert final_reset_card == reset_card, (reset_card, final_reset_card)
            print(f"PASS race metric card tracks real selection/history/reset: {mexico_card} → {vegas_card} → back → {final_reset_card}")

            # Native touch taps select the nearest row; a vertical swipe remains a scroll gesture.
            cdp.command("Emulation.setDeviceMetricsOverride", {
                "width": 390, "height": 844, "deviceScaleFactor": 2, "mobile": True,
            })
            cdp.command("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 1})
            cdp.evaluate("state.sampleOrder='source'; sampleOrderFilter.value='source'; state.selectedPoint=null; renderPredictionTable(orderedSamplePoints());")
            scroll_chart_into_view(cdp)
            target = evaluate_json(cdp,
                "(() => { const p=state.run.prediction_sample.points[501],h=state.chartHitTest,c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(); "
                    "const x=Math.round(r.left+h.x(+p.actual)),y=Math.round(r.top+h.y(+p.predicted)),hit=nearestScatterPoint({clientX:x,clientY:y}); "
                    "return {x,y,index:state.run.prediction_sample.points.indexOf(hit.point)}; })()")
            cdp.command("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{
                "x": target["x"], "y": target["y"], "radiusX": 2, "radiusY": 2, "force": 1, "id": 1,
            }]})
            cdp.command("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            tapped = json.loads(smoke.wait_for_page(cdp,
                "(() => { const row=document.querySelector('#predictionTableBody [data-selected-point=\"true\"]'); "
                "return row&&document.activeElement===row?JSON.stringify({index:state.run.prediction_sample.points.indexOf(state.selectedPoint),"
                "open:predictionDataDetails.open}):null; })()"))
            assert tapped["index"] == target["index"] and tapped["open"], (target, tapped)

            cdp.evaluate("(() => { state.selectedPoint=null; renderPredictionTable(orderedSamplePoints()); })()")
            scroll_chart_into_view(cdp)
            swipe_start = evaluate_json(cdp,
                "(() => { const p=state.run.prediction_sample.points[700],h=state.chartHitTest,c=document.querySelector('#predictionChart'),r=c.getBoundingClientRect(); "
                "return {x:Math.round(r.left+h.x(+p.actual)),y:Math.round(r.top+h.y(+p.predicted)),scrollY}; })()")
            cdp.command("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{
                "x": swipe_start["x"], "y": swipe_start["y"], "radiusX": 3, "radiusY": 3, "force": 1, "id": 1,
            }]})
            cdp.command("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{
                "x": swipe_start["x"], "y": swipe_start["y"] - 90, "radiusX": 3, "radiusY": 3, "force": 1, "id": 1,
            }]})
            cdp.command("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            cdp.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
            swipe = evaluate_json(cdp,
                "({scrollY,selected:state.selectedPoint!==null,selectedRows:predictionTableBody.querySelectorAll('[data-selected-point=true]').length,"
                "tooltipHidden:pointTooltip.hidden})")
            assert swipe["scrollY"] != swipe_start["scrollY"] and not swipe["selected"]
            assert swipe["selectedRows"] == 0 and swipe["tooltipHidden"], swipe
            print(f"PASS touch tap selects nearest point row; vertical swipe scrolls without selection: {swipe}")

            # Validate the view-provenance download against the loaded season JSON
            # and URL. It must describe the filtered deterministic sample without
            # embedding any prediction rows.
            cdp.evaluate(
                "(() => { predictionDataDetails.open=true; raceFilter.value='2023-Mexico-R'; driverFilter.value='44'; "
                "sampleOrderFilter.value='largest-error'; raceFilter.dispatchEvent(new Event('change')); })()"
            )
            smoke.wait_for_page(cdp,
                "state.raceId==='2023-Mexico-R'&&state.driver==='44'&&state.sampleOrder==='largest-error'")
            view_url = cdp.evaluate("location.href")
            cdp.evaluate("document.getElementById('downloadViewProvenance').click()")
            deadline = time.monotonic() + 8
            manifest_files = []
            while time.monotonic() < deadline:
                manifest_files = [path for path in downloads.iterdir()
                                  if path.is_file() and not path.name.endswith('.crdownload')]
                if manifest_files:
                    break
                time.sleep(0.1)
            assert len(manifest_files) == 1, f"Expected one provenance JSON download, got {manifest_files}"
            manifest_path = manifest_files[0]
            manifest = json.loads(manifest_path.read_text())
            season = json.loads((frontend / "data/season_model.json").read_text())
            points = season["prediction_sample"]["points"]
            matching = [p for p in points if p["race_id"] == "2023-Mexico-R" and str(p["driver"]) == "44"]
            assert manifest["run_id"] == season["run_id"]
            assert manifest["source"] == season["source"] and manifest["model"] == season["model"]
            assert manifest["target"] == season["target"]
            assert manifest["view"] == {"race_id": "2023-Mexico-R", "driver_number": 44,
                                        "sample_order": "largest-error",
                                        "ordering_scope": "filtered deterministic sample only"}
            assert manifest["counts"] == {
                "full_held_out_node_targets": season["metrics"]["sample_count"],
                "prediction_sample_source_rows": season["prediction_sample"]["source_rows"],
                "deterministic_sample_points": len(points), "filtered_sample_points": len(matching),
                "train_graph_windows": season["split"]["train_window_count"],
                "validation_graph_windows": season["split"]["validation_window_count"],
                "test_graph_windows": season["split"]["test_window_count"],
            }
            assert manifest["sampling_rule"] == season["prediction_sample"]["method"]
            assert manifest["split"] == {"method": season["split"]["method"],
                                         "test_races": season["split"]["test_races"]}
            assert manifest["checkpoint"] == {key: season["checkpoint"][key]
                                                for key in ("artifact", "selected_epoch", "sha256")}
            assert manifest["provenance"] == {"fingerprint": season["provenance"]["fingerprint"],
                "config_sha256": season["provenance"]["config_sha256"],
                "prepared_artifact_sha256": season["provenance"]["prepared_artifact_sha256"],
                "implementation_sha256": season["provenance"]["implementation_sha256"],
                "source_artifact_sha256": season["provenance"]["source_artifact_sha256"]}
            assert manifest["predictions_included"] is False
            assert all("prediction" not in key.lower() and "rows" not in key.lower()
                       for key in manifest.keys() if key not in ("predictions_included",))
            assert "race=2023-Mexico-R" in view_url and "driver=44" in view_url and "order=largest-error" in view_url
            assert manifest_path.name == "hammertime-" + season["run_id"][:12] + "-2023-Mexico-R-driver-44-largest-error-view-provenance.json"
            print(f"PASS filtered provenance JSON: {manifest_path.name}, {len(matching)} matching sampled rows, URL={view_url}")

            # The driver list is global, so a valid race/driver pair may have no
            # sampled chart points. Verify one such real selectable pair remains
            # a metadata-exportable view, without altering the archived payload.
            cdp.evaluate(
                "(() => { raceFilter.value='2023-Mexico-R'; driverFilter.value='11'; "
                "raceFilter.dispatchEvent(new Event('change')); })()"
            )
            smoke.wait_for_page(cdp,
                "state.raceId==='2023-Mexico-R'&&state.driver==='11'&&matchingPoints().length===0")
            empty_view = evaluate_json(cdp,
                "({raceDisabled:raceFilter.disabled,driverDisabled:driverFilter.disabled,driverValue:driverFilter.value,"
                "sampleLabel:sampleLabel.textContent,csvDisabled:document.getElementById('downloadSampleCsv').disabled,"
                "exportDisabled:document.getElementById('downloadViewProvenance').disabled,"
                "url:location.href})")
            assert not empty_view["raceDisabled"] and not empty_view["driverDisabled"]
            assert empty_view["csvDisabled"] and not empty_view["exportDisabled"] and "0 points" in empty_view["sampleLabel"]
            assert "race=2023-Mexico-R" in empty_view["url"] and "driver=11" in empty_view["url"]
            cdp.evaluate("document.getElementById('downloadViewProvenance').click()")
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                zero_files = [path for path in downloads.iterdir()
                              if path.is_file() and not path.name.endswith('.crdownload')
                              and '-driver-11-' in path.name]
                if zero_files:
                    break
                time.sleep(0.1)
            assert len(zero_files) == 1, f"Expected one zero-match provenance JSON: {list(downloads.iterdir())}"
            zero_manifest = json.loads(zero_files[0].read_text())
            assert zero_manifest["view"]["race_id"] == "2023-Mexico-R"
            assert zero_manifest["view"]["driver_number"] == 11
            assert zero_manifest["counts"]["filtered_sample_points"] == 0
            assert zero_manifest["counts"]["deterministic_sample_points"] == len(points)
            assert zero_manifest["run_id"] == season["run_id"] and zero_manifest["provenance"] == manifest["provenance"]
            assert zero_manifest["predictions_included"] is False
            assert all("prediction" not in key.lower() and "rows" not in key.lower()
                       for key in zero_manifest.keys() if key != "predictions_included")
            print(f"PASS real selectable zero-match view: Mexico/11, export enabled, filtered count 0, URL={empty_view['url']}")

            # A failed fetch leaves export controls disabled instead of enabling
            # a metadata artifact from missing or fallback data.
            cdp.command("Network.enable")
            cdp.command("Network.setBlockedURLs", {"urls": ["*season_model.json*"]})
            cdp.command("Page.reload", {"ignoreCache": True})
            smoke.wait_for_page(cdp, "loadStatus.textContent==='Evaluation unavailable'")
            assert cdp.evaluate("document.getElementById('downloadViewProvenance').disabled")
            print("PASS failed payload keeps view-provenance export disabled")
            print("PASS point-inspection smoke complete")
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
