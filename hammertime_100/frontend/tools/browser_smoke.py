#!/usr/bin/env python3
"""Smoke-test the static HammerTime results page with headless Chromium.

Uses only the Python standard library and Chrome DevTools Protocol. It starts a
temporary localhost server, so the page's relative JSON request behaves as it
does when hosted. It does not modify project data or source files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import signal
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[3]
FRONTEND = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / ".local/research-history/2026-11-02"


class DevTools:
    def __init__(self, websocket_url: str) -> None:
        url = urlparse(websocket_url)
        self.sock = socket.create_connection((url.hostname, url.port), timeout=8)
        self.sock.settimeout(8)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {url.path} HTTP/1.1\r\nHost: {url.hostname}:{url.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            "Origin: http://localhost\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        headers = bytearray()
        while not headers.endswith(b"\r\n\r\n"):
            headers.extend(self.sock.recv(1))
        if b"101 " not in headers:
            raise RuntimeError(f"DevTools WebSocket handshake failed: {headers!r}")
        self._next_id = 0
        self.events: list[dict] = []

    def _read_exactly(self, count: int) -> bytes:
        result = bytearray()
        while len(result) < count:
            part = self.sock.recv(count - len(result))
            if not part:
                raise EOFError("Chromium closed the DevTools socket")
            result.extend(part)
        return bytes(result)

    def _receive(self) -> dict:
        first, second = self._read_exactly(2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exactly(8))[0]
        mask = self._read_exactly(4) if second & 0x80 else None
        payload = self._read_exactly(length)
        if mask:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if opcode == 0x9:  # ping
            self._send_frame(0xA, payload)
            return {}
        if opcode == 0x8:
            raise EOFError("Chromium closed the DevTools socket")
        return json.loads(payload)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        length = len(payload)
        mask = os.urandom(4)
        if length < 126:
            header = bytes([0x80 | opcode, 0x80 | length])
        elif length < 65536:
            header = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack("!H", length)
        else:
            header = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack("!Q", length)
        encoded = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + mask + encoded)

    def command(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        command_id = self._next_id
        payload = json.dumps({"id": command_id, "method": method, "params": params or {}}).encode()
        self._send_frame(0x1, payload)
        while True:
            message = self._receive()
            if message.get("id") == command_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})
            if message:
                self.events.append(message)

    def evaluate(self, expression: str) -> object:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
        )
        if "exceptionDetails" in result:
            raise RuntimeError(f"Browser evaluation failed: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def screenshot(self, path: Path) -> None:
        encoded = self.command("Page.captureScreenshot", {"format": "png"})["data"]
        path.write_bytes(base64.b64decode(encoded))

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        self.sock.close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_for_page(devtools: DevTools, expression: str, timeout: float = 12) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = devtools.evaluate(expression)
        if value:
            return value
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for browser condition: {expression}")


def expected_sample_rows(race: str, driver: str, order: str, limit: int = 40) -> list[list[str]]:
    sample = json.loads((FRONTEND / "data/season_model.json").read_text(encoding="utf-8"))["prediction_sample"]["points"]
    selected = [(index, point) for index, point in enumerate(sample)
                if (race == "all" or point["race_id"] == race)
                and (driver == "all" or str(point["driver"]) == driver)]
    if order == "largest-error":
        selected.sort(key=lambda item: (-abs(item[1]["predicted"] - item[1]["actual"]), item[0]))
    elif order == "smallest-error":
        selected.sort(key=lambda item: (abs(item[1]["predicted"] - item[1]["actual"]), item[0]))
    rows = []
    for _, point in selected[:limit]:
        race_name = point["race_id"].removeprefix("2023-").removesuffix("-R").replace("-", " ")
        rows.append([race_name, str(point["driver"]), str(point["lap"]), str(point["mini_sector"] + 1),
                     f"{point['actual']:.5f}", f"{point['predicted']:.5f}",
                     f"{point['predicted'] - point['actual']:.5f}"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", default=os.environ.get("CHROMIUM", "chromium"),
                        help="Chromium executable (default: chromium or $CHROMIUM)")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS,
                        help=f"Directory for viewport screenshots (default: {DEFAULT_ARTIFACTS})")
    parser.add_argument("--skip-screenshots", action="store_true", help="Do not save viewport screenshots")
    args = parser.parse_args()

    browser = shutil.which(args.chromium)
    if not browser:
        parser.error(f"Chromium executable not found: {args.chromium}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    http_port, debug_port = free_port(), free_port()
    http_url = f"http://127.0.0.1:{http_port}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-browser-smoke-") as temp:
        profile_dir = Path(temp) / "profile"
        download_dir = Path(temp) / "downloads"
        profile_dir.mkdir()
        download_dir.mkdir()
        http = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(http_port), "--bind", "127.0.0.1",
             "--directory", str(FRONTEND)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        chrome = subprocess.Popen(
            [browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-extensions",
             f"--remote-debugging-port={debug_port}", "--remote-allow-origins=*",
             f"--user-data-dir={profile_dir}", http_url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        devtools = None
        try:
            endpoint = f"http://127.0.0.1:{debug_port}/json/list"
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    pages = json.load(urllib.request.urlopen(endpoint, timeout=1))
                    page = next((item for item in pages if item.get("type") == "page"
                                 and item.get("url", "").startswith(http_url)), None)
                    if page:
                        devtools = DevTools(page["webSocketDebuggerUrl"])
                        break
                except (OSError, ValueError, StopIteration):
                    time.sleep(0.2)
            if devtools is None:
                raise TimeoutError("Chromium did not expose the served page via DevTools")

            devtools.command("Runtime.enable")
            devtools.command("Log.enable")
            devtools.command("Page.enable")
            devtools.command("Page.reload", {"ignoreCache": True})
            status = wait_for_page(devtools, "(() => { const e=document.querySelector('#loadStatus'); return e && e.textContent !== 'Loading archived run…' ? e.textContent : null; })()")
            if status != "Run loaded · archived metrics":
                detail = devtools.evaluate("JSON.stringify({status:loadStatus.textContent,error:dataError.textContent,metric:metricMae.textContent})")
                raise RuntimeError(f"Initial page load failed: {detail}")
            check(status == "Run loaded · archived metrics", f"Unexpected loaded status: {status}")
            values = json.loads(devtools.evaluate(
                "JSON.stringify({mae:metricMae.textContent,r2:metricR2.textContent,targets:metricN.textContent,"
                "raceOptions:raceFilter.options.length,driverOptions:driverFilter.options.length,"
                "raceRows:raceTableBody.querySelectorAll('tr').length})"
            ))
            check(values["mae"] == "0.09046 s" and values["r2"] == "0.03314",
                  f"Unexpected summary metrics: {values}")
            check(values["targets"] == "37,956" and values["raceOptions"] == 5
                  and values["driverOptions"] == 21 and values["raceRows"] == 4,
                  f"Unexpected loaded record/options counts: {values}")
            print(f"PASS loaded run: {values}")

            # The keyboard skip link must expose the main content and place
            # focus there so subsequent Tab presses continue in the page.
            devtools.evaluate("document.activeElement.blur()")
            devtools.command("Input.dispatchKeyEvent", {
                "type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9,
            })
            devtools.command("Input.dispatchKeyEvent", {
                "type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9,
            })
            skip_focus = devtools.evaluate(
                "JSON.stringify({className:document.activeElement.className,text:document.activeElement.textContent.trim(),"
                "top:document.activeElement.getBoundingClientRect().top})"
            )
            skip_focus = json.loads(skip_focus)
            check("skip-link" in skip_focus["className"] and "Skip to model results" in skip_focus["text"]
                  and skip_focus["top"] >= 0, f"Keyboard Tab did not reveal the skip link: {skip_focus}")
            for event_type in ("keyDown", "keyUp"):
                devtools.command("Input.dispatchKeyEvent", {
                    "type": event_type, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13,
                })
            skip_result = json.loads(wait_for_page(devtools,
                "(() => location.hash==='#main' ? JSON.stringify({hash:location.hash,activeId:document.activeElement.id,"
                "mainTop:document.querySelector('main').getBoundingClientRect().top}) : null)()"))
            check(skip_result["activeId"] == "main",
                  f"Skip link moved the viewport without transferring focus into main content: {skip_result}")
            print(f"PASS keyboard skip link moves focus into main content: {skip_focus} -> {skip_result}")

            distribution = json.loads(devtools.evaluate(
                "JSON.stringify({title:document.querySelector('#full-error-title').textContent,"
                "population:document.querySelector('#errorPopulation').textContent,"
                "note:document.querySelector('#errorDistributionNote').textContent,"
                "headers:[...document.querySelectorAll('.error-distribution-table thead th')].map(x=>x.textContent.trim()),"
                "rows:[...document.querySelectorAll('#errorDistributionBody tr')].map(r=>[...r.children].map(c=>c.textContent.trim()))})"
            ))
            source_distribution = json.loads((FRONTEND / "data/season_model.json").read_text())["error_distribution"]
            expected_rows = [[b["label"], f'{b["stgnn_count"]:,}', f'{b["zero_predictor_count"]:,}']
                             for b in source_distribution["bins"]]
            check(distribution["title"] == "Absolute residual error distribution",
                  f"Unexpected full-population distribution title: {distribution['title']}")
            check(distribution["headers"] == ["Absolute error band (seconds)", "HammerTime ST-GNN", "Zero predictor"],
                  f"Unexpected distribution headers: {distribution['headers']}")
            check("37,956 targets" in distribution["population"] and "full held-out set" in distribution["population"],
                  f"Distribution population label is unclear: {distribution['population']}")
            check("Maximum absolute error: ST-GNN 5.43064 s" in distribution["note"]
                  and "zero predictor 5.51703 s" in distribution["note"],
                  f"Distribution observed maxima are missing or unexpected: {distribution['note']}")
            check(distribution["rows"] == expected_rows,
                  f"Rendered distribution differs from archived bins: {distribution['rows']}")
            stgnn_sum = sum(int(row[1].replace(",", "")) for row in distribution["rows"])
            zero_sum = sum(int(row[2].replace(",", "")) for row in distribution["rows"])
            check(len(distribution["rows"]) == 8 and stgnn_sum == 37956 and zero_sum == 37956,
                  f"Distribution bins do not cover both full populations: n={len(distribution['rows'])}, sums={stgnn_sum}/{zero_sum}")
            check(distribution["rows"][0] == ["≤ 0.05", "19,148", "19,133"]
                  and distribution["rows"][-1] == ["> 5.00", "1", "1"],
                  f"Unexpected first/tail bin counts: {distribution['rows'][0]}, {distribution['rows'][-1]}")
            shared_scale = json.loads(devtools.evaluate(
                "JSON.stringify({widths:[...document.querySelectorAll('#errorDistributionBody tr:first-child .error-count-fill')].map(e=>parseFloat(e.style.width)),"
                "tableWidth:document.querySelector('.error-distribution-table').scrollWidth})"
            ))
            check(len(shared_scale["widths"]) == 2 and abs(shared_scale["widths"][0] - 100) < 1e-8
                  and abs(shared_scale["widths"][1] - (19133 / 19148 * 100)) < 1e-4,
                  f"Distribution bars do not use one shared count scale: {shared_scale}")
            print(f"PASS full-population error distribution: 8 labeled bins; ST-GNN/zero totals={stgnn_sum}/{zero_sum}; first={distribution['rows'][0]}, tail={distribution['rows'][-1]}")
            print(f"PASS distribution columns share a common bar scale: first-bin widths={shared_scale['widths']}")

            if not args.skip_screenshots:
                devtools.command("Emulation.setDeviceMetricsOverride", {
                    "width": 1440, "height": 1200, "deviceScaleFactor": 1, "mobile": False,
                })
                time.sleep(0.2)
                devtools.screenshot(args.artifacts_dir / "frontend-smoke-wide.png")
                devtools.command("Emulation.setDeviceMetricsOverride", {
                    "width": 390, "height": 1000, "deviceScaleFactor": 1, "mobile": True,
                })
                time.sleep(0.2)
                devtools.screenshot(args.artifacts_dir / "frontend-smoke-mobile.png")

            marker = devtools.command("Page.addScriptToEvaluateOnNewDocument", {
                "source": "window.__htAuditXss = 0;",
            })["identifier"]
            hostile_url = (http_url + "?race=%3Cimg%20src%3D%2Fmissing%20onerror%3Dwindow.__htAuditXss%3D1%3E"
                           "&driver=%3Csvg%20onload%3Dwindow.__htAuditXss%3D2%3E&order=invalid")
            devtools.command("Page.navigate", {"url": hostile_url})
            wait_for_page(devtools, "(() => { const e=document.querySelector('#loadStatus'); return e && e.textContent === 'Run loaded · archived metrics' ? e.textContent : null; })()")
            hostile = json.loads(devtools.evaluate(
                "JSON.stringify({search:location.search,race:raceFilter.value,driver:driverFilter.value,order:sampleOrderFilter.value,"
                "xss:window.__htAuditXss,hostileNodes:document.querySelectorAll('img[onerror],svg[onload]').length})"
            ))
            check(hostile["search"] == "" and hostile["race"] == "all" and hostile["driver"] == "all"
                  and hostile["order"] == "source" and hostile["xss"] == 0 and hostile["hostileNodes"] == 0,
                  f"Invalid/hostile query values were not safely normalized: {hostile}")
            devtools.command("Page.removeScriptToEvaluateOnNewDocument", {"identifier": marker})
            print(f"PASS invalid/hostile URL query normalization: {hostile}")

            # Race selector updates the complete-race metrics and plot sample.
            race = json.loads(devtools.evaluate(
                "(() => { const e=raceFilter; e.value='2023-Mexico-R'; e.dispatchEvent(new Event('change')); "
                "return JSON.stringify({title:selectedRaceTitle.textContent,mae:raceMae.textContent,n:raceN.textContent,sample:sampleLabel.textContent}); })()"
            ))
            check(race["title"] == "Mexico" and race["mae"] == "0.08550 s" and race["n"] == "10,759"
                  and "284 / 37,956" in race["sample"], f"Mexico filter failed: {race}")
            print(f"PASS race filter: {race}")

            driver = json.loads(devtools.evaluate(
                "(() => { const e=driverFilter; e.value='44'; e.dispatchEvent(new Event('change')); "
                "return JSON.stringify({sample:sampleLabel.textContent,aria:predictionChart.getAttribute('aria-label'),"
                "raceN:raceN.textContent,raceMae:raceMae.textContent,context:document.querySelector('.race-footnote').textContent,url:location.search}); })()"
            ))
            check("Mexico · driver 44" in driver["sample"] and "18 archived sampled predictions" in driver["aria"],
                  f"Driver filter/plot description failed: {driver}")
            check(driver["raceN"] == "10,759" and driver["raceMae"] == "0.08550 s"
                  and "every valid target for that race" in driver["context"]
                  and driver["url"] == "?race=2023-Mexico-R&driver=44",
                  f"Driver filter changed the race-wide metrics/context or URL: {driver}")
            print(f"PASS driver filter, race-wide metric denominator, and share URL: {driver}")

            devtools.evaluate("history.back()")
            back_state = json.loads(wait_for_page(devtools,
                "(() => { const r=raceFilter?.value,d=driverFilter?.value; return r==='2023-Mexico-R'&&d==='all' ? JSON.stringify({race:r,driver:d,url:location.search,sample:sampleLabel.textContent,raceN:raceN.textContent}) : null; })()"))
            check("284 / 37,956" in back_state["sample"] and back_state["raceN"] == "10,759",
                  f"History Back did not restore the previous filters: {back_state}")
            devtools.evaluate("history.forward()")
            forward_state = json.loads(wait_for_page(devtools,
                "(() => { const r=raceFilter?.value,d=driverFilter?.value; return r==='2023-Mexico-R'&&d==='44' ? JSON.stringify({race:r,driver:d,url:location.search,sample:sampleLabel.textContent}) : null; })()"))
            check("18 / 37,956" in forward_state["sample"], f"History Forward did not restore filters: {forward_state}")
            print(f"PASS filter history Back/Forward: {back_state} -> {forward_state}")

            # Pagination is not a filter, so a history return re-renders the
            # restored filter state from its first page rather than stale rows.
            devtools.evaluate("(() => { raceFilter.value='all'; raceFilter.dispatchEvent(new Event('change')); driverFilter.value='all'; driverFilter.dispatchEvent(new Event('change')); for(let i=0;i<24;i++) predictionPageNext.click(); })()")
            last_page = devtools.evaluate("predictionPageStatus.textContent")
            last_page_rows = devtools.evaluate("predictionTableBody.querySelectorAll('tr').length")
            check("page 25 of 25" in last_page and last_page_rows == 39,
                  f"Last prediction page boundary failed: {last_page}, rows={last_page_rows}")
            devtools.evaluate("(() => { raceFilter.value='2023-Mexico-R'; raceFilter.dispatchEvent(new Event('change')); })()")
            devtools.evaluate("history.back()")
            history_page = json.loads(wait_for_page(devtools,
                "(() => { const r=raceFilter?.value,d=driverFilter?.value; return r==='all'&&d==='all' ? JSON.stringify({race:r,driver:d,sample:sampleLabel.textContent,page:predictionPageStatus.textContent}) : null; })()"))
            check("999 / 37,956" in history_page["sample"] and "page 1 of 25" in history_page["page"],
                  f"History restore left a stale pagination page: {history_page}")
            print(f"PASS last-page boundary and history pagination reset: {last_page}, rows={last_page_rows}, restored={history_page}")

            # Sorting affects only a copy of the deterministic sample used by
            # the table/CSV. It must not change the chart sample or page count.
            devtools.evaluate("predictionPageNext.click(); predictionPageNext.click();")
            check("page 3 of 25" in devtools.evaluate("predictionPageStatus.textContent"),
                  "Could not move to a later page before testing order reset")
            largest = json.loads(devtools.evaluate(
                "(() => { const chart=predictionChart.toDataURL(),label=sampleLabel.textContent;"
                "sampleOrderFilter.value='largest-error'; sampleOrderFilter.dispatchEvent(new Event('change'));"
                "return JSON.stringify({url:location.search,order:sampleOrderFilter.value,page:predictionPageStatus.textContent,"
                "sample:sampleLabel.textContent,chartUnchanged:predictionChart.toDataURL()===chart,labelUnchanged:sampleLabel.textContent===label,"
                "rows:[...predictionTableBody.rows].map(r=>[...r.cells].map(c=>c.textContent))})})()"
            ))
            expected_largest = expected_sample_rows("all", "all", "largest-error")
            check(largest["url"] == "?order=largest-error" and largest["order"] == "largest-error"
                  and "page 1 of 25" in largest["page"] and "999 / 37,956" in largest["sample"]
                  and largest["chartUnchanged"] and largest["labelUnchanged"]
                  and largest["rows"] == expected_largest,
                  f"Largest-error order changed sample/chart or failed page reset: {largest}")
            print(f"PASS largest-error order is a stable table-only sort with URL state: {largest['url']}, first={largest['rows'][0]}")

            smallest = json.loads(devtools.evaluate(
                "(() => { sampleOrderFilter.value='smallest-error'; sampleOrderFilter.dispatchEvent(new Event('change'));"
                "return JSON.stringify({url:location.search,order:sampleOrderFilter.value,page:predictionPageStatus.textContent,"
                "rows:[...predictionTableBody.rows].map(r=>[...r.cells].map(c=>c.textContent))})})()"
            ))
            expected_smallest = expected_sample_rows("all", "all", "smallest-error")
            check(smallest["url"] == "?order=smallest-error" and smallest["order"] == "smallest-error"
                  and "page 1 of 25" in smallest["page"] and smallest["rows"] == expected_smallest,
                  f"Smallest-error order failed stable ordering or page reset: {smallest}")
            devtools.evaluate("history.back()")
            order_back = json.loads(wait_for_page(devtools,
                "(() => sampleOrderFilter?.value==='largest-error' ? JSON.stringify({order:sampleOrderFilter.value,url:location.search,page:predictionPageStatus.textContent}) : null)()"))
            devtools.evaluate("history.forward()")
            order_forward = json.loads(wait_for_page(devtools,
                "(() => sampleOrderFilter?.value==='smallest-error' ? JSON.stringify({order:sampleOrderFilter.value,url:location.search,page:predictionPageStatus.textContent}) : null)()"))
            check("page 1 of 25" in order_back["page"] and "page 1 of 25" in order_forward["page"],
                  f"Order history did not restore at page one: {order_back} -> {order_forward}")
            devtools.evaluate("sampleOrderFilter.value='source'; sampleOrderFilter.dispatchEvent(new Event('change'));")
            source_order = json.loads(devtools.evaluate(
                "JSON.stringify({url:location.search,order:sampleOrderFilter.value,page:predictionPageStatus.textContent,"
                "rows:[...predictionTableBody.rows].map(r=>[...r.cells].map(c=>c.textContent))})"
            ))
            check(source_order["url"] == "" and source_order["order"] == "source"
                  and source_order["rows"] == expected_sample_rows("all", "all", "source"),
                  f"Source-order reset failed to restore the original sample order: {source_order}")
            tie_order = json.loads(devtools.evaluate(
                "(() => { const original=state.sampleOrder; state.sampleOrder='largest-error';"
                "const input=[{id:'first',actual:0,predicted:2},{id:'second',actual:0,predicted:-2},"
                "{id:'missing-a',actual:null,predicted:3},{id:'missing-b',actual:1,predicted:null}];"
                "const order=orderedSamplePoints(input).map(p=>p.id); state.sampleOrder=original; return JSON.stringify(order); })()"
            ))
            check(tie_order == ["first", "second", "missing-a", "missing-b"],
                  f"Equal-error ties or missing errors did not retain source order: {tie_order}")
            print(f"PASS smallest-error, history restoration, and source-order reset: {order_back} -> {order_forward}")

            direct_url = http_url + "?race=2023-Mexico-R&driver=44&order=largest-error"
            devtools.command("Page.navigate", {"url": direct_url})
            direct_state = json.loads(wait_for_page(devtools,
                "(() => { const e=document.querySelector('#loadStatus'); return e?.textContent==='Run loaded · archived metrics'&&raceFilter?.value==='2023-Mexico-R'&&driverFilter?.value==='44' ? JSON.stringify({url:location.search,race:raceFilter.value,driver:driverFilter.value,order:sampleOrderFilter.value,detailsOpen:predictionDataDetails.open,activeTag:document.activeElement.tagName,activeId:document.activeElement.id,sample:sampleLabel.textContent,n:raceN.textContent,rows:[...predictionTableBody.rows].map(r=>[...r.cells].map(c=>c.textContent))}) : null; })()"))
            check(direct_state["url"] == "?race=2023-Mexico-R&driver=44&order=largest-error"
                  and direct_state["order"] == "largest-error" and "18 / 37,956" in direct_state["sample"]
                  and direct_state["detailsOpen"] and direct_state["activeTag"] == "BODY"
                  and direct_state["activeId"] == ""
                  and direct_state["rows"] == expected_sample_rows("2023-Mexico-R", "44", "largest-error"),
                  f"Direct share URL did not restore filters: {direct_state}")
            print(f"PASS direct share URL: {direct_state}")

            devtools.command("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(download_dir)})
            csv_focus = json.loads(devtools.evaluate(
                "(() => {downloadSampleCsv.focus(); const s=getComputedStyle(downloadSampleCsv); return JSON.stringify({focused:document.activeElement===downloadSampleCsv,"
                "outline:s.outlineStyle,width:s.outlineWidth,disabled:downloadSampleCsv.disabled})})()"
            ))
            check(csv_focus["focused"] and csv_focus["outline"] != "none" and csv_focus["width"] != "0px"
                  and not csv_focus["disabled"], f"CSV button focus is not visible/available: {csv_focus}")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "char", "key": "Enter", "code": "Enter", "text": "\r", "unmodifiedText": "\r", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            csv_files: list[Path] = []
            download_deadline = time.monotonic() + 8
            while time.monotonic() < download_deadline:
                csv_files = [path for path in download_dir.iterdir()
                             if path.is_file() and not path.name.endswith(".crdownload")]
                if csv_files:
                    break
                time.sleep(0.1)
            check(len(csv_files) == 1, f"Expected one filtered CSV download, found: {csv_files}")
            with csv_files[0].open(encoding="utf-8-sig", newline="") as source:
                exported_rows = list(csv.DictReader(source))
                exported_header = list(exported_rows[0]) if exported_rows else []
            expected_header = ["model_run_id", "race_id", "driver_number", "lap", "mini_sector_index_zero_based",
                               "actual_residual_seconds", "predicted_residual_seconds", "prediction_error_seconds"]
            expected_csv_sample = expected_sample_rows("2023-Mexico-R", "44", "largest-error", limit=18)
            csv_sample_rows = [[row["race_id"], row["driver_number"], row["lap"],
                                row["mini_sector_index_zero_based"],
                                f"{float(row['actual_residual_seconds']):.5f}",
                                f"{float(row['predicted_residual_seconds']):.5f}",
                                f"{float(row['prediction_error_seconds']):.5f}"] for row in exported_rows]
            expected_csv_sample = [[f"2023-{row[0].replace(' ', '-')}-R", *row[1:3],
                                    str(int(row[3]) - 1), *row[4:]] for row in expected_csv_sample]
            check(csv_files[0].name == "hammertime-769a76aab164-2023-Mexico-R-driver-44-sampled-predictions.csv"
                  and exported_header == expected_header and len(exported_rows) == 18
                  and all(row["model_run_id"] == "769a76aab164b514cc0a" and row["race_id"] == "2023-Mexico-R" and row["driver_number"] == "44"
                          and 0 <= int(row["mini_sector_index_zero_based"]) < 20 for row in exported_rows)
                  and csv_sample_rows == expected_csv_sample,
                  f"Filtered CSV contents/scope/order failed: name={csv_files[0].name}, header={exported_header}, rows={len(exported_rows)}, "
                  f"actual_first={csv_sample_rows[:2]}, expected_first={expected_csv_sample[:2]}")
            download_status = devtools.evaluate("downloadStatus.textContent")
            check("18 filtered rows" in download_status and "999-point chart sample" in download_status
                  and "largest absolute residual error first" in download_status
                  and "37,956 targets" in download_status, f"CSV download status does not distinguish sample/full count: {download_status}")
            print(f"PASS filtered CSV: {csv_files[0].name}, rows={len(exported_rows)}, header={exported_header}; {download_status}")
            print(f"PASS CSV keyboard focus and Enter activation: {csv_focus}")

            # Round-trip every numeric value in the deterministic sample. This
            # guards against CSV formula-escaping accidentally converting
            # negative measurements into text while preserving row identity.
            csv_files[0].unlink()
            devtools.evaluate("(() => { raceFilter.value='all'; raceFilter.dispatchEvent(new Event('change')); driverFilter.value='all'; driverFilter.dispatchEvent(new Event('change')); sampleOrderFilter.value='source'; sampleOrderFilter.dispatchEvent(new Event('change')); })()")
            devtools.evaluate("downloadSampleCsv.click()")
            full_sample_files: list[Path] = []
            full_download_deadline = time.monotonic() + 8
            while time.monotonic() < full_download_deadline:
                full_sample_files = [path for path in download_dir.iterdir()
                                     if path.is_file() and not path.name.endswith(".crdownload")]
                if full_sample_files:
                    break
                time.sleep(0.1)
            check(len(full_sample_files) == 1, f"Expected one full sample CSV download, found: {full_sample_files}")
            with full_sample_files[0].open(encoding="utf-8-sig", newline="") as source:
                full_sample_rows = list(csv.DictReader(source))
            source_export = json.loads((FRONTEND / "data/season_model.json").read_text(encoding="utf-8"))
            source_points = source_export["prediction_sample"]["points"]
            check(len(full_sample_rows) == len(source_points) == 999,
                  f"Full sample CSV row count differs from JSON: {len(full_sample_rows)} / {len(source_points)}")
            negative_counts = {"actual": 0, "predicted": 0, "error": 0}
            tiny_value_count = 0
            for index, (row, point) in enumerate(zip(full_sample_rows, source_points)):
                check(row["model_run_id"] == source_export["run_id"]
                      and row["race_id"] == point["race_id"]
                      and row["driver_number"] == str(point["driver"])
                      and row["lap"] == str(point["lap"])
                      and row["mini_sector_index_zero_based"] == str(point["mini_sector"]),
                      f"Full sample CSV identity/order mismatch at row {index}: {row}")
                values = (
                    ("actual_residual_seconds", point["actual"], "actual"),
                    ("predicted_residual_seconds", point["predicted"], "predicted"),
                    ("prediction_error_seconds", point["predicted"] - point["actual"], "error"),
                )
                for column, expected, label in values:
                    text_value = row[column]
                    check(bool(text_value) and not text_value.startswith(("'", "−")),
                          f"Numeric CSV value was text escaped at row {index}, {column}: {text_value!r}")
                    observed = float(text_value)
                    check(abs(observed - float(expected)) <= 1e-15,
                          f"Full sample CSV numeric mismatch at row {index}, {column}: {observed} != {expected}")
                    negative_counts[label] += observed < 0
                    tiny_value_count += abs(observed) < 1e-6
            check(full_sample_files[0].name == "hammertime-769a76aab164-all-races-all-drivers-sampled-predictions.csv",
                  f"Unexpected full sample CSV filename: {full_sample_files[0].name}")
            print(f"PASS full sample CSV round-trip: 999 rows match JSON identities/order/numeric values within 1e-15; "
                  f"negative counts={negative_counts}, values <1e-6={tiny_value_count}")
            devtools.evaluate("(() => { sampleOrderFilter.value='largest-error'; sampleOrderFilter.dispatchEvent(new Event('change')); })()")

            # Simulate a valid sparse chart sample with an available race/driver
            # filter combination that has no sampled rows; full race metrics
            # must remain visible while sample export becomes unavailable.
            empty_state = json.loads(devtools.evaluate(
                "(() => { state.run.prediction_sample.points=state.run.prediction_sample.points.filter(p=>p.race_id==='2023-Mexico-R'&&String(p.driver)==='44'); "
                "raceFilter.value='2023-AbuDhabi-R'; raceFilter.dispatchEvent(new Event('change')); driverFilter.value='4'; driverFilter.dispatchEvent(new Event('change')); "
                "return JSON.stringify({sample:sampleLabel.textContent,aria:predictionChart.getAttribute('aria-label'),n:raceN.textContent,mae:raceMae.textContent,"
                "context:document.querySelector('.race-footnote').textContent,table:predictionPageStatus.textContent,order:sampleOrderFilter.value,downloadDisabled:downloadSampleCsv.disabled}); })()"
            ))
            check("0 points" in empty_state["sample"] and "No sampled predictions" in empty_state["aria"]
                  and empty_state["n"] == "10,609" and empty_state["mae"] == "0.09416 s"
                  and "every valid target for that race" in empty_state["context"]
                  and "0 filtered sampled predictions" in empty_state["table"]
                  and empty_state["order"] == "largest-error"
                  and empty_state["downloadDisabled"], f"Zero-sample filter state failed: {empty_state}")
            print(f"PASS zero sampled-row filter with race-wide denominator and disabled CSV: {empty_state}")
            devtools.command("Page.navigate", {"url": http_url + "?race=2023-Mexico-R&driver=44"})
            wait_for_page(devtools, "(() => { const e=document.querySelector('#loadStatus'); return e?.textContent==='Run loaded · archived metrics' ? e.textContent : null; })()")

            breakdown = json.loads(devtools.evaluate(
                "(() => { const e=breakdownFilter; e.value='mini_sector'; e.dispatchEvent(new Event('change')); "
                "return JSON.stringify({title:breakdownTitle.textContent,count:breakdownList.querySelectorAll('.breakdown-row').length,"
                "first:breakdownList.querySelector('.breakdown-row')?.textContent}); })()"
            ))
            check(breakdown["title"] == "By mini-sector" and breakdown["count"] == 20
                  and breakdown["first"].startswith("Mini-sector 1"), f"Mini-sector breakdown failed: {breakdown}")
            print(f"PASS breakdown toggle: {breakdown}")

            # Keyboard focus, native labels, chart text alternative, and aggregate table.
            filtered_rows = devtools.evaluate("predictionTableBody.querySelectorAll('tr').length")
            check(filtered_rows == 18, f"Prediction table did not follow active race/driver filters: {filtered_rows}")
            devtools.evaluate("(() => { raceFilter.value='all'; driverFilter.value='all'; raceFilter.dispatchEvent(new Event('change')); })()")
            keyboard = json.loads(devtools.evaluate(
                "(() => { raceFilter.focus(); const table=document.querySelector('.prediction-data table'); "
                "return JSON.stringify({focused:document.activeElement.id,"
                "raceLabels:raceFilter.labels.length,driverLabels:driverFilter.labels.length,orderLabels:sampleOrderFilter.labels.length,breakdownLabels:breakdownFilter.labels.length,"
                "chartRole:predictionChart.getAttribute('role'),chartLabel:!!predictionChart.getAttribute('aria-label'),"
                "aggregateRows:raceTableBody.querySelectorAll('tr').length,"
                "predictionCaption:table?.caption?.textContent,pointRows:predictionTableBody.querySelectorAll('tr').length,"
                "pointColumns:table?.querySelectorAll('thead th').length,rowColumns:predictionTableBody.querySelector('tr')?.cells.length}); })()"
            ))
            check(keyboard["focused"] == "raceFilter" and keyboard["raceLabels"] == 1
                  and keyboard["driverLabels"] == 1 and keyboard["breakdownLabels"] == 1
                  and keyboard["chartRole"] == "img" and keyboard["chartLabel"]
                  and keyboard["aggregateRows"] == 4 and keyboard["pointRows"] == 40
                  and keyboard["pointColumns"] == 7 and keyboard["rowColumns"] == 7
                  and "Deterministic sample rows" in keyboard["predictionCaption"],
                  f"Accessibility structure check failed: {keyboard}")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            next_focus = devtools.evaluate("document.activeElement.id")
            check(next_focus == "driverFilter", f"Tab did not move race selector focus to driver selector: {next_focus}")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            summary_focus = devtools.evaluate("document.activeElement.id")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "char", "key": "Enter", "code": "Enter", "text": "\r", "unmodifiedText": "\r", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            details_open = devtools.evaluate("predictionRowsSummary.parentElement.open")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            order_focus = devtools.evaluate("document.activeElement.id")
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
            csv_button_focus = devtools.evaluate("document.activeElement.id")
            check(order_focus == "sampleOrderFilter" and keyboard["orderLabels"] == 1
                  and summary_focus == "predictionRowsSummary" and details_open
                  and csv_button_focus == "downloadSampleCsv",
                  f"Keyboard order/table navigation failed: {summary_focus} -> {order_focus} -> {csv_button_focus}, open={details_open}")
            devtools.evaluate("predictionPageNext.click()")
            page_status = devtools.evaluate("predictionPageStatus.textContent")
            check("page 2 of" in page_status, f"Keyboard activation of prediction pager failed: {page_status}")
            print(f"PASS keyboard/label/chart-table structure: {keyboard}; Tab focus -> {next_focus} -> {summary_focus}; Enter opens table -> {order_focus} -> {csv_button_focus}; pager -> {page_status}")

            for width, height, mobile in [(1440, 1000, False), (390, 844, True), (320, 740, True)]:
                devtools.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": 1, "mobile": mobile,
                })
                metrics = json.loads(devtools.evaluate(
                    "JSON.stringify({inner:innerWidth,document:document.documentElement.scrollWidth,"
                    "body:document.body.scrollWidth,chart:Math.round(predictionChart.getBoundingClientRect().width),"
                    "nav:Math.round(document.querySelector('header nav').getBoundingClientRect().width),"
                    "detailsOpen:predictionDataDetails.open,orderLeft:Math.round(sampleOrderFilter.getBoundingClientRect().left),"
                    "orderRight:Math.round(sampleOrderFilter.getBoundingClientRect().right),orderWidth:Math.round(sampleOrderFilter.getBoundingClientRect().width)})"
                ))
                check(metrics["document"] <= metrics["inner"] and metrics["body"] <= metrics["inner"]
                      and metrics["detailsOpen"] and metrics["orderLeft"] >= 0
                      and metrics["orderRight"] <= width + 1 and metrics["orderWidth"] > 0,
                      f"Horizontal overflow or sample-order control clipping at {width}px: {metrics}")
                print(f"PASS responsive width {width}: {metrics}")

                if width in (390, 320):
                    region_state = json.loads(devtools.evaluate(
                        "(() => {const p=document.querySelector('.prediction-data .table-scroll');const r=document.querySelector('#raceTableBody').closest('.table-scroll');"
                        "return JSON.stringify([p,r].map(e=>({role:e.getAttribute('role'),label:e.getAttribute('aria-label'),tabIndex:e.tabIndex,"
                        "client:e.clientWidth,scroll:e.scrollWidth,outline:(e.focus(),getComputedStyle(e).outlineWidth)})))})()"
                    ))
                    keyboard_scroll = []
                    for region_selector in (".prediction-data .table-scroll", "#raceTableBody"):
                        if region_selector == "#raceTableBody":
                            devtools.evaluate("document.querySelector('#raceTableBody').closest('.table-scroll').focus(); document.querySelector('#raceTableBody').closest('.table-scroll').scrollLeft=0")
                        else:
                            devtools.evaluate("document.querySelector('.prediction-data .table-scroll').focus(); document.querySelector('.prediction-data .table-scroll').scrollLeft=0")
                        devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39})
                        devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39})
                        expression = "document.querySelector('.prediction-data .table-scroll')" if region_selector.startswith(".") else "document.querySelector('#raceTableBody').closest('.table-scroll')"
                        right_offset = devtools.evaluate(f"{expression}.scrollLeft")
                        devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "ArrowLeft", "code": "ArrowLeft", "windowsVirtualKeyCode": 37})
                        devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "ArrowLeft", "code": "ArrowLeft", "windowsVirtualKeyCode": 37})
                        left_offset = devtools.evaluate(f"{expression}.scrollLeft")
                        keyboard_scroll.append((right_offset, left_offset))
                    check(len(region_state) == 2 and all(x["role"] == "region" and x["label"] and x["tabIndex"] == 0
                          and x["scroll"] > x["client"] and x["outline"] != "0px" for x in region_state)
                          and all(right > 0 and left == 0 for right, left in keyboard_scroll),
                          f"Scrollable data tables are not keyboard accessible at {width}px: {region_state}, scrolls={keyboard_scroll}")
                    print(f"PASS keyboard-scrollable named table regions at {width}px: {region_state}; ArrowRight offsets={keyboard_scroll}")

            # Anchored navigation and initial fragment loads must leave the
            # destination below the sticky header, including its taller mobile layout.
            devtools.command("Emulation.setEmulatedMedia", {
                "features": [{"name": "prefers-reduced-motion", "value": "reduce"}],
            })
            anchor_results = []
            for width in (1440, 390, 320):
                devtools.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": 844, "deviceScaleFactor": 1, "mobile": width < 600,
                })
                for target in ("results", "explorer", "diagnostics", "method"):
                    devtools.evaluate(f"document.querySelector('header nav a[href=\\\"#{target}\\\"]').click()")
                    position = json.loads(devtools.evaluate(
                        f"JSON.stringify((()=>{{const e=document.querySelector('#{target}'),h=document.querySelector('.topbar');"
                        f"return {{width:{width},target:'{target}',top:Math.round(e.getBoundingClientRect().top),header:Math.round(h.getBoundingClientRect().bottom)}}}})())"
                    ))
                    check(position["top"] >= position["header"],
                          f"Section link {target} is hidden under sticky header at {width}px: {position}")
                    anchor_results.append(position)
                devtools.command("Page.navigate", {"url": http_url + "#method"})
                wait_for_page(devtools, "(() => loadStatus?.textContent==='Run loaded · archived metrics' ? loadStatus.textContent : null)()")
                initial = json.loads(devtools.evaluate(
                    f"JSON.stringify((()=>{{const e=document.querySelector('#method'),h=document.querySelector('.topbar');"
                    f"return {{width:{width},target:'initial #method',top:Math.round(e.getBoundingClientRect().top),header:Math.round(h.getBoundingClientRect().bottom)}}}})())"
                ))
                check(initial["top"] >= initial["header"],
                      f"Initial #method fragment is hidden under sticky header at {width}px: {initial}")
                anchor_results.append(initial)
            print(f"PASS sticky-header navigation visibility for 4 links and initial #method at 1440/390/320px: {anchor_results}")

            # Expand every disclosure at narrow viewports: hashes, point table,
            # and method/target details must wrap or remain inside scroll regions.
            for width in (390, 320):
                devtools.command("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": 740, "deviceScaleFactor": 1, "mobile": True,
                })
                disclosures = json.loads(devtools.evaluate(
                    "(() => { const all=[...document.querySelectorAll('details')]; all.forEach(d=>d.open=true);"
                    "const hashes=[...document.querySelectorAll('.hash-details dd')];"
                    "return JSON.stringify({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth,"
                    "details:all.length,opened:all.filter(d=>d.open).length,hashOverflow:hashes.filter(e=>e.scrollWidth>e.clientWidth+1).length,"
                    "hashWidth:hashes.length?Math.max(...hashes.map(e=>e.getBoundingClientRect().right)):0,"
                    "tableWraps:[...document.querySelectorAll('.table-scroll')].map(e=>({client:e.clientWidth,scroll:e.scrollWidth,right:e.getBoundingClientRect().right})),"
                    "distribution:{client:document.querySelector('.error-distribution-scroll').clientWidth,scroll:document.querySelector('.error-distribution-scroll').scrollWidth,"
                    "table:document.querySelector('.error-distribution-table').scrollWidth,bars:getComputedStyle(document.querySelector('.error-count-track')).display,"
                    "captionAlign:getComputedStyle(document.querySelector('.error-distribution-table caption')).textAlign,"
                    "captionText:document.querySelector('.error-distribution-table caption').textContent.trim(),"
                    "cells:document.querySelectorAll('#errorDistributionBody tr:first-child td').length}}); })()"
                ))
                check(disclosures["body"] <= width and disclosures["document"] <= width
                      and disclosures["opened"] == disclosures["details"]
                      and disclosures["hashOverflow"] == 0
                      and disclosures["hashWidth"] <= width + 1
                      and all(item["right"] <= width + 1 for item in disclosures["tableWraps"])
                      and disclosures["distribution"]["scroll"] <= disclosures["distribution"]["client"] + 1
                      and disclosures["distribution"]["table"] <= disclosures["distribution"]["client"] + 1
                      and disclosures["distribution"]["bars"] == "none"
                      and disclosures["distribution"]["captionAlign"] == "left"
                      and disclosures["distribution"]["cells"] == 2,
                      f"Expanded disclosure overflow at {width}px: {disclosures}")
                devtools.evaluate("document.querySelector('.full-error-panel').scrollIntoView({block:'start'})")
                if not args.skip_screenshots:
                    devtools.screenshot(args.artifacts_dir / f"error-distribution-mobile-{width}.png")
                print(f"PASS expanded disclosures at {width}px: {disclosures}")

            # Measure representative rendered text/background pairs. This is a
            # targeted contrast spot-check, not a full WCAG audit.
            contrast = json.loads(devtools.evaluate(
                "(() => {const lum=s=>{const m=s.match(/[\\d.]+/g).map(Number).slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2);return .2126*m[0]+.7152*m[1]+.0722*m[2]};"
                "const bg=e=>{for(let n=e;n;n=n.parentElement){const c=getComputedStyle(n).backgroundColor;if(c&&c!=='rgba(0, 0, 0, 0)'&&c!=='transparent')return c;}return getComputedStyle(document.body).backgroundColor};"
                "const pair=s=>{const e=document.querySelector(s);if(!e)return null;const fg=getComputedStyle(e).color,b=bg(e),a=lum(fg),z=lum(b);return {selector:s,fg,bg:b,ratio:((Math.max(a,z)+.05)/(Math.min(a,z)+.05)).toFixed(2)}};"
                "return JSON.stringify([pair('.metric-card p'),pair('.muted-note'),pair('.gate-pill.fail'),pair('#errorDistributionNote')]);})()"
            ))
            print(f"INFO targeted contrast spot-check (4.5:1 reference for ordinary text; not full WCAG audit): {contrast}")

            # Test focus visibility and native Enter activation on a provenance disclosure.
            focus_check = json.loads(devtools.evaluate(
                "(() => {const s=document.querySelector('.hash-details summary');const d=s.parentElement;d.open=false;s.focus();const c=getComputedStyle(s);return JSON.stringify({summary:s.textContent.trim(),tabIndex:s.tabIndex,outlineStyle:c.outlineStyle,outlineWidth:c.outlineWidth,open:d.open})})()"
            ))
            devtools.command("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "char", "key": "Enter", "code": "Enter", "text": "\r", "unmodifiedText": "\r", "windowsVirtualKeyCode": 13})
            devtools.command("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
            focus_check["openAfterEnter"] = devtools.evaluate("document.querySelector('.hash-details').open")
            check(focus_check["tabIndex"] == 0 and focus_check["outlineStyle"] != "none"
                  and focus_check["outlineWidth"] != "0px" and focus_check["openAfterEnter"],
                  f"Provenance disclosure lacks visible keyboard operation: {focus_check}")
            print(f"PASS focused provenance disclosure is visibly keyboard operable: {focus_check}")

            devtools.command("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
            motion = json.loads(devtools.evaluate(
                "JSON.stringify({preference:matchMedia('(prefers-reduced-motion: reduce)').matches,scrollBehavior:getComputedStyle(document.documentElement).scrollBehavior,"
                "summaryTransition:getComputedStyle(document.querySelector('.hash-details summary')).transitionDuration})"
            ))
            check(motion["preference"] and motion["scrollBehavior"] == "auto",
                  f"Reduced-motion styling is not applied: {motion}")
            print(f"PASS reduced-motion preference: {motion}")

            normal_errors = [event for event in devtools.events if
                             event.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded")
                             and (event.get("method") == "Runtime.exceptionThrown"
                                  or (event.get("params", {}).get("entry", {}).get("level") == "error"
                                      and not event.get("params", {}).get("entry", {}).get("url", "").endswith("/favicon.ico")))]
            check(not normal_errors, f"Browser reported runtime errors: {normal_errors}")

            # Each schema mutation happens only in the browser response wrapper;
            # the archived source files remain untouched.
            def assert_malformed_payload(label: str, mutation: str) -> None:
                interceptor = (
                    "const sourceFetch=window.fetch.bind(window);"
                    "window.fetch=async(...args)=>{const response=await sourceFetch(...args);"
                    "if(String(args[0]).includes('season_model.json')){const data=await response.json();"
                    f"{mutation};"
                    "return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});"
                    "}return response;};"
                )
                script_id = devtools.command("Page.addScriptToEvaluateOnNewDocument", {
                    "source": interceptor,
                })["identifier"]
                devtools.command("Page.reload", {"ignoreCache": True})
                malformed = json.loads(wait_for_page(devtools,
                    "(() => { const e=document.querySelector('#loadStatus'); return e?.textContent==='Evaluation unavailable' ? JSON.stringify({status:e.textContent,error:dataError.textContent,hidden:dataError.hidden,role:dataError.getAttribute('role'),metric:metricN.textContent,aria:predictionChart.getAttribute('aria-label')}) : null; })()"))
                check(malformed["metric"] == "—" and malformed["hidden"] is False
                      and malformed["role"] == "alert" and "No fallback or synthetic metrics" in malformed["error"]
                      and "unavailable" in malformed["aria"].lower(),
                      f"Malformed payload {label} was not rejected clearly: {malformed}")
                print(f"PASS malformed payload rejected ({label}): {malformed}")
                devtools.command("Page.removeScriptToEvaluateOnNewDocument", {"identifier": script_id})
                devtools.command("Page.navigate", {"url": http_url})
                wait_for_page(devtools, "(() => { const e=document.querySelector('#loadStatus'); return e?.textContent==='Run loaded · archived metrics' ? e.textContent : null; })()")

            assert_malformed_payload("missing target denominator", "delete data.metrics.sample_count")
            assert_malformed_payload("non-numeric sampled target", "data.prediction_sample.points[0].actual='not-a-number'")
            assert_malformed_payload("empty point sample", "data.prediction_sample.points=[]")

            devtools.command("Page.addScriptToEvaluateOnNewDocument", {
                "source": "window.fetch = () => Promise.reject(new Error('audit forced fetch failure'));",
            })
            devtools.command("Page.reload", {"ignoreCache": True})
            failure = wait_for_page(devtools, "(() => { const e=document.querySelector('#loadStatus'); return e && e.textContent === 'Evaluation unavailable' ? JSON.stringify({status:e.textContent,error:dataError.textContent,alertRole:dataError.getAttribute('role'),hidden:dataError.hidden,metric:metricMae.textContent,aria:predictionChart.getAttribute('aria-label')}) : null; })()")
            failure = json.loads(failure)
            check(failure["metric"] == "—" and "No fallback or synthetic metrics" in failure["error"]
                  and failure["alertRole"] == "alert" and failure["hidden"] is False
                  and "unavailable" in failure["aria"].lower(), f"Fetch failure state failed: {failure}")
            print(f"PASS failed-fetch state: {failure}")
            print("PASS browser smoke checks complete")
            return 0
        finally:
            if devtools:
                devtools.close()
            for process in (chrome, http):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, TimeoutError, OSError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
