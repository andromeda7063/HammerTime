#!/usr/bin/env python3
"""Browser smoke check for the static experiment registry; saves local screenshots."""
from __future__ import annotations

import argparse
import base64
import csv
import io
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


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rgb_channels(value):
    import re
    match = re.fullmatch(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)", value)
    if not match:
        raise AssertionError(f"Unexpected computed color: {value}")
    return [int(channel) / 255 for channel in match.groups()]


def contrast_ratio(foreground, background):
    def luminance(color):
        values = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb_channels(color)]
        return 0.2126 * values[0] + 0.7152 * values[1] + 0.0722 * values[2]
    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


class CDP:
    def __init__(self, url):
        u = urlparse(url)
        self.s = socket.create_connection((u.hostname, u.port), timeout=8)
        self.s.settimeout(8)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
                        f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                        "Sec-WebSocket-Version: 13\r\nOrigin: http://localhost\r\n\r\n").encode())
        headers = bytearray()
        while not headers.endswith(b"\r\n\r\n"):
            headers.extend(self.s.recv(1))
        if b"101 " not in headers:
            raise RuntimeError("Chromium DevTools handshake failed")
        self.ident = 0

    def exact(self, n):
        b = bytearray()
        while len(b) < n:
            b.extend(self.s.recv(n - len(b)))
        return bytes(b)

    def send(self, payload):
        mask = os.urandom(4); n = len(payload)
        header = bytes([0x81, 0x80 | n]) if n < 126 else bytes([0x81, 0x80 | 126]) + struct.pack("!H", n)
        self.s.sendall(header + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def recv(self):
        first, second = self.exact(2); n = second & 127
        if n == 126: n = struct.unpack("!H", self.exact(2))[0]
        elif n == 127: n = struct.unpack("!Q", self.exact(8))[0]
        mask = self.exact(4) if second & 128 else None
        data = self.exact(n)
        if mask: data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
        if first & 15 == 9:
            self.send(data); return self.recv()
        return json.loads(data)

    def call(self, method, params=None):
        self.ident += 1; i = self.ident
        self.send(json.dumps({"id": i, "method": method, "params": params or {}}).encode())
        while True:
            msg = self.recv()
            if msg.get("id") == i:
                if "error" in msg: raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result: raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def screenshot(self, target):
        target.write_bytes(base64.b64decode(self.call("Page.captureScreenshot", {"format": "png"})["data"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM", "chromium"))
    ap.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = ap.parse_args()
    browser = shutil.which(args.chromium)
    if not browser: ap.error(f"Chromium executable not found: {args.chromium}")
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    hp, dp = port(), port(); base = f"http://127.0.0.1:{hp}/"
    with tempfile.TemporaryDirectory(prefix="hammertime-experiments-") as profile:
        server = subprocess.Popen([sys.executable, "-m", "http.server", str(hp), "--bind", "127.0.0.1", "--directory", str(FRONTEND)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        chrome = subprocess.Popen([browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-extensions", "--disable-dev-shm-usage", f"--remote-debugging-port={dp}", "--remote-allow-origins=*", f"--user-data-dir={profile}", base + "experiments.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        cdp = None
        try:
            endpoint = f"http://127.0.0.1:{dp}/json/list"; deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    pages = json.load(urllib.request.urlopen(endpoint, timeout=1))
                    page = next((p for p in pages if p.get("type") == "page" and p.get("url", "").endswith("experiments.html")), None)
                    if page: cdp = CDP(page["webSocketDebuggerUrl"]); break
                except (OSError, ValueError): time.sleep(.2)
            if not cdp: raise TimeoutError("Chromium did not expose registry page")
            cdp.call("Runtime.enable"); cdp.call("Page.enable")
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                if cdp.evaluate("document.querySelectorAll('.ex-group').length > 0"): break
                time.sleep(.2)
            values = json.loads(cdp.evaluate("JSON.stringify({groups:document.querySelectorAll('.ex-group').length,rows:[...document.querySelectorAll('.ex-table tbody tr')].length,summary:document.querySelector('#result-count').textContent,bodyWidth:document.body.scrollWidth})"))
            assert values["groups"] == 9 and values["rows"] == 23, values
            assert "23 records" in values["summary"], values
            record_links = json.loads(cdp.evaluate("JSON.stringify([...document.querySelectorAll('.ex-record-link')].map(a=>({id:a.closest('tr').id,href:a.href,model:a.textContent,key:a.dataset.sourceKey,group:a.closest('.ex-group').querySelector('h2').id.slice(6)})) )"))
            assert len(record_links) == 23 and len({link['id'] for link in record_links}) == 23, "record IDs are missing or duplicated"
            expected_labels = {
                'legacy_input_legacy_graph': 'Legacy inputs + legacy graph',
                'circuit_context_legacy_graph': 'Circuit context + legacy graph',
                'circuit_context_physical_scalar': 'Circuit context + physical scalar',
                'circuit_context_physical_v1': 'Circuit context + physical graph v1',
                'circuit_context_self_loop_only': 'Circuit context + self-loop only',
            }
            readable_rows = json.loads(cdp.evaluate("JSON.stringify([...document.querySelectorAll('.ex-group h2')].filter(h=>h.textContent.includes('Hungary')).flatMap(h=>{const g=h.closest('.ex-group'); return [...g.querySelectorAll('.ex-record-link')].map(a=>({id:a.closest('tr').id,key:a.dataset.sourceKey,label:a.textContent,detail:a.closest('td').querySelector('.ex-variant').textContent}))}))"))
            readable_candidates = [row for row in readable_rows if row['key'] in expected_labels]
            assert {row['key']: row['label'] for row in readable_candidates} == expected_labels and all(row['key'] in row['detail'] for row in readable_candidates), readable_candidates
            direct_link = next(link for link in record_links if link['group'] == 'openf1-hungary-circuit-screen' and link['key'] == 'circuit_context_physical_scalar')
            readable_link = next(link for link in record_links if link['group'] == 'openf1-hungary-circuit-screen' and link['key'] == 'circuit_context_physical_v1')
            assert readable_link['model'] == 'Circuit context + physical graph v1' and 'circuit_context_physical_v1' in cdp.evaluate(f"document.getElementById('{readable_link['id']}').querySelector('.ex-variant').textContent"), readable_link
            assert f"group=openf1-hungary-circuit-screen#{direct_link['id']}" in direct_link['href'], direct_link
            cdp.call("Emulation.setDeviceMetricsOverride", {"width":320,"height":900,"deviceScaleFactor":1,"mobile":True})
            cdp.call("Page.navigate", {"url": direct_link['href']})
            deep_state = None; deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                try:
                    raw = cdp.evaluate("JSON.stringify((()=>{const id=decodeURIComponent(location.hash.slice(1)); const row=document.getElementById(id); const box=row?.getBoundingClientRect(); return {group:new URL(location.href).searchParams.get('group'),id,rows:document.querySelectorAll('.ex-table tbody tr').length,groupCount:document.querySelectorAll('.ex-group').length,target:row?.classList.contains('is-deep-link-target')||false,focused:document.activeElement?.id===id,visible:!!box&&box.top<innerHeight&&box.bottom>0,bodyWidth:document.body.scrollWidth,viewport:innerWidth,scrollY:scrollY}})())")
                    deep_state = json.loads(raw)
                    if deep_state['target'] and deep_state['focused']: break
                except (RuntimeError, ValueError): pass
                time.sleep(.2)
            assert deep_state and deep_state['group'] == 'openf1-hungary-circuit-screen' and deep_state['id'] == direct_link['id'] and deep_state['rows'] == 9 and deep_state['groupCount'] == 1 and deep_state['target'] and deep_state['focused'] and deep_state['visible'] and deep_state['bodyWidth'] <= deep_state['viewport'] + 1, deep_state
            cdp.evaluate("(() => { history.pushState({},'', '?group=openf1-hungary-circuit-screen#unknown-record'); dispatchEvent(new PopStateEvent('popstate')); })()")
            unknown_hash = json.loads(cdp.evaluate("JSON.stringify({groups:document.querySelectorAll('.ex-group').length,rows:document.querySelectorAll('.ex-table tbody tr').length,focusedRow:document.activeElement?.matches('tr.ex-run-row')||false})"))
            assert unknown_hash == {"groups":1,"rows":9,"focusedRow":False}, unknown_hash
            cdp.evaluate(f"(() => {{ history.pushState({{}},'', '?group=fastf1-season-test#{direct_link['id']}'); dispatchEvent(new PopStateEvent('popstate')); }})()")
            hidden_hash = json.loads(cdp.evaluate("JSON.stringify({groups:document.querySelectorAll('.ex-group').length,rows:document.querySelectorAll('.ex-table tbody tr').length,focusedRow:document.activeElement?.matches('tr.ex-run-row')||false})"))
            assert hidden_hash == {"groups":1,"rows":2,"focusedRow":False}, hidden_hash
            cdp.evaluate("document.querySelector('#reset-filters').click(); scrollTo(0,0)")
            reset_deep_link = json.loads(cdp.evaluate("JSON.stringify({query:location.search,hash:location.hash,rows:document.querySelectorAll('.ex-table tbody tr').length})"))
            assert reset_deep_link == {"query":"","hash":"","rows":23}, reset_deep_link
            cdp.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 850, "deviceScaleFactor": 1, "mobile": False})
            time.sleep(.25); cdp.screenshot(args.artifacts_dir / "experiments-wide.png")
            cdp.evaluate("(() => { const e=document.querySelector('#family-filter'); e.value='ST-GNN'; e.dispatchEvent(new Event('change')); })()")
            rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert rows == 14, f"ST-GNN family filter returned {rows} rows"
            url_family = cdp.evaluate("new URL(location.href).searchParams.get('family')")
            assert url_family == "ST-GNN", f"family selection not shared in URL: {url_family}"
            cdp.evaluate("(() => { const e=document.querySelector('#family-filter'); e.value='all'; const s=document.querySelector('#status-filter'); s.value='held-out test'; e.dispatchEvent(new Event('change')); s.dispatchEvent(new Event('change')); })()")
            rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert rows == 2, f"held-out test filter returned {rows} rows"
            cdp.evaluate("(() => { const s=document.querySelector('#status-filter'); s.value='all'; const g=document.querySelector('#group-filter'); g.value='openf1-hungary-circuit-screen'; s.dispatchEvent(new Event('change')); g.dispatchEvent(new Event('change')); })()")
            rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert rows == 9, f"evaluation group filter returned {rows} rows"
            cdp.evaluate("(() => { const s=document.querySelector('#sort-filter'); s.value='mae-asc'; s.dispatchEvent(new Event('change')); })()")
            mae_sort = json.loads(cdp.evaluate("JSON.stringify({value:document.querySelector('#sort-filter').value,param:new URL(location.href).searchParams.get('sort'),groupIds:[...document.querySelectorAll('.ex-group h2')].map(e=>e.id.slice(6)),scores:[...document.querySelectorAll('.ex-table tbody tr')].map(r=>Number(r.cells[1].textContent))})"))
            assert mae_sort['value'] == mae_sort['param'] == 'mae-asc' and mae_sort['groupIds'] == ['openf1-hungary-circuit-screen'] and mae_sort['scores'] == sorted(mae_sort['scores']), mae_sort
            cdp.evaluate("new Promise(resolve=>{addEventListener('popstate',resolve,{once:true}); history.back()})")
            back_sort = cdp.evaluate("document.querySelector('#sort-filter').value")
            cdp.evaluate("new Promise(resolve=>{addEventListener('popstate',resolve,{once:true}); history.forward()})")
            forward_sort = cdp.evaluate("document.querySelector('#sort-filter').value")
            assert back_sort == 'source' and forward_sort == 'mae-asc', (back_sort, forward_sort)
            csv_sorted = cdp.evaluate("(async()=>{let blob; const original=URL.createObjectURL; URL.createObjectURL=(value)=>{blob=value;return original.call(URL,value)}; document.querySelector('#download-csv').click(); URL.createObjectURL=original; return await blob.text()})()")
            sorted_rows = list(csv.DictReader(io.StringIO(csv_sorted)))
            csv_mae = [float(row['mae_sec']) for row in sorted_rows]
            assert len(sorted_rows) == 9 and all(row['sort_within_group'] == 'mae-asc' for row in sorted_rows) and csv_mae == sorted(csv_mae), sorted_rows
            cdp.evaluate("(() => { const s=document.querySelector('#sort-filter'); s.value='r2-desc'; s.dispatchEvent(new Event('change')); })()")
            r2_sort = json.loads(cdp.evaluate("JSON.stringify({param:new URL(location.href).searchParams.get('sort'),scores:[...document.querySelectorAll('.ex-table tbody tr')].map(r=>Number(r.cells[3].textContent))})"))
            assert r2_sort['param'] == 'r2-desc' and r2_sort['scores'] == sorted(r2_sort['scores'], reverse=True) and r2_sort['scores'][0] < 0 and r2_sort['scores'][-1] < r2_sort['scores'][0], r2_sort
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='physical'; q.dispatchEvent(new Event('input')); })()")
            rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert rows == 2, f"search filter returned {rows} rows"
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='physical scalar'; q.dispatchEvent(new Event('input')); })()")
            normalized_rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert normalized_rows == 1, f"underscore-to-space search returned {normalized_rows} rows"
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='physical graph v1'; q.dispatchEvent(new Event('input')); })()")
            readable_search = json.loads(cdp.evaluate("JSON.stringify({rows:document.querySelectorAll('.ex-table tbody tr').length,label:document.querySelector('.ex-record-link')?.textContent||''})"))
            assert readable_search == {"rows":1,"label":"Circuit context + physical graph v1"}, readable_search
            readable_id = cdp.evaluate("document.querySelector('.ex-record-link')?.closest('tr').id||''")
            assert readable_id == readable_link['id'], (readable_id, readable_link)
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='circuit_context_physical_v1'; q.dispatchEvent(new Event('input')); })()")
            raw_key_search = json.loads(cdp.evaluate("JSON.stringify({rows:document.querySelectorAll('.ex-table tbody tr').length,id:document.querySelector('.ex-record-link')?.closest('tr').id||''})"))
            assert raw_key_search == {"rows":1,"id":readable_link['id']}, raw_key_search
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='physical'; q.dispatchEvent(new Event('input')); })()")
            cdp.evaluate("new Promise(resolve=>{addEventListener('popstate',resolve,{once:true}); history.back()})")
            back_state = json.loads(cdp.evaluate("JSON.stringify({group:document.querySelector('#group-filter').value,q:document.querySelector('#search-filter').value,rows:document.querySelectorAll('.ex-table tbody tr').length})"))
            assert back_state["group"] == "openf1-hungary-circuit-screen" and back_state["q"] == "" and back_state["rows"] == 9, back_state
            cdp.evaluate("new Promise(resolve=>{addEventListener('popstate',resolve,{once:true}); history.forward()})")
            forward_q = cdp.evaluate("document.querySelector('#search-filter').value")
            assert forward_q == "physical", f"forward navigation did not restore search: {forward_q}"
            csv_data = cdp.evaluate("(async()=>{let blob; const original=URL.createObjectURL; URL.createObjectURL=(value)=>{blob=value; return original.call(URL,value)}; document.querySelector('#download-csv').click(); URL.createObjectURL=original; return await blob.text()})()")
            assert 'source_sha256' in csv_data and 'circuit_context_physical_v1' in csv_data, "filtered CSV is missing provenance or expected row"
            cdp.evaluate("(() => { history.pushState({},'', '?group=not-a-real-group&sort=unknown'); dispatchEvent(new PopStateEvent('popstate')); })()")
            invalid = json.loads(cdp.evaluate("JSON.stringify({group:document.querySelector('#group-filter').value,sort:document.querySelector('#sort-filter').value,hint:document.querySelector('#filter-message').textContent,shown:!document.querySelector('#filter-message').hidden})"))
            assert invalid["group"] == "all" and invalid["sort"] == "source" and invalid["shown"] and "Unknown URL filter" in invalid["hint"], invalid
            normalized = cdp.evaluate("JSON.stringify({group:new URL(location.href).searchParams.has('group'),sort:new URL(location.href).searchParams.has('sort')})")
            assert json.loads(normalized) == {"group": False, "sort": False}, f"invalid URL filters were not cleared: {normalized}"
            cdp.evaluate("(() => { const f=document.querySelector('#family-filter'); const s=document.querySelector('#status-filter'); f.value='all'; s.value='all'; f.dispatchEvent(new Event('change')); s.dispatchEvent(new Event('change')); const g=document.querySelector('#group-filter'); g.value='all'; g.dispatchEvent(new Event('change')); })()")
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value='Sao Paulo'; q.dispatchEvent(new Event('input')); })()")
            accent_rows = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert accent_rows == 2, f"accent-insensitive Sao Paulo search returned {accent_rows} rows"
            cdp.evaluate("(() => { const q=document.querySelector('#search-filter'); q.value=''; q.dispatchEvent(new Event('input')); const f=document.querySelector('#family-filter'); const s=document.querySelector('#status-filter'); const g=document.querySelector('#group-filter'); f.value='ST-GNN'; s.value='pilot'; g.value='fastf1-monza-pilots'; f.dispatchEvent(new Event('change')); s.dispatchEvent(new Event('change')); g.dispatchEvent(new Event('change')); q.value='current k3'; q.dispatchEvent(new Event('input')); })()")
            reset_before = cdp.evaluate("document.querySelectorAll('.ex-table tbody tr').length")
            assert reset_before == 1, f"pre-reset selected-filter count was {reset_before}"
            cdp.evaluate("document.querySelector('#reset-filters').click()")
            reset_state = json.loads(cdp.evaluate("JSON.stringify({family:document.querySelector('#family-filter').value,evidence:document.querySelector('#status-filter').value,group:document.querySelector('#group-filter').value,q:document.querySelector('#search-filter').value,rows:document.querySelectorAll('.ex-table tbody tr').length,query:location.search})"))
            assert reset_state["family"] == reset_state["evidence"] == reset_state["group"] == "all" and reset_state["q"] == "" and reset_state["rows"] == 23 and not reset_state["query"], reset_state
            reset_sort = cdp.evaluate("document.querySelector('#sort-filter').value")
            assert reset_sort == 'source', reset_sort
            cdp.evaluate("(() => { const s=document.querySelector('#sort-filter'); s.value='r2-desc'; s.dispatchEvent(new Event('change')); })()")
            sorted_groups = json.loads(cdp.evaluate("JSON.stringify({ids:[...document.querySelectorAll('.ex-group h2')].map(e=>e.id.slice(6)),badNulls:[...document.querySelectorAll('.ex-group')].flatMap(g=>{const vals=[...g.querySelectorAll('tbody tr')].map(r=>r.cells[3].textContent.trim()); const first=vals.indexOf('—'); return first<0||vals.slice(first).every(v=>v==='—')?[]:[g.querySelector('h2').id, vals]})})"))
            expected_group_order = [group['id'] for group in json.loads((FRONTEND / 'data/experiments.json').read_text())['groups']]
            assert sorted_groups['ids'] == expected_group_order and not sorted_groups['badNulls'], sorted_groups
            csv_all_sorted = cdp.evaluate("(async()=>{let blob; const original=URL.createObjectURL; URL.createObjectURL=(value)=>{blob=value;return original.call(URL,value)}; document.querySelector('#download-csv').click(); URL.createObjectURL=original; return await blob.text()})()")
            csv_records = list(csv.DictReader(io.StringIO(csv_all_sorted)))
            source_groups = json.loads((FRONTEND / 'data/experiments.json').read_text())['groups']
            expected_records = {(g['id'], r['model'], r['variant']): r for g in source_groups for r in g['runs']}
            assert len(csv_records) == 23, f"CSV round-trip returned {len(csv_records)} rows"
            for row in csv_records:
                expected = expected_records[(row['group_id'], row['model'], row['variant'])]
                for key in ('mae_sec', 'rmse_sec', 'r2_score', 'sample_count'):
                    value = expected.get(key)
                    if value is None:
                        assert row[key] == '', (row, key)
                    else:
                        assert abs(float(row[key]) - float(value)) <= 1e-15, (row, key, value)
                        assert not row[key].startswith(("'", '−')), (row, key)
            failed_rows = [row for row in csv_records if row['evidence'] == 'failed']
            assert len(failed_rows) == 2 and all(all(row[key] == '' for key in ('mae_sec','rmse_sec','r2_score','sample_count')) for row in failed_rows), failed_rows
            cdp.evaluate("document.querySelector('#reset-filters').click()")
            region = json.loads(cdp.evaluate("JSON.stringify({role:document.querySelector('.ex-table-wrap').getAttribute('role'),tabindex:document.querySelector('.ex-table-wrap').tabIndex,caption:!!document.querySelector('.ex-table caption'),described:!!document.querySelector('.ex-table-wrap').getAttribute('aria-describedby')})"))
            assert region == {"role":"region","tabindex":0,"caption":True,"described":True}, region
            cdp.call("Emulation.setDeviceMetricsOverride", {"width":390,"height":1000,"deviceScaleFactor":1,"mobile":True})
            keyboard_scroll = json.loads(cdp.evaluate("(() => { const r=document.querySelector('.ex-table-wrap'); r.focus(); r.scrollLeft=80; const direct=r.scrollLeft; r.scrollLeft=0; const e=new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true,cancelable:true}); r.dispatchEvent(e); return JSON.stringify({scroll:r.scrollLeft,direct,width:r.scrollWidth,client:r.clientWidth,active:document.activeElement===r,cancelled:e.defaultPrevented,overflow:getComputedStyle(r).overflowX,direction:getComputedStyle(r).direction}) })()"))
            assert keyboard_scroll["scroll"] > 0 and keyboard_scroll["cancelled"], f"focused table region did not scroll by keyboard: {keyboard_scroll}"
            cdp.evaluate("(() => { const r=document.querySelector('.ex-table-wrap'); r.scrollLeft=0; r.blur(); scrollTo(0,0); })()")
            time.sleep(.25); cdp.screenshot(args.artifacts_dir / "experiments-mobile.png")
            widths = json.loads(cdp.evaluate("JSON.stringify({body:document.body.scrollWidth,viewport:innerWidth,table:document.querySelector('.ex-table-wrap').scrollWidth,tableClient:document.querySelector('.ex-table-wrap').clientWidth})"))
            assert widths["body"] <= widths["viewport"] + 1, widths
            cdp.evaluate("(() => { const d=document.querySelector('.ex-sources'); d.open=false; d.querySelector('summary').focus(); })()")
            cdp.call("Input.dispatchKeyEvent", {"type":"keyDown","key":"Enter","code":"Enter","windowsVirtualKeyCode":13,"nativeVirtualKeyCode":13,"text":"\r"})
            cdp.call("Input.dispatchKeyEvent", {"type":"keyUp","key":"Enter","code":"Enter","windowsVirtualKeyCode":13,"nativeVirtualKeyCode":13})
            source_open = cdp.evaluate("document.querySelector('.ex-sources').open")
            assert source_open is True, "Enter did not toggle the focused source details summary"
            focus_style = json.loads(cdp.evaluate("(() => { const s=document.querySelector('.ex-sources summary'); const c=getComputedStyle(s); return JSON.stringify({outlineStyle:c.outlineStyle,outlineWidth:c.outlineWidth,outlineColor:c.outlineColor}) })()"))
            assert focus_style["outlineStyle"] != "none" and float(focus_style["outlineWidth"].replace("px", "")) >= 2, focus_style
            colors = json.loads(cdp.evaluate("(() => { const group=document.querySelector('.ex-group'); const variant=document.querySelector('.ex-variant'); const badge=document.querySelector('.ex-badge[data-evidence=failed]'); const summary=document.querySelector('.ex-sources summary'); const th=document.querySelector('.ex-table th'); return JSON.stringify({panel:getComputedStyle(group).backgroundColor,variant:getComputedStyle(variant).color,failed:getComputedStyle(badge).color,summary:getComputedStyle(summary).color,th:getComputedStyle(th).color,thBg:getComputedStyle(th).backgroundColor}) })()"))
            ratios = {
                "variant": contrast_ratio(colors["variant"], colors["panel"]),
                "failed badge": contrast_ratio(colors["failed"], colors["panel"]),
                "source summary": contrast_ratio(colors["summary"], colors["panel"]),
                "table heading": contrast_ratio(colors["th"], colors["thBg"]),
            }
            assert all(ratio >= 4.5 for ratio in ratios.values()), ratios
            cdp.evaluate("document.querySelectorAll('.ex-sources').forEach(d=>d.open=true)")
            detail_widths = {}
            for width in (390, 320):
                cdp.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":1000,"deviceScaleFactor":1,"mobile":True})
                detail_widths[width] = cdp.evaluate("JSON.stringify({body:document.body.scrollWidth,viewport:innerWidth,openDetails:document.querySelectorAll('.ex-sources[open]').length})")
                detail_widths[width] = json.loads(detail_widths[width])
                assert detail_widths[width]["body"] <= detail_widths[width]["viewport"] + 1 and detail_widths[width]["openDetails"] == 9, detail_widths[width]
            cdp.evaluate("document.querySelectorAll('.ex-sources').forEach(d=>d.open=false)")
            cdp.call("Emulation.setDeviceMetricsOverride", {"width":390,"height":1000,"deviceScaleFactor":1,"mobile":True})
            bad_data = "{schema_version:1,groups:[{id:'bad',title:'Bad',provider:'x',evaluation:'test',protocol:'test',evidence:'held-out test',note:'test',sources:[{path:'source',sha256:'" + "a"*64 + "'}],runs:[{model:'bad',family:'ST-GNN',variant:'bad',status:'scored',mae_sec:null,rmse_sec:null,r2_score:0,sample_count:1,sample_unit:'rows',source_paths:[]}]}]}"
            cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source":f"window.fetch=async()=>({{ok:true,json:async()=>({bad_data})}});"})
            cdp.call("Page.reload", {"ignoreCache":True})
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                invalid_data_state = cdp.evaluate("JSON.stringify({alert:document.querySelector('#registry .ex-error')?.textContent || '',groups:document.querySelectorAll('.ex-group').length})")
                if invalid_data_state and 'data validation failed' in json.loads(invalid_data_state)['alert']: break
                time.sleep(.2)
            invalid_data_state = json.loads(invalid_data_state)
            assert "data validation failed" in invalid_data_state["alert"] and invalid_data_state["groups"] == 0, invalid_data_state
            print(f"PASS registry load and counts: {values}")
            print(f"PASS evidence/model/group filters: ST-GNN=14, held-out=2, Hungary=9; search: physical=2, physical scalar=1, Sao Paulo=2")
            print(f"PASS within-group MAE/R² order, null-last behavior, stable group order, sort URL/back-forward/CSV: MAE={mae_sort['scores']}, R²={r2_sort['scores']}")
            print(f"PASS reset clears family/evidence/group/search/sort URL state: {reset_state}")
            print(f"PASS URL back/forward, 23-row CSV precision/null blank fields, invalid-filter hint, accessible table caption/region, keyboard scroll={keyboard_scroll}")
            print(f"PASS 390px viewport has no page overflow; table scroll region: {widths}")
            print(f"PASS keyboard Enter source disclosure with visible focus: {focus_style}; text contrast spot-check: {ratios}")
            print(f"PASS 9 disclosures open at 390px and 320px without page overflow: {detail_widths}")
            print("PASS malformed score payload fails closed with no rendered metrics")
            print(f"Screenshots: {args.artifacts_dir / 'experiments-wide.png'}, {args.artifacts_dir / 'experiments-mobile.png'}")
        finally:
            if cdp:
                try: cdp.s.close()
                except OSError: pass
            for proc in (chrome, server):
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
