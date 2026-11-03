#!/usr/bin/env python3
"""Build the static, provenance-bearing experiment registry from archived results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("experiments.json")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path: Path) -> dict:
    return {"path": rel(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_exact_report_rows(path: Path, section_start: str, section_end: str, expected_rows: tuple[str, ...], required_marker: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        start = text.index(section_start)
        end = text.index(section_end, start + len(section_start))
    except ValueError as error:
        raise ValueError(f"Refusing report-only scores: expected section markers changed in {rel(path)}") from error
    block = text[start:end]
    if required_marker and required_marker not in block:
        raise ValueError(f"Refusing report-only scores: required disclosure changed in {rel(path)}")
    rows = {line.strip().removesuffix("\\\\").strip() for line in block.splitlines()}
    missing = [row for row in expected_rows if row not in rows]
    if missing:
        raise ValueError(f"Refusing report-only scores: expected exact table rows missing in {rel(path)}: {missing}")


def run(model, family, variant, metrics=None, status="scored", note="", source_paths=(), unit="valid target rows"):
    metrics = metrics or {}
    return {
        "model": model,
        "family": family,
        "variant": variant,
        "status": status,
        "status_note": note,
        "mae_sec": metrics.get("mae_sec"),
        "rmse_sec": metrics.get("rmse_sec"),
        "r2_score": metrics.get("r2_score"),
        "sample_count": metrics.get("sample_count"),
        "sample_unit": unit,
        "source_paths": list(source_paths),
    }


def group(id, title, provider, evaluation, protocol, evidence, note, paths, runs):
    files = [ROOT / p for p in paths]
    return {
        "id": id,
        "title": title,
        "provider": provider,
        "evaluation": evaluation,
        "protocol": protocol,
        "evidence": evidence,
        "note": note,
        "sources": [source(p) for p in files],
        "runs": runs,
    }


def main():
    hist = ROOT / ".local/research-history/2026-10-31"
    season_dir = ROOT / ".local/research-history/2023-fastf1-season/training/769a76aab164b514cc0a"
    season_metric = season_dir / "test_metrics.json"
    season_train = season_dir / "training_complete.json"
    if not season_train.exists():
        season_train = season_dir / "training_summary.json"
    season_doc = read_json(season_metric)
    season = season_doc["metrics"]
    # Archived evaluator uses the top-level metric keys; validate the identity and split at build time.
    run_id = season_doc.get("run_id", "769a76aab164b514cc0a")
    if run_id != "769a76aab164b514cc0a":
        raise ValueError(f"Unexpected season run id: {run_id}")
    season_sources = [rel(season_metric)]
    if season_train.exists():
        season_sources.append(rel(season_train))
    race_names = {"2023-AbuDhabi-R": "Abu Dhabi", "2023-LasVegas-R": "Las Vegas", "2023-Mexico-R": "Mexico", "2023-SãoPaulo-R": "São Paulo"}
    heldout_races = ", ".join(race_names.get(race_id, race_id) for race_id in season_doc["split"]["test"]["race_ids"])
    clean_air = next(item for item in season["physics_tests"] if item["test"].startswith("Clean Air"))
    drs_check = next(item for item in season["physics_tests"] if item["test"].startswith("DRS Train"))
    season_note = (
        f"{season['sample_count']:,} valid node-target predictions ({season_doc['split']['test']['sample_count']:,} graph windows; different count unit). "
        f"Matched zero baseline uses the same target rows. MAE gate {'passes' if season['target_mae_pass'] else 'fails'}; "
        f"R² gate {'passes' if season['target_r2_pass'] else 'fails'}. "
        f"Clean-air check {'passes' if clean_air['passed'] else 'fails'} ({clean_air['observed_mae']:.5f} s over {clean_air['sample_count']:,}); "
        f"DRS is {'unmeasurable' if drs_check['complete_laps'] == 0 else ('passes' if drs_check['passed'] else 'fails')} ({drs_check['complete_laps']} complete held-out laps). "
        f"Test races: {heldout_races}."
    )
    season_runs = [
        run("ST-GNN · season run 769a76aab164b514cc0a", "ST-GNN", "FastF1 2023 race-held-out test", season, status="held-out test", source_paths=season_sources),
        run("Zero predictor", "Baseline", "Matched zero predictor", season.get("zero_baseline", {}), status="held-out test", source_paths=season_sources),
    ]
    groups = [group(
        "fastf1-season-test", "FastF1 · 2023 season held-out test", "FastF1", "Test · 4 unseen races",
        "15 train / 3 validation / 4 test races; target is a retrospective full-session calibrated residual.",
        "held-out test", season_note,
        season_sources, season_runs)]

    # Fixed OpenF1 Hungary validation seed-screen CSV; use only candidate_seed_screen rows and baseline rows.
    circuit_csv = hist / "circuit-physics-v1-validation/exports_222520/aggregate_metrics.csv"
    with circuit_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    circuit_runs = []
    for row in rows:
        typ = row["model_type"]
        if typ not in {"candidate_seed_screen", "comparison_baseline", "preregistration_baseline"}:
            continue
        family = "ST-GNN" if typ == "candidate_seed_screen" else ("Baseline" if "baseline" in typ else "Other")
        circuit_runs.append(run(
            row["model"], family, "Seed 42 · " + typ.replace("_", " "),
            {"mae_sec": float(row["validation_mae_sec"]), "r2_score": float(row["validation_r2_score"]), "sample_count": int(row["validation_sample_count"]) if row["validation_sample_count"] else None},
            status="validation", note="Candidate screen; suite outcome inconclusive; not a test result.", source_paths=[rel(circuit_csv)]))
    groups.append(group("openf1-hungary-circuit-screen", "OpenF1 · Hungary 2023 validation screen", "OpenF1", "Validation · fixed Hungary split", "Seed 42 candidate screen; n=3,068 valid targets per scored model.", "validation", "Five distinct ST-GNN variants and four same-suite baselines. Suite marked inconclusive; no candidate selected. Not comparable to the FastF1 season test.", [rel(circuit_csv), ".local/research-history/2026-10-31/circuit-physics-v1-validation/results.json"], circuit_runs))

    pilot_specs = [
        ("ablation-pilot-2026-10-31-current-k3-seed42", "Monza FastF1 · current k=3 pilot", "current_k3"),
        ("gated-dilated-pilot-2026-10-31-seed42-retry1", "Monza FastF1 · gated dilated retry pilot", "gated_dilated_full_history"),
    ]
    monza_runs = []
    monza_paths = []
    for folder, label, candidate in pilot_specs:
        p = hist / folder / "results.json"
        monza_paths.append(rel(p))
        d = read_json(p)
        for item in d.get("runs", []):
            if item.get("candidate") != candidate:
                continue
            m = item["validation_metrics"]
            monza_runs.append(run(candidate, "ST-GNN", label, m, status="pilot · incomplete", note=f"{item.get('epochs')} epoch completed; incomplete pilot, validation only.", source_paths=[rel(p)]))
    prereg_sources = [hist / s / "preregistration.json" for s in (pilot_specs[0][0], pilot_specs[1][0])]
    monza_paths.extend(rel(p) for p in prereg_sources)
    groups.append(group("fastf1-monza-pilots", "FastF1 · Monza validation pilots", "FastF1", "Validation · Monza prepared data", "Seed 42; same prepared data and train/validation target mask; pilot runs are incomplete.", "pilot", "One completed epoch each. Validation scores only; these are distinct model variants and not final or held-out test results.", monza_paths, monza_runs))

    openf1_pilot_dir = hist / "openf1-2023-multi-session-temporal-gated-seed42"
    openf1_pilot_file = openf1_pilot_dir / "results.json"
    d = read_json(openf1_pilot_file)
    openf1_runs = []
    for item in d.get("runs", []):
        m = item["validation_metrics"]
        openf1_runs.append(run(item["candidate"], "ST-GNN", "OpenF1 3-session gated temporal · seed 42", m, status="pilot · incomplete", note=f"{item.get('epochs')} epochs; incomplete pilot, validation only.", source_paths=[rel(openf1_pilot_file)]))
    groups.append(group("openf1-multisession-pilot", "OpenF1 · 2023 multi-session temporal pilot", "OpenF1", "Validation · Monza / Silverstone / Hungary cohort", "Seed 42; separately prepared incomplete pilot. The preregistration shares the recorded validation-membership hash with the Hungary candidate screen, but no target-mask digest was saved, so target-mask equivalence is unverified.", "pilot", "Incomplete pilot; n=3,068 valid targets. The same sample count does not establish that the effective target mask or sampled targets match. Do not compare directly to the Hungary screen.", [rel(openf1_pilot_file), rel(openf1_pilot_dir / "preregistration.json")], openf1_runs))

    failed_specs = [
        ("gated-dilated-pilot-2026-10-31-seed42", "Monza gated dilated pilot · failed before metrics", "FastF1", "Validation pilot setup; no metric evaluation"),
        ("openf1-2023-multi-session-temporal-gated-mse-seed42", "OpenF1 multi-session gated MSE · interrupted", "OpenF1", "Validation pilot; no metric evaluation"),
    ]
    for folder, label, provider, evaluation in failed_specs:
        p = hist / folder / "results.json"
        d = read_json(p)
        err = d.get("error") or {}
        status = "failed · no score"
        note = err.get("type", d.get("status", "failed"))
        groups.append(group("failed-" + folder, label, provider, evaluation, "No completed evaluation record.", "failed", f"Run status: {d.get('status')}; error: {note}. No score is shown.", [rel(p)], [run("ST-GNN attempt", "ST-GNN", "Gated-dilated pilot attempt · no completed score", status=status, note=note, source_paths=[rel(p)])]))

    # Historical frontend payload is retained as a source but its count and split metadata conflict with the report.
    legacy_metrics = ROOT / "hammertime_100/frontend/data/metrics.json"
    legacy = read_json(legacy_metrics)
    if legacy.get("source") != "pipeline":
        raise ValueError(f"Refusing legacy frontend metrics source={legacy.get('source')!r}; only source='pipeline' is eligible for the visibly unverified legacy group. Existing experiments.json is not replaced.")
    legacy_runs = [
        run("ST-GNN", "ST-GNN", "Legacy Monza frontend payload", {"mae_sec": legacy["mae_sec"], "r2_score": legacy["r2_score"], "sample_count": legacy["sample_count"]}, status="report-only · unverified", note="Payload says source=pipeline but lacks run ID, prediction rows and split metadata; the report gives 983 observations versus payload 984.", source_paths=[rel(legacy_metrics)]),
    ]
    for key, label in (("xgboost", "XGBoost"), ("lstm", "LSTM")):
        m = legacy["baselines"][key]
        legacy_runs.append(run(label, label, "Legacy Monza frontend payload", {"mae_sec": m["mae_sec"], "r2_score": m["r2_score"], "sample_count": None}, status="report-only · unverified", note="Legacy payload; split/sample count unavailable.", source_paths=[rel(legacy_metrics)]))
    groups.append(group("legacy-monza-payload", "Monza · legacy frontend metric payload", "Unresolved legacy pipeline", "Reported evaluation · split unknown", "Legacy static metrics JSON; no reproducible run bundle located.", "report-only", "Historical payload only. ST-GNN sample count 984 conflicts with report text (983); no independent recomputation possible. Do not compare to current season test.", [rel(legacy_metrics), "reports/HammerTime_100_LaTeX/main.tex"], legacy_runs))

    # Report-only final project table: preserve as quoted report values, with no inferred sample count.
    tex = ROOT / "reports/HammerTime_100_LaTeX/main.tex"
    verify_exact_report_rows(
        tex,
        r"\chapter*{9 Saved results and training history}",
        r"The ST-GNN has the lowest saved MAE",
        (
            "HammerTime ST-GNN & 0.046294644 & -0.057058828",
            "Isolated temporal LSTM & 0.046343371 & -0.057039857",
            "Isolated XGBoost & 0.066571377 & -0.012636900",
        ),
        required_marker="It contains 983 target observations",
    )
    report_runs = [
        run("ST-GNN", "ST-GNN", "Saved-result table in final report", {"mae_sec": 0.046294644, "r2_score": -0.057058828}, status="report-only · unverified", note="Report says n=983; metric artifact/predictions and split metadata unavailable.", source_paths=[rel(tex)], unit="not independently verified"),
        run("LSTM", "LSTM", "Saved-result table in final report", {"mae_sec": 0.046343371, "r2_score": -0.057039857}, status="report-only · unverified", note="Report table only; no matched run artifacts or split metadata.", source_paths=[rel(tex)], unit="not reported"),
        run("XGBoost", "XGBoost", "Saved-result table in final report", {"mae_sec": 0.066571377, "r2_score": -0.012636900}, status="report-only · unverified", note="Report table only; no matched run artifacts or split metadata.", source_paths=[rel(tex)], unit="not reported"),
    ]
    groups.append(group("latex-report-saved-results", "HammerTime 100 · final report saved-result table", "Report-only", "Evaluation scope unspecified in saved table", "Values transcribed from report; no associated canonical evaluation artifact located.", "report-only", "Do not compare to the Monza legacy payload or season test. Final report describes 983 observations, while legacy payload says 984.", [rel(tex), rel(legacy_metrics)], report_runs))

    # HammerTime50 report score is historical and has different target/session/window protocol.
    h50 = ROOT / "hammertime_50/REPORT.md"
    verify_exact_report_rows(
        h50,
        "### 4.4 Unseen Test Set Performance & Honest Diagnostic",
        "#### Root Cause Analysis",
        (
            "| **Test MAE** | $< 0.150\\text{ s}$ | **$1.1407\\text{ s}$** | Needs Safety Car Filtering |",
            "| **Test $R^2$** | $0.680 - 0.820$ | **$0.2943$** | Needs Safety Car Filtering |",
        ),
    )
    h50_metric = ROOT / ".local/research-history/2026-10-31/monza-2023-baseline.json"
    groups.append(group("hammertime50-report", "HammerTime 50 · 2021 Abu Dhabi report", "FastF1 (report attribution)", "Reported test · historical session", "Report describes laps 44–58; test window count and target differ from 100-tier evaluations.", "report-only", "MAE 1.1407 s and R² 0.2943 are transcribed from the 50-tier report; no matching run artifact was verified. Not a directly comparable before/after score.", [rel(h50)], [run("ST-GNN · 50-tier report", "ST-GNN", "Historical report score", {"mae_sec": 1.1407, "r2_score": 0.2943, "sample_count": 1371}, status="report-only · unverified", note="Report states 1,371 test windows; score is report-only and unit differs from valid target rows.", source_paths=[rel(h50)], unit="test graph windows")]))

    document = {"schema_version": 1, "title": "HammerTime experiment registry", "generated_by": "export_experiments.py", "policy": "Evidence is grouped by data source, target and split. Metrics must not be ranked across groups.", "groups": groups}
    payload = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{OUT.name}.", suffix=".tmp", dir=OUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, OUT)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    print(f"Wrote {OUT.relative_to(ROOT)}: {sum(len(g['runs']) for g in groups)} records across {len(groups)} groups")


if __name__ == "__main__":
    main()
