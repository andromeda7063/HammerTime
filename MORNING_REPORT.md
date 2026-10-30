# HammerTime morning report — 2026-10-30

The audit ran from approximately **00:14 to 00:53 IST** on `overnight-improvements`. It completed the scoped priority audit and follow-up regression checks; it did not run for eight hours. Nothing was pushed. `hammertime_50/`, the visual generators' aesthetics, existing reports/artifacts, and the user's pre-existing untracked `reports/` were preserved.

**Result:** the original 26 tests pass, and the expanded **106-test offline suite passes** after the continuation. Several original tests missed critical defects, including fabricated training targets. No race performance, passing physics gate, or tuned hyperparameter winner is claimed. FastF1 did not provide usable laps/telemetry for either live session attempted.

## 1. One line per implementation commit

| Commit | Change |
|---|---|
| `ec5409d` | Replace formula/hash/wake-generated sector durations with measured boundary-crossing times. |
| `0cd041b` | Align drivers to shared session time and anchor cumulative distance at observed lap starts. |
| `1968c63` | Mask missing/unsupported tyre metadata instead of inventing compound/age defaults. |
| `e7c4e0b` | Reject FastF1 partial loads without nonempty laps and car telemetry. |
| `13ca982` | Make Streamlit handle missing/malformed artifacts without invented scores or stale images. |
| `5369e21` | Propagate session/preprocessing/target failures instead of substituting combined old race data. |
| `4f1b801` | Exclude inactive graph message sources and reject duplicate identities/grid truncation. |
| `6098144` | Reject empty validation and non-finite gradients before invalid checkpoint/optimizer updates. |
| `e12a1fd` | Move existing graph edges to the model device before concatenating self-loops. |
| `5a5cfa4` | Represent undefined one-target baseline R² as unavailable rather than NaN. |
| `12702d7` | Validate configuration, session identity, directory collisions and per-circuit geometry early. |
| `c170d7d` | Preserve prior artifacts, train each configured run afresh, and render observed data into fresh presentation directories. |
| `460a399` | Add optional seeded hyperparameter comparison using validation MAE only. |
| `f737360` | Test race-held-out scaling/isolation and repeatable checkpoint-to-evaluation integration. |
| `adffb0c` | Keep missing/malformed distance channels within the interpolation validity contract. |
| `0f2b686` | Return the actual configured evaluation-report path. |
| `8aeb55e` | Keep V1/dashboard slices on one absolute timestamp instead of widening tolerance late in a session. |
| `23d71ec` | Exercise existing finite-lap safeguards with missing, nonnumeric and infinite dashboard values. |

The final documentation commit adds this report, open questions, README/test instructions, and append-only assumption/report updates. It has no numerical or executable changes.

## 2. Bugs and disposition

References below point to the audited source; the original source is commit `45ae65d9456ad6e490e07a3c966d4dba882f7dbc`.

| Finding | Source reference | Disposition |
|---|---|---|
| Sector durations came from a sine curve, Python hashes and prescribed wake losses instead of telemetry. | `hammertime_100/src/target.py:26`, `:163` | **Fixed.** Use finite observed distance/time boundary brackets, configured geometry and no extrapolation. Preserve the existing duration plausibility limits, clipping and clean-air hierarchy. Historical numerical artifacts require revalidation. |
| Driver-relative `Time` was compared with session-relative lap timing. | `hammertime_100/src/preprocess.py:61`, `:285` | **Fixed.** Prefer explicit `TimeSec`, then shared `SessionTime`, then `Time`. |
| Relative timestamp tolerance could treat an unsupported near-endpoint observation as exact at large session timestamps. | `hammertime_100/src/preprocess.py:100`, `:111` | **Fixed.** Exactness uses `rtol=0`; interpolation gap limits are unchanged. |
| Multi-lap cumulative distance was used as if it were lap-relative; unmatched rows invented lap 1/tyre metadata. | `hammertime_100/src/preprocess.py:148`, `:165`, `:187` | **Fixed.** Anchor at observed lap starts, and invalidate unsupported timing/tyre metadata. |
| New lap anchoring initially bypassed distance-channel coercion and could raise for missing/nonnumeric values. | `hammertime_100/src/preprocess.py:148` | **Fixed in follow-up.** Two failing tests exposed this audit-introduced edge case; use the same supported numeric observations as interpolation. |
| FastF1 could return normally after logging acquisition failures, and our loader then logged success. | `hammertime_100/src/ingest.py:46` | **Fixed.** Check required loaded/nonempty properties and preserve original exceptions/causes. |
| A broad runner catch substituted combined processed data for a failed requested session, including target/preprocessing errors. | `hammertime_100/run_pipeline.py:366` | **Fixed.** Errors propagate, with no synthetic or stale-data fallback. |
| Existing checkpoints silently skipped the requested experiment, with an example validation-score fallback. | `hammertime_100/run_pipeline.py:401` | **Fixed.** Ordinary runs train afresh; previous checkpoint/history/report contents are copied to `artifact_history/`. |
| Nearby presentation timestamps could be merged because relative tolerance grows with session time. | `hammertime_100/run_pipeline.py:138`; `hammertime_100/dashboard/app.py:285` | **Fixed.** Use `rtol=0` with the existing absolute tolerance. Late-session adjacent observations remain separate. |
| V1 referenced an uninitialized route and otherwise fabricated car positions, edges and lap context. Old images were deleted before rendering. | `hammertime_100/run_pipeline.py:115`, `:134`, `:157` | **Fixed.** Use observed route/positions; missing observations leave V1 unavailable. Fresh run directories and a report manifest preserve historical images. Generator aesthetics are unchanged. |
| Inactive leaders could send graph messages; excess drivers were silently discarded; duplicate driver/timestamps were overwritten. | `hammertime_100/src/graph.py:134`, `:139`, `:187` | **Fixed.** Require valid node identity/capacity and active edge endpoints. |
| Validation with no valid targets was reported as MAE 0, permitting a false best checkpoint. | `hammertime_100/src/train.py:63` | **Fixed.** Raise a descriptive error. The node-weighted MAE formula is unchanged. |
| A finite forward loss with non-finite derivatives could corrupt optimizer state. | `hammertime_100/src/train.py:118` | **Fixed.** Reject invalid gradients before the optimizer update. |
| CPU edges were concatenated with GPU self-loops before moving devices. | `hammertime_100/src/model.py:99` | **Fixed.** Transfer before concatenation. CPU and hardware-independent device tests pass; actual CUDA execution remains unverified. |
| Baselines returned NaN R² for one held-out target, unlike main evaluation, breaking strict JSON serialization. | `hammertime_100/src/baselines.py:122`, `:176` | **Fixed.** Use `None` when R² cannot be measured; formulas for measurable scores are unchanged. |
| Missing/invalid config values could fail late or silently use defaults; session aliases could duplicate an exact configured source or collide on disk. Different circuits inherited Monza geometry. | `hammertime_100/src/config_validation.py:6`; `hammertime_100/run_pipeline.py:33` | **Fixed for declared identities.** Validate types, ranges, dimensions, required paths, duplicate configured sources and safe directory names; require explicit geometry for other circuits. Circuit-name aliases with different spellings remain a reviewer consideration. |
| Dashboard example scores, malformed report structures, empty numeric lap lists, missing optional columns and bad images could mislead or crash. | `hammertime_100/dashboard/app.py:57`, `:201`, `:211`, `:256`, `:359` | **Fixed.** Scores/images remain unavailable unless supported; valid zero MAE and negative R² remain visible. Physics status is matched by test name. Ten AppTest methods pass, including nearby-timestamp isolation. |
| Runner summary ignored a custom report path. | `hammertime_100/run_pipeline.py:463` | **Fixed.** Summary and log identify the actual report. |

**Audited without changing the implementation:** dataset windows, purged temporal split, race-held-out partitioning, isolated baseline histories, and train-only scaler fitting. Five added dataset contracts verify race disjointness, held-out extreme-value isolation, continuity, alignment tolerance and inactive target masks. `src/dataset.py` and `src/evaluate.py` have no changes.

**Deferred:** whole-session clean-air reference calibration; DRS threshold discrepancy; physical versus classification car-ahead semantics; integrated-distance accuracy; legacy frontend sample claims; and custom evaluation sampler metadata. Details and source references are in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). These require scientific/product decisions, so the audit did not change their definitions to improve scores.

## 3. Test results and exact commands

All Python work after the user's environment clarification used **uv**. Tests use small in-memory fixtures or temporary local artifact files and mocked loaders. They do not download FastF1 data. The separate live acquisition attempt below was not part of a test.

### Environment and original suite

- Initial `.venv/bin/python -m unittest discover -s hammertime_100/tests`: **4 import errors**, missing NumPy/PyTorch. This was an environment failure, not four established product regressions.
- `uv sync --frozen --offline`: failed because locked PyTorch was absent from the cache.
- `uv sync --frozen`: **107 declared packages installed**, with the large downloads taking about 25 minutes. No dependency files changed.
- Runtime: Python **3.12.14**, NumPy **2.5.3**, pandas **2.3.3**, PyTorch **2.14.1+cu130**, PyG **2.8.0.post1**, scikit-learn **1.9.1**, FastF1 **3.8.3**. `torch.cuda.is_available()` was **False**.

The pristine original source was extracted into `/tmp/hammertime-overnight-pristine` and tested with the same locked environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s /tmp/hammertime-overnight-pristine/hammertime_100/tests
```

Output: `Ran 26 tests in 0.162s` / **OK**. The original suite passing did not establish pipeline correctness.

Final integrated command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests
```

Output: `Ran 103 tests in 4.175s` / **OK**. This adds **77 tests**, including a tiny checkpoint/evaluation integration test that confirms identical CPU weights, histories and predictions on repeated seeded fixture runs. Its numerical outputs are fixture checks, not race results.

A follow-up timestamp test failed in V1 and Streamlit because relative tolerance mixed observations at 6000.0 and 6000.05 seconds. Setting `rtol=0` fixed both without changing absolute tolerances. The V1 artifact tests (2) and dashboard AppTest methods (10) passed. Expanded missing/infinite lap subcases also passed the existing dashboard guard.

The first expanded run found a `NameError` in a newly authored tuning test (a config-restoration assertion was in the wrong method). That test-authoring error was corrected, the four tuning tests passed, and the final integrated suite above passed.

### Focused regression evidence

For the rows below, these are the exact command prefixes used with each listed suffix:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s /tmp/hammertime-overnight-baseline/hammertime_100/tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests
```

The first prefix targets archived original source with regression test copies; the second targets the working tree. The archive is separate from the pristine 26-test baseline.

| Suffix | Before | After |
|---|---|---|
| `-p test_training_contracts.py` | 7 tests; 4 failures | 7 passed |
| `-p test_model_device_contracts.py` | 3 tests; 1 error | 3 passed |
| `-p test_baseline_metrics.py` | 2 tests; 2 failures | 2 passed |
| `-p 'test_pipeline_*.py'` | 3 tests; 2 failures, 1 error | All 3 pass in final suite |
| `-p test_graph_contracts.py` | 3 tests; 3 failures | All 3 pass in final suite |
| `-p test_session_config.py` | 5 methods; 7 failing assertions/subcases | All 5 pass in final suite |
| `-p test_checkpoint_provenance.py` | 1 test; 1 failure | Passes; later extended for custom report paths |

Before full environment setup, data tests also ran in an isolated UV environment using existing declared dependencies:

```bash
uv run --quiet --no-project --python 3.12 --with fastf1==3.8.3 --with pyyaml==6.0.3 python -m unittest discover -s hammertime_100/tests -p test_target_measurements.py
uv run --quiet --no-project --python 3.12 --with fastf1==3.8.3 --with pyyaml==6.0.3 python -m unittest discover -s hammertime_100/tests -p 'test_preprocess_*.py'
uv run --quiet --no-project --python 3.12 --with fastf1==3.8.3 --with pyyaml==6.0.3 python -m unittest discover -s hammertime_100/tests -p test_ingest_contracts.py
```

At that stage: **8 target**, **12 preprocessing**, and **6 ingestion** tests passed. All were later included in the locked full-suite run. The target test compatible with the original API failed on original code because it fabricated three sectors, including a partial one, rather than measuring two supported traversals. Seven additional geometry tests required the newly added arguments; their original-source signature errors are not counted as seven separate established bugs. Original shared-clock/anchoring tests had six failures; metadata regressions against the intermediate implementation had nine failing subcases and two errors.

Dashboard AppTest before/after commands, with versions pinned from the existing lockfile:

```bash
uv run --quiet --no-project --python 3.12 --with streamlit==1.65.0 --with pandas==2.3.3 --with numpy==2.5.3 --with plotly==7.1.0 --with pyyaml==6.0.3 python -m unittest discover -s /tmp -p test_dashboard_contracts_before.py
uv run --quiet --no-project --python 3.12 --with streamlit==1.65.0 --with pandas==2.3.3 --with numpy==2.5.3 --with plotly==7.1.0 --with pyyaml==6.0.3 python -m unittest discover -s hammertime_100/tests -p test_dashboard_contracts.py
```

Before: **9 methods, 17 failing assertions/subcases** against the archived app. After: **9 passed**; also passed in the final locked suite.

Follow-up checks:

```bash
uv run --frozen python -m unittest discover -s hammertime_100/tests -p test_preprocess_channels.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests -p test_checkpoint_provenance.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests -p test_tuning_contracts.py
uv run --frozen python hammertime_100/run_pipeline.py --help
git diff --check
```

Distance-channel regression: **2 errors before**, **2 passed after**; broader preprocessing group **14 passed**. Custom report-path assertion: **1 failed before**, **1 passed after**. Tuning safety tests: **4 passed**, covering mean-score selection, common shuffle histories, forbidden split/target/gate overrides, early validation and RNG restoration. CLI help lists `--config` and `--tune`; whitespace check passed.

Non-failing compatibility warnings remain from PyG's deprecated `torch.jit.script` use, Streamlit's `use_container_width` API, and pandas/NumPy timedelta fixture conversion. No warning was suppressed to obtain a passing test.

## 4. Verification limits and hyperparameter result

**Authentic FastF1 acquisition was attempted once, separately from tests.** The command was:

```bash
uv run --quiet --no-project --python 3.12 --with fastf1==3.8.3 --with pyyaml==6.0.3 python - <<'PY'
import sys
from pathlib import Path
root = Path('/home/andromeda/Documents/code/projects/HammerTime')
sys.path.insert(0, str(root / 'hammertime_100' / 'src'))
from ingest import load_race
session = load_race(2023, 'Monza', 'R', str(root / 'hammertime_100' / 'data' / 'cache' / '2023-Monza-R'))
print('AUTHENTIC_SESSION', session.event['EventName'], 'DRIVERS', len(session.drivers), 'LAPS', len(session.laps))
first = session.laps.pick_drivers(session.drivers[0]).get_telemetry()
print('TELEMETRY_COLUMNS', list(first.columns))
print('TELEMETRY_ROWS', len(first), 'SESSION_TIME_RANGE', first['SessionTime'].min(), first['SessionTime'].max())
PY
```

FastF1's primary endpoint and mirror returned no usable session data (`SessionNotAvailableError`). It obtained event/driver metadata, but accessing `session.laps` raised `DataNotLoadedError`. The log is `/tmp/hammertime-live-load.log`. This exposed the partial-load bug fixed in `e7c4e0b`. Its corrected failure behavior is covered by offline mocks. The cause of provider unavailability was not established; this is not evidence that all network access failed.

Consequently **full real-race preprocessing/training/evaluation, live physics outcomes, production image generation and actual hyperparameter selection were skipped**. No synthetic race data, demo exporter, invented benchmark scores or old processed race substitution was used. Checked-in checkpoints, histories, frontend JSON and images remain historical/unverified.

The optional tuning workflow is ready, but is **not a measured winner**: it compares learning rates **0.001, 0.0003, 0.003** over seeds **42, 43, 44**, on one fixed split/scaler, selecting mean best validation MAE. Seed 42 supplies the chosen candidate's checkpoint instead of selecting the luckiest seed. All trials keep separate checkpoints/history and a `study.json`; test evaluation follows selection. The workflow cannot search split settings, targets or thresholds. Existing default optimizer/model values, metrics, split implementation, target clipping and gate/physics thresholds are unchanged.

Actual GPU execution is unverified because CUDA hardware was unavailable. Dashboard behavior was checked through Streamlit AppTest; no claim is made about a manual browser replay of real artifacts.

## 5. Human review and suggested next steps

1. Resolve the scientific questions in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), especially the YAML DRS minimum **0.10 s** versus prose/function default **0.40 s**, and whether session-wide clean-air calibration is acceptable for the intended forecasting claim. Neither criterion was altered.
2. Restore usable FastF1 acquisition or supply an authentic cache with provenance. Inspect measured sector coverage, last-sector completeness, compound support and integrated-distance accuracy before fitting models.
3. Review the declared study before running `uv run --frozen python hammertime_100/run_pipeline.py --config hammertime_100/configs/default.yaml --tune`. Do not choose candidates from test scores or physics gate results. Record candidate decisions before inspecting the final held-out report.
4. Review the selected checkpoint, full study, report/scaler metadata and fresh presentation manifest together. A failed/unmeasurable gate must remain reported as such. Treat the present label calibration as retrospective until an independent reference-fitting protocol is agreed.
5. Use the artifact-backed Streamlit dashboard for audited output. The legacy standalone frontend still contains sample/random content and static passing assertions; replacing its demonstration logic is deferred rather than presented as verified telemetry.

One additional reporting limitation remains for reviewer awareness: the runner's existing `gates.all_passed` combines module/physics/presentation gates; metric acceptance flags live separately under `metrics`. It should not be interpreted as a substitute for inspecting MAE/R² acceptance flags. No gate thresholds or aggregation semantics were changed during this audit.

The original documentation/assumption history is preserved. Session-time and distance-origin decisions were checked against primary FastF1 sources: [core telemetry implementation](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/core.py) and [time-origin documentation](https://github.com/theOehrly/Fast-F1/blob/main/docs/data_reference/time_explanation.rst).

## 6. Dependencies and repository state

**No new dependencies added.** `pyproject.toml`, `uv.lock`, and both implementations' requirements files are unchanged. Isolated verification environments only used dependencies already declared by the project. The locked environment was installed through UV.

All executable changes and tests are committed separately on `overnight-improvements`. No files were deleted, no push was performed, `hammertime_50/` was untouched, and existing `reports/` remains untracked as found. This report and OPEN_QUESTIONS.md retain all unverified/deferred work explicitly.

## 7. Continuation through the 08:00 IST cutoff

The follow-up continuation reached the requested 08:00 IST cutoff while fixes were in progress; verification and report closeout finished shortly afterward. It did not establish that development ran continuously for every intervening hour. The four additional verified commits are:

| Commit | Change |
|---|---|
| `52b9f80` | Add SHA-256 hashes of source/effective configuration, raw exports, processed tables and selected checkpoint, plus Git state and software versions, to the evaluation report. |
| `245dfa7` | Use left-padded causal temporal convolution so the prediction consumes the full latest kernel window. |
| `d059daa` | Reject reordered evaluation loaders before matching predictions to driver/lap/gap metadata. |
| `97bb799` | Preserve the first finite measurement per channel at duplicate telemetry timestamps. |

**Bugs and limits.** In `hammertime_100/src/model.py:76,141`, symmetric `Conv1d` padding made the last kernel-3 output depend on only two real history frames and a right-side zero. The causal padding fix is a model-behavior correction, not a measured performance improvement; old checkpoints should be retrained. In `src/evaluate.py:27`, a shuffled loader could attach predictions to the wrong sample metadata and thus corrupt physics interpretation; it now raises clearly. In `src/preprocess.py:68,96,153`, global timestamp deduplication could discard a real channel measurement when another row at that time was missing; channel-specific validation now precedes deduplication. In `run_pipeline.py:90,480`, completed reports lacked raw-data and code provenance; hashes and environment state are now recorded. The hashes establish file identity, not that FastF1's upstream data are complete or scientifically valid.

**Before and after tests.** The new provenance assertion in `test_checkpoint_provenance.py` first failed with `KeyError: 'metadata'`; after the fix, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests -p test_checkpoint_provenance.py` ran **1 test, OK**. The model regression `uv run python -m unittest tests.test_stgnn_model.STGNNModelTests.test_temporal_kernel_uses_full_recent_history -v` first failed (`0.0` versus expected `2.0`); afterward `uv run python -m unittest tests.test_stgnn_model -v` ran **9 tests, OK**. The evaluation-order regression `uv run --project . python -m unittest hammertime_100.tests.test_evaluation_order -v` failed before the guard (expected `ValueError` was not raised); its focused after-run with physics, baselines, graph and dataset contracts ran **24 tests, OK**. The duplicate-time regression `uv run --no-sync --project . python -m unittest hammertime_100.tests.test_preprocess_channels.DistanceChannelTests.test_duplicate_timestamp_keeps_later_valid_measurement` first failed (Distance `50` rather than observed `60` at the duplicate timestamp) and then ran **1 test, OK**. The final integrated command `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests` ran **106 tests in 4.029s, OK**. All regression fixtures were in memory or temporary local files. No test fetched data. `git diff --check` passed. Existing warnings remain visible.

**Authentic acquisition.** Separately from tests, the agent ran `uv run --no-sync --project . python -c 'import sys; sys.path.insert(0, "hammertime_100/src"); from ingest import load_race; s=load_race(2023, "Monza", "R", "hammertime_100/data/cache"); print("loaded", s.event["EventName"], "laps", len(s.laps), "drivers_with_car_data", sum(not d.empty for d in s.car_data.values()))'`. FastF1 3.8.3 reported missing session info/laps/car telemetry from its live-timing mirror; `load_race` raised `RuntimeError: FastF1 session 2023 Monza [R] did not load required laps`, with `DataNotLoadedError` as the cause. A 2024 Bahrain attempt also failed to load required laps. The precise upstream cause remains unverified. No old processed files were substituted, no synthetic race data was created, and the validation-only tuning study remained unrun.

**Research context.** The [DCRNN paper](https://arxiv.org/abs/1707.01926) and [STGCN paper](https://www.ijcai.org/Proceedings/2018/0505) study graph and temporal modeling for road-traffic forecasting; they motivate checking the time direction and window semantics here but provide no F1 performance evidence. [PyTorch Conv1d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) specifies symmetric integer padding and the output-length behavior behind the observed defect. No new architecture, dependency, metric, split, target, gate or physics threshold was introduced from these papers. The next scientific step remains a provenance-backed authentic race run, sector-coverage audit, then the preregistered validation-only comparison and a single held-out evaluation. The clean-air calibration and DRS-definition questions remain unresolved in `OPEN_QUESTIONS.md`.

## 8. FastF1 documentation and HTTP verification

At the user's request, the installed FastF1 **3.8.3** API documentation/docstrings and its official version-tagged source were checked. `Session.load(*, laps=True, telemetry=True, weather=True, messages=True, livedata=None)` confirms the existing loader call is valid. `DataNotLoadedError` means the accessed property was not populated; it does not by itself identify a network cause. FastF1's [exception definitions](https://github.com/theOehrly/Fast-F1/blob/v3.8.3/fastf1/exceptions.py) and [soft exception handler](https://github.com/theOehrly/Fast-F1/blob/v3.8.3/fastf1/logger.py) explain why a loader can log warnings and return before a later property access raises. The hosted documentation returned HTTP 403 through the browsing tool, so the corresponding installed docstrings and official source were used.

A separate read-only diagnostic used `uv run --frozen python` with `requests.get`, FastF1's own request headers, a 15-second timeout, and the known session-information path `/static/2023/2023-09-03_Italian_Grand_Prix/2023-09-03_Race/SessionInfo.jsonStream`. The primary `https://livetiming.formula1.com` returned **HTTP 401, zero response bytes**; `https://livetiming-mirror.fastf1.dev` returned **HTTP 404, text/html, 27150 bytes**. This was a live diagnostic, not a test, and did not fetch race telemetry. The observed access/resource failures explain why this environment cannot populate the session; the reason for the 401 remains unverified. An [upstream report with the same FastF1 version and symptoms](https://github.com/theOehrly/Fast-F1/issues/916) was resolved by changing to a home network, but that does not prove this environment has the same cause. No dependency or loader-code change was justified by this verification.
