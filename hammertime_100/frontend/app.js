'use strict';

const DATA_URL = 'data/season_model.json?v=singapore-1';
const MARINA_BAY_REPLAY_URL = 'data/marina_bay_replay.json';
const MARINA_BAY_INFERENCE_URL = 'data/marina_bay_inference.json?v=singapore-1';
// Official 2023 team palette from the source-backed Singapore driver-asset manifest.
const MARINA_BAY_TEAM_COLORS_2023 = Object.freeze({
  'Red Bull Racing': '#3671c6',
  Williams: '#37bedd',
  McLaren: '#f58020',
  Alpine: '#2293d1',
  'Aston Martin': '#358c75',
  Ferrari: '#f91536',
  'Haas F1 Team': '#B6BABD',
  AlphaTauri: '#5e8faa',
  'Alfa Romeo': '#c92d4b',
  Mercedes: '#6cd3bf',
});
const CHART_HIT_RADIUS_PX = 11;
const state = { run: null, raceId: 'all', driver: 'all', sampleOrder: 'source', breakdown: 'driver', chartHitTest: null, selectedPoint: null };
const replayState = { data: null, frameIndex: 0, elapsedSeconds: 0, focusDriver: 'all', speed: 1, playing: false, lastAnimationTime: null, animationId: null, resizeObserver: null };
let marinaBayDriverAssetMap = null;
const $ = (id) => document.getElementById(id);
const number = (value, digits = 5) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const integer = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US') : '—';
const raceName = (value) => String(value || '').replace(/^2023-/, '').replace(/-R$/, '').replaceAll('-', ' ');
const cell = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
};

function resultRaceIds(run = state.run) {
  return run?.race_metrics?.map((row) => row.race_id) || [];
}

function populationDescription(run = state.run) {
  return run?.inference_scope?.held_out_test === false
    ? 'in-sample inference population'
    : 'held-out evaluation population';
}

function validRun(run) {
  const numeric = (value) => (typeof value === 'number'
    || (typeof value === 'string' && value.trim() !== '')) && Number.isFinite(Number(value));
  const count = (value) => numeric(value) && Number.isInteger(Number(value)) && Number(value) >= 0;
  const sha256 = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/i.test(value);
  const gateFieldsValid = (row) => (typeof row.target_mae_pass === 'boolean'
    && typeof row.target_r2_pass === 'boolean')
    || (run?.inference_scope?.held_out_test === false
      && row.target_mae_pass === null && row.target_r2_pass === null);
  const metricRow = (row) => row && numeric(row.mae_sec) && numeric(row.rmse_sec)
    && numeric(row.r2_score) && count(row.sample_count) && Number(row.sample_count) > 0
    && gateFieldsValid(row);
  const metrics = run?.metrics;
  const races = resultRaceIds(run);
  const physicsTests = metrics?.physics_tests;
  const errorDistribution = run?.error_distribution;
  const racesValid = Array.isArray(races) && races.length > 0
    && races.every((race) => typeof race === 'string')
    && new Set(races).size === races.length
    && Array.isArray(run?.race_metrics) && run.race_metrics.length === races.length
    && run.race_metrics.every((row) => metricRow(row) && typeof row.race_id === 'string')
    && races.every((race) => run.race_metrics.some((row) => row.race_id === race));
  const groupsCoverTargets = (rows) => rows.reduce((sum, row) => sum + Number(row.sample_count), 0)
    === Number(metrics?.sample_count);
  const distributionValid = errorDistribution && count(errorDistribution.population_count)
    && Number(errorDistribution.population_count) === Number(metrics?.sample_count)
    && typeof errorDistribution.population === 'string' && typeof errorDistribution.method === 'string'
    && typeof errorDistribution.unit === 'string'
    && errorDistribution.observed_range_sec
    && ['stgnn', 'zero_predictor'].every((series) => errorDistribution.observed_range_sec[series]
      && numeric(errorDistribution.observed_range_sec[series].min_sec)
      && numeric(errorDistribution.observed_range_sec[series].max_sec)
      && Number(errorDistribution.observed_range_sec[series].min_sec) <= Number(errorDistribution.observed_range_sec[series].max_sec))
    && Array.isArray(errorDistribution.bins) && errorDistribution.bins.length > 0
    && errorDistribution.bins.every((bin) => bin && typeof bin.label === 'string'
      && count(bin.stgnn_count) && count(bin.zero_predictor_count))
    && errorDistribution.bins.reduce((sum, bin) => sum + Number(bin.stgnn_count), 0)
      === Number(errorDistribution.population_count)
    && errorDistribution.bins.reduce((sum, bin) => sum + Number(bin.zero_predictor_count), 0)
      === Number(errorDistribution.population_count);
  return Boolean(run && typeof run.run_id === 'string' && run.run_id
    && typeof run.source === 'string' && typeof run.model === 'string' && typeof run.target === 'string'
    && metrics && numeric(metrics.mae_sec) && numeric(metrics.rmse_sec)
    && numeric(metrics.r2_score) && count(metrics.sample_count) && Number(metrics.sample_count) > 0
    && gateFieldsValid(metrics)
    && metrics.zero_baseline && numeric(metrics.zero_baseline.mae_sec)
    && numeric(metrics.zero_baseline.r2_score) && count(metrics.zero_baseline.sample_count)
    && Number(metrics.zero_baseline.sample_count) === Number(metrics.sample_count)
    && gateFieldsValid(metrics.zero_baseline)
    && Array.isArray(physicsTests) && physicsTests.length > 0 && physicsTests.every((test) =>
      test && typeof test.test === 'string' && typeof test.expected === 'string'
      && (typeof test.passed === 'boolean' || test.passed === null))
    && racesValid
    && Array.isArray(run.driver_metrics) && run.driver_metrics.length > 0 && run.driver_metrics.every((row) =>
      metricRow(row) && (typeof row.driver === 'string' || numeric(row.driver)))
    && Array.isArray(run.mini_sector_metrics) && run.mini_sector_metrics.length > 0
    && run.mini_sector_metrics.every((row) => metricRow(row) && count(row.mini_sector)
      && Number(row.mini_sector) < 20)
    && groupsCoverTargets(run.race_metrics) && groupsCoverTargets(run.driver_metrics)
    && groupsCoverTargets(run.mini_sector_metrics) && distributionValid
    && count(run.split?.train_window_count) && count(run.split?.validation_window_count)
    && count(run.split?.test_window_count) && Array.isArray(run.split?.test_races)
    && (run.inference_scope?.held_out_test !== false
      || (typeof run.inference_scope.race_id === 'string'
        && run.inference_scope.split_role === 'training'
        && count(run.inference_scope.inference_window_count)
        && !run.split.test_races.includes(run.inference_scope.race_id)))
    && run.prediction_sample && count(run.prediction_sample.source_rows)
    && Number(run.prediction_sample.source_rows) === Number(run.metrics.sample_count)
    && Array.isArray(run.prediction_sample.points) && run.prediction_sample.points.length > 0
    && run.prediction_sample.points.length <= Number(run.prediction_sample.source_rows)
    && run.prediction_sample.points.every((point) => point && typeof point.race_id === 'string'
      && races.includes(point.race_id)
      && (typeof point.driver === 'string' || numeric(point.driver)) && count(point.lap)
      && numeric(point.time_sec) && count(point.mini_sector) && Number(point.mini_sector) < 20
      && numeric(point.actual) && numeric(point.predicted))
    && run.checkpoint && sha256(run.checkpoint.sha256)
    && numeric(run.checkpoint.selected_epoch) && numeric(run.checkpoint.best_validation_mae_sec)
    && run.provenance && typeof run.provenance.fingerprint === 'string'
    && sha256(run.provenance.config_sha256) && sha256(run.provenance.prepared_artifact_sha256)
    && run.provenance.implementation_sha256 && typeof run.provenance.implementation_sha256 === 'object'
    && Object.keys(run.provenance.implementation_sha256).length > 0
    && Object.values(run.provenance.implementation_sha256).every(sha256)
    && run.provenance.source_artifact_sha256 && typeof run.provenance.source_artifact_sha256 === 'object'
    && Object.keys(run.provenance.source_artifact_sha256).length > 0
    && Object.values(run.provenance.source_artifact_sha256).every(sha256)
    && typeof run.count_note === 'string' && typeof run.prediction_sample.method === 'string');
}

function gateLabel(passed, assessed = true) {
  if (!assessed) return ['NOT ASSESSED', 'unknown'];
  return passed === true ? ['PASS', 'pass'] : passed === false ? ['FAIL', 'fail'] : ['NOT REPORTED', 'unknown'];
}

function renderMetricSummary() {
  const run = state.run;
  const m = run.metrics;
  $('metricMae').textContent = `${number(m.mae_sec)} s`;
  $('metricRmse').textContent = `${number(m.rmse_sec)} s`;
  $('metricR2').textContent = number(m.r2_score);
  $('metricN').textContent = integer(m.sample_count);
  const gatesAssessed = state.run.inference_scope?.held_out_test !== false;
  for (const [id, passed] of [['maeGate', m.target_mae_pass], ['r2Gate', m.target_r2_pass]]) {
    const [label, klass] = gateLabel(passed, gatesAssessed);
    $(id).textContent = label;
    $(id).className = `gate-pill ${klass}`;
  }
  $('metricGrid').setAttribute('aria-busy', 'false');
  $('loadStatus').textContent = 'Run loaded · archived metrics';
  $('runIdShort').textContent = run.run_id.slice(0, 12);
  $('runIdFull').textContent = run.run_id;
  $('countNote').textContent = run.count_note;
  $('splitStrip').replaceChildren();
  for (const [label, value] of [
    ['Train windows', run.split.train_window_count],
    ['Validation windows', run.split.validation_window_count],
    ['Held-out test windows', run.split.test_window_count],
  ]) {
    const item = cell('div', undefined, 'split-item');
    item.append(cell('span', label), cell('strong', integer(value)));
    $('splitStrip').append(item);
  }
}

function fillRaceSelector() {
  const select = $('raceFilter');
  select.replaceChildren();
  const races = resultRaceIds();
  if (races.length > 1) {
    select.append(new Option('All races', 'all'));
  }
  for (const raceId of races) {
    select.append(new Option(`${raceName(raceId)} Grand Prix`, raceId));
  }
  if (races.length === 1) {
    select.value = races[0];
    select.disabled = true;
  } else {
    select.disabled = false;
  }
  const drivers = state.run.driver_metrics.map((row) => String(row.driver)).sort((a, b) => Number(a) - Number(b));
  const driverSelect = $('driverFilter');
  driverSelect.replaceChildren(new Option('All drivers', 'all'));
  for (const driver of drivers) driverSelect.append(new Option(`#${driver}`, driver));
  driverSelect.disabled = false;
  $('sampleOrderFilter').disabled = false;
  $('downloadViewProvenance').disabled = false;
}

function renderRaceMetrics() {
  const all = state.raceId === 'all';
  const race = all ? (state.run.race_metrics[0] || state.run.metrics) : (state.run.race_metrics.find((row) => row.race_id === state.raceId) || state.run.metrics);
  const m = race;
  const displayName = raceName(race?.race_id || (all ? state.run.split.test_races[0] : state.raceId));
  $('selectedRaceTitle').textContent = `${displayName} Grand Prix`;
  $('raceMae').textContent = `${number(m?.mae_sec)} s`;
  $('raceRmse').textContent = `${number(m?.rmse_sec)} s`;
  $('raceR2').textContent = number(m?.r2_score);
  $('raceN').textContent = integer(m?.sample_count);
}

function renderRaceTable() {
  const tbody = $('raceTableBody');
  tbody.replaceChildren();
  for (const race of state.run.race_metrics) {
    const row = document.createElement('tr');
    const raceCell = document.createElement('th');
    raceCell.scope = 'row';
    const button = cell('button', raceName(race.race_id), 'table-select');
    button.type = 'button';
    button.setAttribute('aria-label', `Show ${raceName(race.race_id)} prediction sample`);
    button.addEventListener('click', () => {
      $('raceFilter').value = race.race_id;
      updateExplorer();
      $('explorer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    raceCell.append(button);
    row.append(raceCell, cell('td', integer(race.sample_count)), cell('td', number(race.mae_sec)),
      cell('td', number(race.rmse_sec)), cell('td', number(race.r2_score)));
    const resultCell = document.createElement('td');
    const [label, klass] = gateLabel(race.target_mae_pass);
    resultCell.append(cell('span', label, `gate-pill ${klass}`));
    row.append(resultCell);
    tbody.append(row);
  }
}

function renderBaseline() {
  const baseline = state.run.metrics.zero_baseline;
  if (!baseline) {
    $('baselineCompare').replaceChildren(cell('p', 'No matched baseline was recorded for this run.', 'muted-note'));
    return;
  }
  const rows = [
    ['HammerTime ST-GNN', state.run.metrics.mae_sec, state.run.metrics.r2_score, 'model-bar'],
    ['Zero prediction', baseline.mae_sec, baseline.r2_score, 'baseline-bar'],
  ];
  const root = $('baselineCompare');
  root.replaceChildren();
  const max = Math.max(...rows.map((row) => row[1]));
  for (const [name, mae, r2, color] of rows) {
    const wrapper = cell('div', undefined, 'baseline-row');
    const heading = cell('div', undefined, 'baseline-heading');
    heading.append(cell('span', name), cell('strong', `${number(mae)} s MAE · R² ${number(r2)}`));
    const track = cell('div', undefined, 'bar-track');
    track.setAttribute('role', 'img');
    track.setAttribute('aria-label', `${name} mean absolute error ${number(mae)} seconds`);
    const bar = cell('span', undefined, `bar-fill ${color}`);
    bar.style.width = `${Math.max(2, (mae / max) * 100)}%`;
    track.append(bar);
    wrapper.append(heading, track);
    root.append(wrapper);
  }
}

function renderErrorDistribution() {
  const distribution = state.run.error_distribution;
  const bins = distribution.bins;
  const sharedMaximum = Math.max(...bins.flatMap((bin) =>
    [Number(bin.stgnn_count), Number(bin.zero_predictor_count)]), 1);
  $('errorPopulation').textContent = `${integer(distribution.population_count)} targets · full ${populationDescription()}`;
  const stgnnRange = distribution.observed_range_sec.stgnn;
  const zeroRange = distribution.observed_range_sec.zero_predictor;
  $('errorDistributionNote').textContent = `${distribution.population} · ${distribution.unit}. ${distribution.method} Maximum absolute error: ST-GNN ${number(stgnnRange.max_sec)} s; zero predictor ${number(zeroRange.max_sec)} s.`;
  const body = $('errorDistributionBody');
  body.replaceChildren();
  for (const bin of bins) {
    const row = document.createElement('tr');
    const label = document.createElement('th');
    label.scope = 'row'; label.textContent = bin.label;
    row.append(label);
    for (const [key, maximum, color] of [
      ['stgnn_count', sharedMaximum, ''], ['zero_predictor_count', sharedMaximum, ' zero'],
    ]) {
      const count = Number(bin[key]);
      const wrapper = cell('div', undefined, 'error-count');
      wrapper.append(cell('strong', integer(count)));
      const track = cell('span', undefined, 'error-count-track');
      track.setAttribute('aria-hidden', 'true');
      const fill = cell('span', undefined, `error-count-fill${color}`);
      fill.style.width = `${(count / maximum) * 100}%`;
      track.append(fill); wrapper.append(track);
      const dataCell = cell('td');
      dataCell.append(wrapper);
      row.append(dataCell);
    }
    body.append(row);
  }
}

function renderPhysics() {
  const root = $('physicsList');
  root.replaceChildren();
  for (const test of state.run.metrics.physics_tests || []) {
    const unmeasurable = test.passed === false && /no complete/i.test(test.notes || '');
    const notAssessed = test.passed === null && /not assessed/i.test(test.notes || '');
    const status = test.passed === true ? 'PASS' : unmeasurable ? 'UNMEASURABLE' : test.passed === false ? 'FAIL' : notAssessed ? 'NOT ASSESSED' : 'NOT REPORTED';
    const statusClass = test.passed === true ? 'pass' : unmeasurable ? 'unmeasurable' : test.passed === false ? 'fail' : 'unknown';
    const measured = test.observed_mae != null ? `${number(test.observed_mae)} s MAE`
      : test.observed_lap_delay != null ? `${number(test.observed_lap_delay)} s/lap` : '—';
    const evidence = test.sample_count != null ? `${integer(test.sample_count)} observations`
      : test.complete_laps != null ? `${integer(test.complete_laps)} complete laps` : 'No count recorded';
    const item = cell('article', undefined, 'physics-item');
    const header = cell('div', undefined, 'physics-header');
    header.append(cell('h4', test.test), cell('span', status, `gate-pill ${statusClass}`));
    const details = cell('p', undefined, 'physics-result');
    details.append(cell('span', `Observed ${measured}`), cell('span', `Expected ${test.expected}`), cell('span', evidence));
    item.append(header, details, cell('p', test.notes || '', 'muted-note'));
    root.append(item);
  }
}

function matchingPoints() {
  return state.run.prediction_sample.points.filter((point) =>
    (state.raceId === 'all' || point.race_id === state.raceId)
    && (state.driver === 'all' || String(point.driver) === state.driver));
}

function orderedSamplePoints(points = matchingPoints()) {
  if (state.sampleOrder === 'source') return points;
  const descending = state.sampleOrder === 'largest-error';
  return points.map((point, sourceIndex) => {
    const absoluteError = point.actual == null || point.predicted == null
      ? Number.NaN : Math.abs(Number(point.predicted) - Number(point.actual));
    return { point, sourceIndex, absoluteError: Number.isFinite(absoluteError) ? absoluteError : null };
  }).sort((a, b) => {
    if (a.absoluteError === null || b.absoluteError === null) {
      if (a.absoluteError === null && b.absoluteError !== null) return 1;
      if (a.absoluteError !== null && b.absoluteError === null) return -1;
      return a.sourceIndex - b.sourceIndex;
    }
    const difference = descending ? b.absoluteError - a.absoluteError : a.absoluteError - b.absoluteError;
    return difference || a.sourceIndex - b.sourceIndex;
  }).map((entry) => entry.point);
}

function sampleOrderLabel(order = state.sampleOrder) {
  if (order === 'largest-error') return 'largest absolute residual error first';
  if (order === 'smallest-error') return 'smallest absolute residual error first';
  return 'source order';
}

function drawScatter() {
  const canvas = $('predictionChart');
  const context = canvas.getContext('2d');
  if (!context) return;
  hidePointTooltip();
  state.chartHitTest = null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(390 * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = 390;
  const margin = { top: 25, right: 22, bottom: 58, left: 65 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const points = matchingPoints();
  if (state.selectedPoint && !points.includes(state.selectedPoint)) state.selectedPoint = null;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#121a20';
  context.fillRect(0, 0, width, height);
  if (!points.length) {
    context.fillStyle = '#b3bdc2'; context.font = '14px sans-serif';
    context.fillText('No sampled points match these filters.', margin.left, margin.top + 25);
    $('sampleLabel').textContent = 'Chart sample: 0 points';
    renderPredictionTable(points);
    canvas.setAttribute('aria-label', `No sampled predictions match the selected filters; this does not mean the full ${populationDescription()} is empty`);
    return;
  }
  const values = points.flatMap((point) => [point.actual, point.predicted]);
  let min = Math.min(...values); let max = Math.max(...values);
  const span = Math.max(max - min, 0.001);
  min -= span * 0.05; max += span * 0.05;
  const x = (value) => margin.left + ((value - min) / (max - min)) * plotWidth;
  const y = (value) => margin.top + plotHeight - ((value - min) / (max - min)) * plotHeight;
  state.chartHitTest = { points, x, y };
  context.strokeStyle = '#29363e'; context.lineWidth = 1; context.fillStyle = '#9aa8af';
  context.font = '11px ui-monospace, monospace'; context.textAlign = 'right'; context.textBaseline = 'middle';
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + ((max - min) * tick) / 4;
    const yy = y(value);
    context.beginPath(); context.moveTo(margin.left, yy); context.lineTo(width - margin.right, yy); context.stroke();
    context.fillText(value.toFixed(2), margin.left - 9, yy);
  }
  context.strokeStyle = '#f0f3ed'; context.setLineDash([5, 5]); context.lineWidth = 1.25;
  context.beginPath(); context.moveTo(x(min), y(min)); context.lineTo(x(max), y(max)); context.stroke(); context.setLineDash([]);
  context.fillStyle = '#4ad3c0'; context.globalAlpha = 0.72;
  for (const point of points) {
    context.beginPath(); context.arc(x(point.actual), y(point.predicted), 2.6, 0, Math.PI * 2); context.fill();
  }
  context.globalAlpha = 1; context.fillStyle = '#dce4e4'; context.font = '12px sans-serif'; context.textAlign = 'center'; context.textBaseline = 'alphabetic';
  context.textBaseline = 'top';
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + ((max - min) * tick) / 4;
    context.fillText(value.toFixed(2), x(value), height - margin.bottom + 10);
  }
  context.fillStyle = '#dce4e4'; context.font = '12px sans-serif';
  context.fillText('Actual residual (seconds)', margin.left + plotWidth / 2, height - 13);
  context.save(); context.translate(16, margin.top + plotHeight / 2); context.rotate(-Math.PI / 2);
  context.fillText('Predicted residual (seconds)', 0, 0); context.restore();
  const raceText = state.raceId === 'all'
    ? (resultRaceIds().length === 1 ? `${raceName(resultRaceIds()[0])} Grand Prix` : 'all races')
    : `${raceName(state.raceId)} Grand Prix`;
  const driverText = state.driver === 'all' ? 'all drivers' : `driver ${state.driver}`;
  $('sampleLabel').textContent = `Chart sample: ${integer(points.length)} / ${integer(state.run.prediction_sample.source_rows)} · ${raceText} · ${driverText}`;
  renderPredictionTable(orderedSamplePoints(points));
  canvas.setAttribute('aria-label', `Actual versus predicted residuals for ${points.length} archived sampled predictions, ${raceText}, ${driverText}`);
}

function hidePointTooltip() {
  const tooltip = $('pointTooltip');
  if (!tooltip) return;
  tooltip.hidden = true;
  tooltip.setAttribute('aria-hidden', 'true');
}

function nearestScatterPoint(event) {
  const geometry = state.chartHitTest;
  if (!geometry) return null;
  const bounds = $('predictionChart').getBoundingClientRect();
  const px = event.clientX - bounds.left;
  const py = event.clientY - bounds.top;
  let nearest = null;
  let nearestDistance = CHART_HIT_RADIUS_PX ** 2;
  let nearbyCount = 0;
  for (const point of geometry.points) {
    const dx = geometry.x(Number(point.actual)) - px;
    const dy = geometry.y(Number(point.predicted)) - py;
    const distance = dx * dx + dy * dy;
    if (distance <= CHART_HIT_RADIUS_PX ** 2) nearbyCount += 1;
    if (distance <= nearestDistance) { nearest = point; nearestDistance = distance; }
  }
  return nearest ? { point: nearest, nearbyCount } : null;
}

function inspectScatterPoint(event) {
  if (event.pointerType === 'touch') return hidePointTooltip();
  const tooltip = $('pointTooltip');
  const nearest = nearestScatterPoint(event);
  $('predictionChart').style.cursor = nearest ? 'pointer' : 'default';
  if (!nearest || !tooltip) return hidePointTooltip();
  const point = nearest.point;
  const error = Number(point.predicted) - Number(point.actual);
  const crowded = nearest.nearbyCount > 1 ? ` · ${nearest.nearbyCount} sampled points within ${CHART_HIT_RADIUS_PX} px` : '';
  tooltip.textContent = `Nearest sampled prediction${crowded} | ${raceName(point.race_id)} · driver ${point.driver} · lap ${point.lap} · mini-sector ${Number(point.mini_sector) + 1} | Actual residual ${number(point.actual, 5)} s · predicted ${number(point.predicted, 5)} s · error ${number(error, 5)} s`;
  tooltip.hidden = false;
  tooltip.setAttribute('aria-hidden', 'false');
  const chartRect = $('predictionChart').getBoundingClientRect();
  const wrapRect = $('predictionChart').parentElement.getBoundingClientRect();
  const tipRect = tooltip.getBoundingClientRect();
  const left = Math.max(8, Math.min(event.clientX - wrapRect.left + 12, Math.max(8, wrapRect.width - tipRect.width - 8)));
  const top = Math.max(8, Math.min(event.clientY - chartRect.top + 12, Math.max(8, chartRect.height - tipRect.height - 8)));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function selectScatterPoint(event) {
  // Native click fires after a tap but not a scroll gesture; leave touch defaults intact.
  const nearest = nearestScatterPoint(event);
  const point = nearest?.point;
  if (!point) return;
  hidePointTooltip();
  state.selectedPoint = point;
  const ordered = orderedSamplePoints();
  const index = ordered.indexOf(point);
  if (index < 0) return;
  const pageSize = 40;
  state.predictionPage = Math.floor(index / pageSize);
  $('predictionDataDetails').open = true;
  renderPredictionTable(ordered);
  const selectedRow = $('predictionTableBody').querySelector('[data-selected-point="true"]');
  selectedRow?.focus({ preventScroll: true });
  selectedRow?.scrollIntoView({ block: 'nearest' });
}

function enableScatterInspection() {
  const canvas = $('predictionChart');
  canvas.addEventListener('pointermove', inspectScatterPoint);
  canvas.addEventListener('pointerleave', hidePointTooltip);
  canvas.addEventListener('click', selectScatterPoint);
}

function renderPredictionTable(points) {
  const pageSize = 40;
  const pages = Math.max(1, Math.ceil(points.length / pageSize));
  state.predictionPage = Math.min(state.predictionPage || 0, pages - 1);
  const start = state.predictionPage * pageSize;
  const rows = points.slice(start, start + pageSize);
  const body = $('predictionTableBody');
  body.replaceChildren();
  for (const point of rows) {
    const row = document.createElement('tr');
    if (point === state.selectedPoint) {
      row.tabIndex = -1;
      row.dataset.selectedPoint = 'true';
      row.classList.add('selected-point-row');
    }
    row.append(cell('td', raceName(point.race_id)), cell('td', point.driver),
      cell('td', integer(point.lap)), cell('td', integer(Number(point.mini_sector) + 1)),
      cell('td', number(point.actual, 5)), cell('td', number(point.predicted, 5)),
      cell('td', number(point.predicted - point.actual, 5)));
    body.append(row);
  }
  $('predictionPageStatus').textContent = `Showing ${points.length ? start + 1 : 0}–${Math.min(start + pageSize, points.length)} of ${integer(points.length)} filtered sampled predictions · page ${state.predictionPage + 1} of ${pages}`;
  $('predictionPagePrev').disabled = state.predictionPage === 0;
  $('predictionPageNext').disabled = state.predictionPage >= pages - 1;
  $('predictionTableCaption').textContent = `Deterministic sample rows matching the race and driver filters, in ${sampleOrderLabel()}; mini-sector labels are 1–20. This order applies only to sampled rows, not the full ${populationDescription()}.`;
  $('predictionRowsSummary').textContent = `View sampled rows · ${integer(points.length)} match current filters · ${sampleOrderLabel()}`;
  $('predictionPageStatus').textContent += ` · ${sampleOrderLabel()}`;
  $('downloadSampleCsv').disabled = points.length === 0;
}

function readFiltersFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const raceValues = new Set(Array.from($('raceFilter').options, (option) => option.value));
  const driverValues = new Set(Array.from($('driverFilter').options, (option) => option.value));
  const requestedRace = params.get('race');
  const requestedDriver = params.get('driver');
  const requestedOrder = params.get('order');
  state.raceId = raceValues.size === 1 ? [...raceValues][0] : raceValues.has(requestedRace) ? requestedRace : 'all';
  state.driver = driverValues.has(requestedDriver) ? requestedDriver : 'all';
  state.sampleOrder = ['source', 'largest-error', 'smallest-error'].includes(requestedOrder) ? requestedOrder : 'source';
  $('raceFilter').value = state.raceId;
  $('driverFilter').value = state.driver;
  $('sampleOrderFilter').value = state.sampleOrder;
}

function writeFiltersToUrl(mode = 'push') {
  const url = new URL(window.location.href);
  if (state.raceId === 'all') url.searchParams.delete('race');
  else url.searchParams.set('race', state.raceId);
  if (state.driver === 'all') url.searchParams.delete('driver');
  else url.searchParams.set('driver', state.driver);
  if (state.sampleOrder === 'source') url.searchParams.delete('order');
  else url.searchParams.set('order', state.sampleOrder);
  const next = `${url.pathname}${url.search}${url.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next === current) return;
  if (mode === 'replace') window.history.replaceState(null, '', next);
  else window.history.pushState(null, '', next);
}

function downloadFilteredSample() {
  const points = orderedSamplePoints();
  if (!points.length) return;
  const header = ['model_run_id', 'race_id', 'driver_number', 'lap', 'mini_sector_index_zero_based',
    'actual_residual_seconds', 'predicted_residual_seconds', 'prediction_error_seconds'];
  const quote = (value) => {
    let text = String(value ?? '');
    if (!(typeof value === 'number' && Number.isFinite(value))
      && /^[\t\r\n ]*[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = points.map((point) => [state.run.run_id, point.race_id, point.driver, point.lap, point.mini_sector,
    point.actual, point.predicted, point.predicted - point.actual]);
  const csv = [header, ...rows].map((row) => row.map(quote).join(',')).join('\r\n');
  const blob = new Blob([`\uFEFF${csv}\r\n`], { type: 'text/csv;charset=utf-8' });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const runPart = state.run.run_id.slice(0, 12).replace(/[^a-zA-Z0-9-]/g, '-');
  const racePart = state.raceId === 'all' ? 'all-races' : state.raceId.replace(/[^a-zA-Z0-9-]/g, '-');
  const driverPart = state.driver === 'all' ? 'all-drivers' : `driver-${state.driver.replace(/[^a-zA-Z0-9-]/g, '-')}`;
  link.href = objectUrl;
  link.download = `hammertime-${runPart}-${racePart}-${driverPart}-sampled-predictions.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    $('downloadStatus').textContent = `CSV prepared: ${integer(points.length)} filtered rows from the deterministic ${integer(state.run.prediction_sample.points.length)}-point chart sample, in ${sampleOrderLabel()}; the full ${populationDescription()} contains ${integer(state.run.metrics.sample_count)} targets.`;
}

function downloadViewProvenance() {
  if (!state.run) return;
  try {
    const run = state.run;
    const matchedPoints = matchingPoints();
    const manifest = {
      artifact_type: 'Archived sampled-view provenance (metadata only)',
      schema_version: 1,
      inference_status: 'saved evaluation output; no live inference was run',
      predictions_included: false,
      run_id: run.run_id,
      source: run.source,
      model: run.model,
      target: run.target,
      view: {
        race_id: state.raceId,
        driver_number: state.driver === 'all' ? null : Number(state.driver),
        sample_order: state.sampleOrder,
        ordering_scope: 'filtered deterministic sample only',
      },
      counts: {
        full_inference_node_targets: Number(run.metrics.sample_count),
        prediction_sample_source_rows: Number(run.prediction_sample.source_rows),
        deterministic_sample_points: run.prediction_sample.points.length,
        filtered_sample_points: matchedPoints.length,
        train_graph_windows: Number(run.split.train_window_count),
        validation_graph_windows: Number(run.split.validation_window_count),
        test_graph_windows: Number(run.split.test_window_count),
      },
      sampling_rule: run.prediction_sample.method,
      split: { method: run.split.method, test_races: [...run.split.test_races] },
      inference_scope: run.inference_scope ? { ...run.inference_scope } : null,
      checkpoint: {
        artifact: run.checkpoint.artifact,
        selected_epoch: Number(run.checkpoint.selected_epoch),
        sha256: run.checkpoint.sha256,
      },
      provenance: {
        fingerprint: run.provenance.fingerprint,
        config_sha256: run.provenance.config_sha256,
        prepared_artifact_sha256: run.provenance.prepared_artifact_sha256,
        implementation_sha256: { ...run.provenance.implementation_sha256 },
        source_artifact_sha256: { ...run.provenance.source_artifact_sha256 },
      },
    };
    const blob = new Blob([`${JSON.stringify(manifest, null, 2)}\n`], { type: 'application/json;charset=utf-8' });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const runPart = run.run_id.slice(0, 12).replace(/[^a-zA-Z0-9-]/g, '-');
    const racePart = state.raceId === 'all' ? 'all-races' : state.raceId.replace(/[^a-zA-Z0-9-]/g, '-');
    const driverPart = state.driver === 'all' ? 'all-drivers' : `driver-${state.driver.replace(/[^a-zA-Z0-9-]/g, '-')}`;
    link.href = objectUrl;
    link.download = `hammertime-${runPart}-${racePart}-${driverPart}-${state.sampleOrder}-view-provenance.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    $('downloadStatus').textContent = `View provenance JSON prepared for run ${runPart}: ${integer(matchedPoints.length)} matching rows in the deterministic sample; inference targets and original held-out split counts are recorded separately.`;
  } catch (error) {
    $('downloadStatus').textContent = `Could not prepare view provenance JSON: ${error.message}`;
  }
}

function renderExplorer() {
  renderRaceMetrics();
  drawScatter();
}

function renderBreakdown() {
  const isDriver = state.breakdown === 'driver';
  const rows = isDriver ? state.run.driver_metrics : state.run.mini_sector_metrics;
  const key = isDriver ? 'driver' : 'mini_sector';
  const title = isDriver ? 'By driver number' : 'By mini-sector';
  $('breakdownTitle').textContent = title;
  $('breakdownDescription').textContent = isDriver
    ? `Full ${populationDescription()} by driver number. Bars show exact MAE; target counts are shown per row.`
    : `Full ${populationDescription()} by mini-sector index. Bars show exact MAE; target counts are shown per row.`;
  const root = $('breakdownList');
  root.replaceChildren();
  const sorted = [...rows].sort((a, b) => Number(b.mae_sec) - Number(a.mae_sec));
  const max = Math.max(...sorted.map((row) => Number(row.mae_sec)), 0.001);
  for (const row of sorted) {
    const item = cell('div', undefined, 'breakdown-row');
    const label = isDriver ? `Driver #${row[key]}` : `Mini-sector ${Number(row[key]) + 1}`;
    const top = cell('div', undefined, 'breakdown-top');
    top.append(cell('span', label), cell('span', `${number(row.mae_sec)} s · n ${integer(row.sample_count)}`));
    const track = cell('div', undefined, 'bar-track');
    const bar = cell('span', undefined, 'bar-fill');
    bar.style.width = `${Math.max(2, (Number(row.mae_sec) / max) * 100)}%`;
    track.append(bar); item.append(top, track); root.append(item);
  }
}

function renderMethod() {
  const run = state.run;
  $('checkpointEpoch').textContent = run.checkpoint.selected_epoch;
  $('validationMae').textContent = `${number(run.checkpoint.best_validation_mae_sec)} s`;
  $('checkpointHash').textContent = run.checkpoint.sha256;
  $('checkpointHash').title = run.checkpoint.sha256;
  $('predictionRows').textContent = integer(run.prediction_sample.source_rows);
  $('samplingMethod').textContent = run.prediction_sample.method;
  $('testRaceList').replaceChildren();
  const provenanceRaces = run.inference_scope?.held_out_test === false
    ? [run.inference_scope.race_id] : run.split.test_races;
  for (const race of provenanceRaces) $('testRaceList').append(cell('span', raceName(race), 'race-chip'));
  $('targetDefinition').textContent = `Signed residual: actual mini-sector duration minus a clean-air reference fitted from the same full session; the target is clipped under the saved run's rules. Because the reference uses the full session, this result is retrospective, not a prospective live estimate or a causal measurement of traffic's effect.`;
  const hashes = $('hashList'); hashes.replaceChildren();
  const entries = [
    ['Run fingerprint', run.provenance.fingerprint],
    ['Config SHA-256', run.provenance.config_sha256],
    ['Prepared data SHA-256', run.provenance.prepared_artifact_sha256],
    ...Object.entries(run.provenance.source_artifact_sha256).map(([name, hash]) => [`${name} SHA-256`, hash]),
  ];
  for (const [label, value] of entries) {
    hashes.append(cell('dt', label));
    const dd = cell('dd', value); dd.title = value; hashes.append(dd);
  }
}

function updateExplorer() {
  state.raceId = $('raceFilter').value;
  state.driver = $('driverFilter').value;
  state.sampleOrder = $('sampleOrderFilter').value;
  state.predictionPage = 0;
  state.selectedPoint = null;
  writeFiltersToUrl('push');
  $('downloadStatus').textContent = '';
  renderExplorer();
}

function updateSampleOrder() {
  state.sampleOrder = $('sampleOrderFilter').value;
  state.predictionPage = 0;
  state.selectedPoint = null;
  writeFiltersToUrl('push');
  $('downloadStatus').textContent = '';
  renderPredictionTable(orderedSamplePoints());
}

function enableTableKeyboardScroll() {
  for (const region of document.querySelectorAll('.table-scroll[tabindex="0"]')) {
    region.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      if (region.scrollWidth <= region.clientWidth) return;
      event.preventDefault();
      const step = Math.max(40, Math.round(region.clientWidth * 0.8));
      region.scrollLeft += event.key === 'ArrowRight' ? step : -step;
    });
  }
}

function validMarinaBayReplay(data) {
  const numeric = (value) => typeof value === 'number' && Number.isFinite(value);
  const event = data?.event;
  const drivers = data?.drivers;
  const outline = data?.track_outline;
  const bounds = outline?.bounds_m;
  const replay = data?.replay;
  const sourceHashes = data?.provenance?.source_sha256;
  const startTime = data?.time_axis?.race_start_session_sec;
  if (data?.schema_version !== 1 || event?.race_id !== '2023-Singapore-R'
    || event?.circuit_id !== 'singapore' || event?.venue !== 'Marina Bay Street Circuit'
    || !numeric(event?.lap_length_m) || !Number.isInteger(event?.lap_count)
    || !Number.isInteger(event?.turn_count) || data?.evaluation_context?.held_out_test !== false
    || data?.evaluation_context?.split_role !== 'training race' || !numeric(startTime)
    || !Array.isArray(drivers) || drivers.length < 10
    || !drivers.every((driver) => driver && typeof driver.number === 'string'
      && typeof driver.acronym === 'string' && typeof driver.team === 'string')
    || new Set(drivers.map((driver) => driver.number)).size !== drivers.length
    || !outline || !Array.isArray(outline.points) || outline.points.length < 100
    || !outline.points.every((point) => numeric(point.x_m) && numeric(point.y_m) && numeric(point.session_time_sec))
    || !bounds || !numeric(bounds.min_x) || !numeric(bounds.max_x) || bounds.max_x <= bounds.min_x
    || !numeric(bounds.min_y) || !numeric(bounds.max_y) || bounds.max_y <= bounds.min_y
    || !drivers.some((driver) => driver.number === String(outline.driver_number))
    || !sourceHashes || typeof sourceHashes !== 'object'
    || !Object.values(sourceHashes).every((hash) => typeof hash === 'string' && /^[a-f0-9]{64}$/i.test(hash))
    || !replay || !Array.isArray(replay.frames) || replay.frames.length !== replay.frame_count
    || !Number.isInteger(replay.frame_stride) || replay.frame_stride < 1
    || replay.output_sample_hz <= 0 || !numeric(replay.output_sample_hz)
    || !Array.isArray(replay.frames) || replay.frames.length < 1000) return false;
  let previousTime = -Infinity;
  let observedPositions = 0;
  for (const frame of replay.frames) {
    if (!numeric(frame.session_time_sec) || !numeric(frame.race_elapsed_sec)
      || frame.session_time_sec <= previousTime
      || (previousTime !== -Infinity
        && Math.abs(frame.session_time_sec - previousTime - (1 / replay.output_sample_hz)) > 1e-6)
      || Math.abs(frame.race_elapsed_sec - (frame.session_time_sec - startTime)) > 1e-6
      || !Array.isArray(frame.positions) || frame.positions.length !== drivers.length) return false;
    previousTime = frame.session_time_sec;
    for (const position of frame.positions) {
      if (position === null) continue;
      if (!position || !numeric(position.x_m) || !numeric(position.y_m)
        || !(position.lap_number === null || (Number.isInteger(position.lap_number) && position.lap_number >= 1))
        || !(position.speed_kph === null || (numeric(position.speed_kph) && position.speed_kph >= 0))
        || typeof position.is_pit_lap !== 'boolean') return false;
      observedPositions += 1;
    }
  }
  return observedPositions === replay.sampled_position_count;
}

function formatReplayTime(seconds) {
  const whole = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remainder = whole % 60;
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, '0')).join(':');
}

function replayReferenceDriver() {
  if (replayState.focusDriver !== 'all') return replayState.focusDriver;
  return String(replayState.data?.track_outline?.driver_number || replayState.data?.drivers?.[0]?.number || '');
}

function replayDriverAtFrame(frameIndex, driverNumber = replayReferenceDriver()) {
  const driverIndex = replayState.data.drivers.findIndex((driver) => driver.number === String(driverNumber));
  if (driverIndex < 0) return null;
  return replayState.data.replay.frames[frameIndex]?.positions[driverIndex] || null;
}

function populateReplayDriverSelector() {
  const select = $('replayDriverFilter');
  select.replaceChildren(new Option('All recorded cars', 'all'));
  for (const driver of replayState.data.drivers) {
    select.append(new Option(`${driver.acronym} · #${driver.number}`, driver.number));
  }
  select.disabled = false;
  $('replayLapFilter').disabled = false;
  $('replayPlayPause').disabled = false;
  $('replayReset').disabled = false;
  $('replayScrubber').disabled = false;
  document.querySelectorAll('[data-replay-speed]').forEach((button) => { button.disabled = false; });
}

function availableReplayLaps(driverNumber = replayReferenceDriver()) {
  const driverIndex = replayState.data.drivers.findIndex((driver) => driver.number === String(driverNumber));
  if (driverIndex < 0) return [];
  const lapIndices = new Map();
  replayState.data.replay.frames.forEach((frame, index) => {
    const lap = frame.positions[driverIndex]?.lap_number;
    if (Number.isInteger(lap) && !lapIndices.has(lap)) lapIndices.set(lap, index);
  });
  return [...lapIndices.entries()].sort((a, b) => a[0] - b[0]);
}

function populateReplayLapSelector() {
  const select = $('replayLapFilter');
  select.replaceChildren(new Option('Jump to a lap…', ''));
  const reference = replayReferenceDriver();
  const descriptor = replayState.data.drivers.find((driver) => driver.number === reference);
  for (const [lap, frameIndex] of availableReplayLaps(reference)) {
    select.append(new Option(`Lap ${lap}`, String(frameIndex)));
  }
  select.title = replayState.focusDriver === 'all'
    ? `Lap jumps follow ${descriptor?.acronym || ''} #${reference} as the reference car`
    : `Lap jumps follow ${descriptor?.acronym || ''} #${reference}`;
  $('replayLapReference').textContent = replayState.focusDriver === 'all'
    ? `With all cars shown, lap jumps follow ${descriptor?.acronym || ''} #${reference}.`
    : `Lap jumps follow ${descriptor?.acronym || ''} #${reference}.`;
  select.value = '';
  const currentPosition = replayDriverAtFrame(replayState.frameIndex, reference);
  if (currentPosition?.lap_number != null) {
    const match = [...select.options].find((option) => Number(option.value) >= 0
      && replayState.data.replay.frames[Number(option.value)]?.positions[
        replayState.data.drivers.findIndex((driver) => driver.number === reference)
      ]?.lap_number === currentPosition.lap_number);
    if (match) select.value = match.value;
  }
}

function drawMarinaBayTrack(updateSampleDetails = true) {
  const data = replayState.data;
  const canvas = $('marinaBayCanvas');
  const context = canvas?.getContext('2d');
  if (!data || !canvas || !context) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(rect.width * ratio);
  const pixelHeight = Math.round(rect.height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#0a1013';
  context.fillRect(0, 0, width, height);

  const bounds = data.track_outline.bounds_m;
  const padding = Math.max(32, Math.min(56, width * 0.06));
  const spanX = bounds.max_x - bounds.min_x;
  const spanY = bounds.max_y - bounds.min_y;
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const mapWidth = spanX * scale;
  const mapHeight = spanY * scale;
  const offsetX = (width - mapWidth) / 2;
  const offsetY = (height - mapHeight) / 2;
  const mapPoint = (x, y) => ({
    x: offsetX + (x - bounds.min_x) * scale,
    y: height - offsetY - (y - bounds.min_y) * scale,
  });

  const outline = data.track_outline.points;
  context.beginPath();
  outline.forEach((point, index) => {
    const mapped = mapPoint(point.x_m, point.y_m);
    if (index === 0) context.moveTo(mapped.x, mapped.y);
    else context.lineTo(mapped.x, mapped.y);
  });
  context.closePath();
  context.strokeStyle = '#19242a';
  context.lineWidth = 20;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.stroke();
  context.strokeStyle = '#4ad3c0';
  context.lineWidth = 2;
  context.stroke();
  const start = mapPoint(outline[0].x_m, outline[0].y_m);
  context.beginPath();
  context.arc(start.x, start.y, 5, 0, Math.PI * 2);
  context.fillStyle = '#f1f3ee';
  context.fill();
  context.fillStyle = '#aab4b8';
  context.font = '10px ui-monospace, monospace';
  const startLabel = 'LAP TRACE START';
  const labelRight = start.x + 8 + context.measureText(startLabel).width;
  const labelLeft = start.x - 8 - context.measureText(startLabel).width;
  let labelX;
  if (labelRight <= width - 6) {
    context.textAlign = 'left';
    labelX = start.x + 8;
  } else if (labelLeft >= 6) {
    context.textAlign = 'right';
    labelX = start.x - 8;
  } else {
    context.textAlign = 'left';
    labelX = Math.max(6, Math.min(width - 6 - context.measureText(startLabel).width, start.x + 8));
  }
  context.textBaseline = 'bottom';
  const labelY = Math.max(14, Math.min(height - 8, start.y - 6));
  context.fillText(startLabel, labelX, labelY);

  const frame = data.replay.frames[replayState.frameIndex];
  const nextFrame = data.replay.frames[replayState.frameIndex + 1] || null;
  const outputHz = Number(data.replay.output_sample_hz);
  const interval = nextFrame ? nextFrame.race_elapsed_sec - frame.race_elapsed_sec : 0;
  const intervalIsRecorded = nextFrame && Number.isFinite(outputHz) && outputHz > 0
    && Math.abs(interval - 1 / outputHz) <= 1e-6;
  const alpha = intervalIsRecorded
    ? Math.max(0, Math.min(1, (replayState.elapsedSeconds - frame.race_elapsed_sec) / interval)) : 0;
  const focusIndex = data.drivers.findIndex((driver) => driver.number === replayState.focusDriver);
  let positionCount = 0;
  frame.positions.forEach((position, index) => {
    const following = nextFrame?.positions[index] || null;
    let visualPosition = position;
    const sameLap = position && following && position.lap_number === following.lap_number;
    const pitDiscontinuity = position?.is_pit_lap === true || following?.is_pit_lap === true;
    const coordinateDelta = position && following
      ? Math.hypot(following.x_m - position.x_m, following.y_m - position.y_m) : Infinity;
    const maxRecordedSpeed = position && following
      && Number.isFinite(position.speed_kph) && Number.isFinite(following.speed_kph)
      ? Math.max(position.speed_kph, following.speed_kph) / 3.6 : 0;
    const plausibleMotion = position && following && maxRecordedSpeed > 0
      && coordinateDelta <= maxRecordedSpeed * interval + 0.1;
    const lapIncrement = position && following && Number.isInteger(position.lap_number)
      && Number.isInteger(following.lap_number) && following.lap_number === position.lap_number + 1;
    const lapContinuous = sameLap || (lapIncrement && plausibleMotion);
    const canInterpolate = alpha > 0 && intervalIsRecorded && position && following
      && lapContinuous && !pitDiscontinuity && plausibleMotion;
    if (canInterpolate) {
      visualPosition = {
        ...position,
        x_m: position.x_m + (following.x_m - position.x_m) * alpha,
        y_m: position.y_m + (following.y_m - position.y_m) * alpha,
      };
    }
    if (!visualPosition) return;
    positionCount += 1;
    const driver = data.drivers[index];
    const mapped = mapPoint(visualPosition.x_m, visualPosition.y_m);
    const focused = index === focusIndex;
    const teamColor = window.HammerTimeDriverAssets?.officialTeamColorFor(driver, marinaBayDriverAssetMap)
      || MARINA_BAY_TEAM_COLORS_2023[driver.team] || '#aab4b8';
    context.beginPath();
    context.arc(mapped.x, mapped.y, focused ? 7 : 4.1, 0, Math.PI * 2);
    context.fillStyle = teamColor;
    context.fill();
    context.lineWidth = focused ? 2.5 : 0.8;
    context.strokeStyle = focused ? '#f5f5ef' : '#0c1115';
    context.stroke();
    if (focused || width > 620) {
      context.font = focused ? '700 10px ui-monospace, monospace' : '9px ui-monospace, monospace';
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      context.lineWidth = 3;
      context.strokeStyle = '#0a1013';
      context.strokeText(driver.number, mapped.x + 6, mapped.y);
      context.fillStyle = focused ? '#fff' : '#d9e3e3';
      context.fillText(driver.number, mapped.x + 6, mapped.y);
    }
  });

  $('trackPositionCoverage').textContent = `Markers ${positionCount}/${data.drivers.length}`;
  const reference = replayReferenceDriver();
  const refDriver = data.drivers.find((driver) => driver.number === reference);
  const referencePosition = replayDriverAtFrame(replayState.frameIndex, reference);
  $('trackCurrentLap').textContent = Number.isInteger(referencePosition?.lap_number)
    ? `Lap ${referencePosition.lap_number} · ${replayState.focusDriver === 'all' ? `${refDriver?.acronym} #${reference} reference` : `#${reference}`}`
    : `Lap — · ${refDriver?.acronym || ''} #${reference} reference`;
  const current = replayState.elapsedSeconds;
  $('trackReplayClock').textContent = `Race elapsed ${formatReplayTime(current)}`;
  $('replayScrubber').value = String(current);
  $('replayScrubberRange').textContent = `${formatReplayTime(0)} – ${formatReplayTime(data.replay.frames.at(-1).race_elapsed_sec)}`;
  if (replayState.focusDriver === 'all') {
    const focusTarget = $('replayFocusDriver');
    const focusLabel = `All recorded cars · lap reference ${refDriver?.acronym || ''} #${reference}`;
    if (window.HammerTimeDriverAssets) window.HammerTimeDriverAssets.renderSummary(focusTarget, null, focusLabel, marinaBayDriverAssetMap);
    else focusTarget.textContent = focusLabel;
    $('replayFocusSpeed').textContent = '—';
    $('replayFocusStatus').textContent = `${positionCount} recorded positions at this sample`;
  } else {
    const descriptor = data.drivers[focusIndex];
    const position = frame.positions[focusIndex];
    const focusTarget = $('replayFocusDriver');
    const focusLabel = `${descriptor.acronym} · #${descriptor.number}`;
    if (window.HammerTimeDriverAssets) window.HammerTimeDriverAssets.renderSummary(focusTarget, descriptor, focusLabel, marinaBayDriverAssetMap);
    else focusTarget.textContent = focusLabel;
    $('replayFocusSpeed').textContent = position?.speed_kph == null ? '—' : `${number(position.speed_kph, 1)} km/h`;
    $('replayFocusStatus').textContent = position
      ? `Recorded position · lap ${position.lap_number ?? '—'}` : 'No recorded coordinates at this sample';
  }
  const when = `${formatReplayTime(current)} replay elapsed`;
  if (updateSampleDetails) {
    const sampleWhen = `${formatReplayTime(frame.race_elapsed_sec)} replay elapsed; source session timestamp ${number(frame.session_time_sec, 3)} seconds`;
    $('trackFrameCaption').textContent = `Recorded FastF1 data for the preceding 2 Hz sample at ${sampleWhen}; canvas markers interpolate only between adjacent samples that pass continuity checks.`;
    $('trackFrameRows').replaceChildren();
    data.drivers.forEach((driver, index) => {
      const position = frame.positions[index];
      const row = document.createElement('tr');
      const driverCell = cell('th', `${driver.acronym} · #${driver.number}`);
      driverCell.scope = 'row';
      row.append(driverCell,
        cell('td', position?.lap_number ?? '—'),
        cell('td', position ? number(position.x_m, 3) : '—'),
        cell('td', position ? number(position.y_m, 3) : '—'),
        cell('td', position?.speed_kph == null ? '—' : number(position.speed_kph, 1)),
        cell('td', position ? 'Coordinates recorded' : 'No coordinates at sample'));
      $('trackFrameRows').append(row);
    });
    canvas.setAttribute('aria-label', `Archived 2023 Singapore Grand Prix replay at ${formatReplayTime(current)}. ${positionCount} markers are visible. Canvas interpolation is visual only; unsafe intervals use the current source point until the next sample, and missing current positions are omitted.`);
  }
  if ($('replayLapFilter').options.length > 1) {
    const selectedLap = referencePosition?.lap_number;
    const matchingOption = [...$('replayLapFilter').options].find((option) => option.value
      && data.replay.frames[Number(option.value)]?.positions[
        data.drivers.findIndex((driver) => driver.number === reference)
      ]?.lap_number === selectedLap);
    $('replayLapFilter').value = matchingOption?.value || '';
  }
}

function populateReplayMetadata() {
  const { event, track_outline: outline, replay } = replayState.data;
  $('trackCircuitFacts').textContent = `${event.venue} · ${(event.lap_length_m / 1000).toFixed(2)} km · ${event.turn_count} turns · ${event.lap_count} laps`;
  const outlineDriver = outline.driver_acronym
    || replayState.data.drivers.find((driver) => driver.number === String(outline.driver_number))?.acronym
    || `#${outline.driver_number}`;
  $('trackOutlineSource').textContent = `Track trace: FastF1 ${outlineDriver} #${outline.driver_number}, lap ${outline.lap_number}; ${outline.source_sample_count} raw Source=pos samples sorted within one recorded ${number(outline.lap_duration_sec, 3)} s lap. Coordinate trace spans ${number(outline.distance_delta_m, 1)} m; it is a trace, not a surveyed boundary.`;
  $('trackReplayStatus').textContent = `Archived replay ready · ${event.date} · ${replay.output_sample_hz} recorded samples/s`;
  $('replayScrubber').max = String(replay.frames.at(-1).race_elapsed_sec);
  $('replayScrubber').step = String(1 / replay.output_sample_hz);
}

function setReplayFrame(index) {
  const maxIndex = replayState.data.replay.frames.length - 1;
  replayState.frameIndex = Math.max(0, Math.min(maxIndex, Math.floor(index)));
  replayState.elapsedSeconds = replayState.data.replay.frames[replayState.frameIndex].race_elapsed_sec;
  drawMarinaBayTrack();
}

function setReplayElapsed(seconds) {
  const frames = replayState.data.replay.frames;
  replayState.elapsedSeconds = Math.max(0, Math.min(frames.at(-1).race_elapsed_sec, seconds));
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].race_elapsed_sec <= replayState.elapsedSeconds) low = middle;
    else high = middle - 1;
  }
  const sampleChanged = low !== replayState.frameIndex;
  replayState.frameIndex = low;
  drawMarinaBayTrack(sampleChanged);
}

function pauseMarinaBayReplay() {
  replayState.playing = false;
  replayState.lastAnimationTime = null;
  if (replayState.animationId !== null) cancelAnimationFrame(replayState.animationId);
  replayState.animationId = null;
  $('replayPlayPause').textContent = 'Play';
  $('replayPlayPause').setAttribute('aria-label', 'Play archived race replay');
  $('trackReplayStatus').textContent = 'Archived replay paused';
}

function advanceMarinaBayReplay(timestamp) {
  if (!replayState.playing) return;
  if (replayState.lastAnimationTime === null) replayState.lastAnimationTime = timestamp;
  const deltaSeconds = Math.min(0.25, Math.max(0, timestamp - replayState.lastAnimationTime) / 1000);
  replayState.lastAnimationTime = timestamp;
  const frames = replayState.data.replay.frames;
  const nextTime = replayState.elapsedSeconds + deltaSeconds * replayState.speed;
  if (nextTime >= frames.at(-1).race_elapsed_sec) {
    setReplayFrame(frames.length - 1);
    pauseMarinaBayReplay();
    return;
  }
  setReplayElapsed(nextTime);
  replayState.animationId = requestAnimationFrame(advanceMarinaBayReplay);
}

function playMarinaBayReplay() {
  if (!replayState.data) return;
  if (replayState.frameIndex >= replayState.data.replay.frames.length - 1) setReplayFrame(0);
  replayState.playing = true;
  replayState.lastAnimationTime = null;
  $('replayPlayPause').textContent = 'Pause';
  $('replayPlayPause').setAttribute('aria-label', 'Pause archived race replay');
  $('trackReplayStatus').textContent = `Archived replay playing · ${replayState.speed}× · visual interpolation between valid samples`;
  replayState.animationId = requestAnimationFrame(advanceMarinaBayReplay);
}

function bindMarinaBayControls() {
  $('replayDriverFilter').addEventListener('change', () => {
    replayState.focusDriver = $('replayDriverFilter').value;
    populateReplayLapSelector();
    drawMarinaBayTrack();
  });
  $('replayLapFilter').addEventListener('change', () => {
    if ($('replayLapFilter').value) setReplayFrame(Number($('replayLapFilter').value));
  });
  $('replayPlayPause').addEventListener('click', () => {
    if (replayState.playing) pauseMarinaBayReplay();
    else playMarinaBayReplay();
  });
  $('replayReset').addEventListener('click', () => {
    pauseMarinaBayReplay();
    setReplayFrame(0);
  });
  document.querySelectorAll('[data-replay-speed]').forEach((button) => {
    button.addEventListener('click', () => {
      replayState.speed = Number(button.dataset.replaySpeed);
      document.querySelectorAll('[data-replay-speed]').forEach((item) =>
        item.setAttribute('aria-pressed', String(item === button)));
      if (replayState.playing) {
        $('trackReplayStatus').textContent = `Archived replay playing · ${replayState.speed}× · sampled positions only`;
      }
    });
  });
  $('replayScrubber').addEventListener('input', () => {
    pauseMarinaBayReplay();
    setReplayElapsed(Number($('replayScrubber').value));
  });
  window.addEventListener('resize', drawMarinaBayTrack, { passive: true });
  if ('ResizeObserver' in window) {
    replayState.resizeObserver = new ResizeObserver(drawMarinaBayTrack);
    replayState.resizeObserver.observe($('marinaBayCanvas'));
  }
}

async function startMarinaBayReplay() {
  const started = performance.now();
  try {
    const response = await fetch(MARINA_BAY_REPLAY_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Archived Singapore replay request failed (HTTP ${response.status})`);
    const data = await response.json();
    if (!validMarinaBayReplay(data)) throw new Error('The archived Singapore replay failed its required data checks.');
    replayState.data = data;
    populateReplayMetadata();
    populateReplayDriverSelector();
    populateReplayLapSelector();
    bindMarinaBayControls();
    setReplayFrame(0);
    window.HammerTimeDriverAssets?.load().then((map) => {
      marinaBayDriverAssetMap = map;
      drawMarinaBayTrack();
    }).catch((error) => console.info('Singapore team logos unavailable', error));
    console.info(`Marina Bay replay first frame rendered in ${(performance.now() - started).toFixed(0)} ms`);
  } catch (error) {
    $('trackReplayStatus').textContent = 'Archived replay unavailable';
    $('trackReplayError').hidden = false;
    $('trackReplayError').textContent = `${error.message} No synthetic position fallback is being shown.`;
    $('marinaBayCanvas').setAttribute('aria-label', 'Marina Bay archived replay unavailable because verified Singapore position data could not be loaded');
    $('trackFrameRows').replaceChildren();
    const row = document.createElement('tr');
    row.append(cell('td', 'Verified Singapore replay unavailable.', undefined));
    row.cells[0].colSpan = 6;
    $('trackFrameRows').append(row);
  }
}

async function startMarinaBayInference() {
  try {
    const response = await fetch(MARINA_BAY_INFERENCE_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Singapore checkpoint inference request failed (HTTP ${response.status})`);
    const data = await response.json();
    if (data.schema_version !== 1 || data.run_id !== '769a76aab164b514cc0a'
      || data.scope?.race_id !== '2023-Singapore-R' || data.scope?.held_out_test !== false
      || data.scope?.in_sample !== true || !Array.isArray(data.records)
      || data.records.length !== data.count?.target_labels_in_lap
      || !data.records.every((row) => row?.race_id === '2023-Singapore-R' && row.driver === '55'
        && row.lap_number === 1 && Number.isInteger(row.mini_sector_index_zero_based)
        && Number.isFinite(row.actual_residual_sec)
        && (row.predicted_residual_sec === null || Number.isFinite(row.predicted_residual_sec)))) {
      throw new Error('Singapore inference export failed required provenance and row checks.');
    }
    const predictions = data.records.filter((row) => row.predicted_residual_sec !== null);
    $('marinaInferenceCount').textContent = String(predictions.length);
    $('marinaInferenceMae').textContent = number(data.aggregate.mae_sec, 5);
    $('marinaInferenceRmse').textContent = number(data.aggregate.rmse_sec, 5);
    $('marinaInferenceStatus').textContent = data.count.unavailable === 0
      ? `Checkpoint outputs loaded · all ${predictions.length} mini-sectors predicted (0 unavailable)`
      : `Checkpoint outputs loaded · ${data.count.unavailable} unavailable`;
    $('marinaInferenceRows').replaceChildren();
    for (const record of data.records) {
      const row = document.createElement('tr');
      row.append(cell('th', String(record.mini_sector_display)),
        cell('td', record.predicted_residual_sec === null ? 'Unavailable' : number(record.predicted_residual_sec, 5)),
        cell('td', number(record.actual_residual_sec, 5)),
        cell('td', record.absolute_error_sec === null ? '—' : number(record.absolute_error_sec, 5)),
        cell('td', number(record.target_entry_time_sec, 3)));
      $('marinaInferenceRows').append(row);
    }
  } catch (error) {
    $('marinaInferenceStatus').textContent = 'Checkpoint inference unavailable';
    $('marinaInferenceError').hidden = false;
    $('marinaInferenceError').textContent = `${error.message} No fallback predictions are shown.`;
    const row = document.createElement('tr');
    const message = cell('td', 'Verified checkpoint outputs unavailable.');
    message.colSpan = 5;
    row.append(message);
    $('marinaInferenceRows').replaceChildren(row);
  }
}

async function start() {
  enableTableKeyboardScroll();
  enableScatterInspection();
  try {
    const response = await fetch(DATA_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Model result request failed (HTTP ${response.status})`);
    const run = await response.json();
    if (!validRun(run)) throw new Error('The season model export is missing required metrics or breakdown data. Regenerate it from the saved run.');
    state.run = run;
    renderMetricSummary(); fillRaceSelector(); renderRaceTable(); renderBaseline(); renderErrorDistribution(); renderPhysics(); renderMethod(); renderBreakdown();
    $('raceFilter').addEventListener('change', updateExplorer);
    $('driverFilter').addEventListener('change', updateExplorer);
    $('sampleOrderFilter').addEventListener('change', updateSampleOrder);
    $('breakdownFilter').addEventListener('change', () => { state.breakdown = $('breakdownFilter').value; renderBreakdown(); });
    $('predictionPagePrev').addEventListener('click', () => { state.predictionPage = Math.max(0, (state.predictionPage || 0) - 1); renderPredictionTable(orderedSamplePoints()); });
    $('predictionPageNext').addEventListener('click', () => { state.predictionPage = (state.predictionPage || 0) + 1; renderPredictionTable(orderedSamplePoints()); });
    $('downloadSampleCsv').addEventListener('click', downloadFilteredSample);
    $('downloadViewProvenance').addEventListener('click', downloadViewProvenance);
    window.addEventListener('resize', drawScatter, { passive: true });
    window.addEventListener('popstate', () => {
      state.selectedPoint = null;
      readFiltersFromUrl();
      state.predictionPage = 0;
      $('downloadStatus').textContent = '';
      renderExplorer();
      writeFiltersToUrl('replace');
    });
    readFiltersFromUrl();
    writeFiltersToUrl('replace');
    if (state.sampleOrder !== 'source') $('predictionDataDetails').open = true;
    renderExplorer();
  } catch (error) {
    $('loadStatus').textContent = 'Evaluation unavailable';
    $('metricGrid').setAttribute('aria-busy', 'false');
    $('dataError').hidden = false;
    $('dataError').textContent = `${error.message} No fallback or synthetic metrics are being shown.`;
    $('raceTableBody').replaceChildren();
    const row = document.createElement('tr');
    const messageCell = cell('td', 'Run data unavailable.');
    messageCell.colSpan = 6; row.append(messageCell); $('raceTableBody').replaceChildren(row);
    $('predictionChart').setAttribute('aria-label', 'Prediction chart unavailable because archived model results could not be loaded');
    $('predictionChart').getContext('2d')?.clearRect(0, 0, $('predictionChart').width, $('predictionChart').height);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  start();
  startMarinaBayReplay();
  startMarinaBayInference();
});
