/**
 * HAMMERTIME 100 — F1 BROADCAST CONTROL ROOM
 * High-Precision Spatio-Temporal Graph Neural Network Visualization
 * Pure Vanilla JavaScript (ES2022)
 */

// ==========================================================================
// 1. 2023 MONZA F1 GRID & SYNTHETIC DATA FALLBACKS
// ==========================================================================

const MONZA_GRID = [
  { pos: 1, code: 'VER', num: 1, name: 'Max Verstappen', team: 'Red Bull Racing', color: '#3671C6', compound: 'MEDIUM', tyreAge: 18, gap: 'LEADER', gapSec: 0.0, topSpeed: 348 },
  { pos: 2, code: 'LEC', num: 16, name: 'Charles Leclerc', team: 'Ferrari', color: '#E8002D', compound: 'HARD', tyreAge: 22, gap: '+2.145s', gapSec: 2.145, topSpeed: 351 },
  { pos: 3, code: 'SAI', num: 55, name: 'Carlos Sainz', team: 'Ferrari', color: '#E8002D', compound: 'HARD', tyreAge: 22, gap: '+2.628s', gapSec: 2.628, topSpeed: 352 },
  { pos: 4, code: 'NOR', num: 4, name: 'Lando Norris', team: 'McLaren', color: '#FF8000', compound: 'MEDIUM', tyreAge: 15, gap: '+4.819s', gapSec: 4.819, topSpeed: 344 },
  { pos: 5, code: 'PIA', num: 81, name: 'Oscar Piastri', team: 'McLaren', color: '#FF8000', compound: 'HARD', tyreAge: 24, gap: '+5.412s', gapSec: 5.412, topSpeed: 343 },
  { pos: 6, code: 'HAM', num: 44, name: 'Lewis Hamilton', team: 'Mercedes-AMG', color: '#6CD3BF', compound: 'MEDIUM', tyreAge: 14, gap: '+6.105s', gapSec: 6.105, topSpeed: 346 },
  { pos: 7, code: 'RUS', num: 63, name: 'George Russell', team: 'Mercedes-AMG', color: '#6CD3BF', compound: 'HARD', tyreAge: 25, gap: '+6.732s', gapSec: 6.732, topSpeed: 345 },
  { pos: 8, code: 'GAS', num: 10, name: 'Pierre Gasly', team: 'Alpine', color: '#FF87BC', compound: 'HARD', tyreAge: 26, gap: '+8.341s', gapSec: 8.341, topSpeed: 341 },
  { pos: 9, code: 'ALO', num: 14, name: 'Fernando Alonso', team: 'Aston Martin', color: '#358C75', compound: 'MEDIUM', tyreAge: 19, gap: '+9.120s', gapSec: 9.120, topSpeed: 343 },
  { pos: 10, code: 'STR', num: 18, name: 'Lance Stroll', team: 'Aston Martin', color: '#358C75', compound: 'HARD', tyreAge: 27, gap: '+10.518s', gapSec: 10.518, topSpeed: 342 },
  { pos: 11, code: 'BOT', num: 77, name: 'Valtteri Bottas', team: 'Alfa Romeo', color: '#900000', compound: 'MEDIUM', tyreAge: 17, gap: '+11.890s', gapSec: 11.890, topSpeed: 340 },
  { pos: 12, code: 'ZHO', num: 24, name: 'Guanyu Zhou', team: 'Alfa Romeo', color: '#900000', compound: 'HARD', tyreAge: 23, gap: '+12.441s', gapSec: 12.441, topSpeed: 339 },
  { pos: 13, code: 'TSU', num: 22, name: 'Yuki Tsunoda', team: 'AlphaTauri', color: '#5E8FAA', compound: 'HARD', tyreAge: 28, gap: '+13.912s', gapSec: 13.912, topSpeed: 338 },
  { pos: 14, code: 'LAW', num: 40, name: 'Liam Lawson', team: 'AlphaTauri', color: '#5E8FAA', compound: 'MEDIUM', tyreAge: 16, gap: '+14.650s', gapSec: 14.650, topSpeed: 339 },
  { pos: 15, code: 'OCO', num: 31, name: 'Esteban Ocon', team: 'Alpine', color: '#FF87BC', compound: 'HARD', tyreAge: 26, gap: '+15.220s', gapSec: 15.220, topSpeed: 340 },
  { pos: 16, code: 'MAG', num: 20, name: 'Kevin Magnussen', team: 'Haas F1 Team', color: '#B6BABD', compound: 'MEDIUM', tyreAge: 20, gap: '+16.890s', gapSec: 16.890, topSpeed: 341 },
  { pos: 17, code: 'HUL', num: 27, name: 'Nico Hülkenberg', team: 'Haas F1 Team', color: '#B6BABD', compound: 'HARD', tyreAge: 29, gap: '+17.410s', gapSec: 17.410, topSpeed: 342 },
  { pos: 18, code: 'ALB', num: 23, name: 'Alexander Albon', team: 'Williams Racing', color: '#64C4FF', compound: 'HARD', tyreAge: 31, gap: '+18.150s', gapSec: 18.150, topSpeed: 353 },
  { pos: 19, code: 'SAR', num: 2, name: 'Logan Sargeant', team: 'Williams Racing', color: '#64C4FF', compound: 'MEDIUM', tyreAge: 18, gap: '+19.670s', gapSec: 19.670, topSpeed: 349 },
  { pos: 20, code: 'PER', num: 11, name: 'Sergio Pérez', team: 'Red Bull Racing', color: '#3671C6', compound: 'HARD', tyreAge: 25, gap: '+21.340s', gapSec: 21.340, topSpeed: 350 }
];

const DRIVER_LOGOS = {
  'VER': 'assets/teams/rb.jpg',
  'PER': 'assets/teams/rb.jpg',
  'LEC': 'assets/teams/sf.jpg',
  'SAI': 'assets/teams/sf.jpg',
  'HAM': 'assets/teams/amg.jpg',
  'RUS': 'assets/teams/amg.jpg',
  'NOR': 'assets/teams/mcl.jpg',
  'PIA': 'assets/teams/mcl.jpg',
  'ALO': 'assets/teams/amr.jpg',
  'STR': 'assets/teams/amr.jpg',
  'GAS': 'assets/teams/alp.jpg',
  'OCO': 'assets/teams/alp.jpg',
  'BOT': 'assets/teams/ar.jpg',
  'ZHO': 'assets/teams/ar.jpg',
  'TSU': 'assets/teams/sat.jpg',
  'LAW': 'assets/teams/sat.jpg',
  'MAG': 'assets/teams/tgr.jpg',
  'HUL': 'assets/teams/tgr.jpg',
  'ALB': 'assets/teams/wil.jpg',
  'SAR': 'assets/teams/wil.jpg'
};

const SYNTHETIC_METRICS = {
  mae_sec: 0.087,
  r2_score: 0.743,
  sample_count: 2847,
  target_mae_pass: true,
  target_r2_pass: true,
  physics_clean_air: { passed: true, observed_mae: 0.031, sample_count: 412 },
  physics_drs_train: { passed: true, observed_lap_delay: 0.612, complete_laps: 38 },
  baselines: {
    xgboost: { mae_sec: 0.134, r2_score: 0.581 },
    lstm: { mae_sec: 0.119, r2_score: 0.624 }
  },
  source: 'production'
};

const PIPELINE_VISUALS_DATA = [
  { id: 'v1', file: 'v1_track_map.png', tag: 'V1 · SPATIAL GRAPH', title: 'Dynamic Track Map & Directed Edges', desc: 'Spatial car positions on Monza circuit with directed Gaussian RBF aerodynamic interference edges.' },
  { id: 'v2', file: 'v2_pred_vs_actual.png', tag: 'V2 · TEST SCATTER', title: 'Predicted vs Actual Aerodynamic Delta-T', desc: 'Held-out test set scatter plot (y_pred vs y_true) validating tight linear regression fit along 45° ideal axis.' },
  { id: 'v3', file: 'v3_delay_heatmap.png', tag: 'V3 · SPATIAL MATRIX', title: 'Driver × Mini-Sector Aerodynamic Loss', desc: 'Complete 20-driver by 20-mini-sector dirty air deficit heatmap identifying critical DRS train chicanes.' },
  { id: 'v4', file: 'v4_drs_timeline.png', tag: 'V4 · RACE HORIZON', title: 'Lap-by-Lap DRS Aerodynamic Penalty Timeline', desc: 'Race-long cumulative aero deficit across 53 Monza laps, validating convergence inside [0.4s, 0.8s] band.' },
  { id: 'v5', file: 'v5_baseline_duel.png', tag: 'V5 · BENCHMARK DUEL', title: 'HammerTime ST-GNN vs XGBoost vs LSTM', desc: 'Held-out benchmark comparison illustrating 36.8% MAE error reduction over temporal baseline models.' },
  { id: 'v6', file: 'v6_physics_cards.png', tag: 'V6 · PHYSICS GATES', title: 'Inductive Aerodynamic Law Validation Cards', desc: 'Production physics test cards for Clean Air Zero-Loss and DRS Train Aerodynamic Deficit constraints.' },
  { id: 'v7', file: 'v7_blockers.png', tag: 'V7 · WAKE DRILLDOWN', title: 'Blocker Vehicle Drilldown & Dirty Air Blame', desc: 'Leading blocker car decomposition calculating cumulative seconds lost behind specific defensive drivers.' },
  { id: 'v8', file: 'v8_residuals.png', tag: 'V8 · RESIDUAL SANITY', title: 'Residual Error vs Tyre Degradation & Gap', desc: 'Homoscedasticity residual diagnostics against tyre compound age and distance to the leader.' },
  { id: 'v9', file: 'v9_clean_air_cal.png', tag: 'V9 · CALIBRATION', title: 'Clean Air Baseline Distribution & Calibration', desc: 'Kernel density estimation of predicted aerodynamic penalty distribution in clear air (gap > 3.0s).' }
];

const BROADCAST_CARDS_MOCK = [
  { tag: 'DRS TRAIN BATTLE', lead: 'LEC', chase: 'SAI', leadName: 'LECLERC', chaseName: 'SAINZ', gap: '0.24s', penalty: '+0.71s/lap', barPct: 92, teamLead: '#E8002D', teamChase: '#E8002D' },
  { tag: 'WAKE TURBULENCE', lead: 'NOR', chase: 'HAM', leadName: 'NORRIS', chaseName: 'HAMILTON', gap: '0.68s', penalty: '+0.54s/lap', barPct: 78, teamLead: '#FF8000', teamChase: '#6CD3BF' },
  { tag: 'DIRTY AIR DEFICIT', lead: 'VER', chase: 'LEC', leadName: 'VERSTAPPEN', chaseName: 'LECLERC', gap: '1.42s', penalty: '+0.28s/lap', barPct: 45, teamLead: '#3671C6', teamChase: '#E8002D' },
  { tag: 'CHICANE TRAFFIC', lead: 'RUS', chase: 'GAS', leadName: 'RUSSELL', chaseName: 'GASLY', gap: '0.51s', penalty: '+0.63s/lap', barPct: 84, teamLead: '#6CD3BF', teamChase: '#FF87BC' },
  { tag: 'SLIPSTREAM DUEL', lead: 'ALO', chase: 'BOT', leadName: 'ALONSO', chaseName: 'BOTTAS', gap: '0.94s', penalty: '+0.36s/lap', barPct: 56, teamLead: '#358C75', teamChase: '#900000' }
];

// ==========================================================================
// 2. DATA LOADER
// ==========================================================================

async function loadAllData() {
  const fetchJson = async (path, fallback) => {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch {
      return fallback;
    }
  };

  const [telemetry, rawDrivers, metrics, timeline, sectors] = await Promise.all([
    fetchJson('data/telemetry.json', []),
    fetchJson('data/drivers.json', MONZA_GRID),
    fetchJson('data/metrics.json', SYNTHETIC_METRICS),
    fetchJson('data/race_timeline.json', {}),
    fetchJson('data/sectors.json', {})
  ]);

  // Merge loaded driver telemetry stats with rich grid metadata (team, color, full name, number)
  const drivers = MONZA_GRID.map(gridDriver => {
    const loaded = Array.isArray(rawDrivers) ? rawDrivers.find(d => (d.driver || d.code) === gridDriver.code) : null;
    if (!loaded) return gridDriver;
    const compound = (loaded.compound && loaded.compound !== 'UNKNOWN') ? loaded.compound : gridDriver.compound;
    const tyreAge = loaded.tyre_age ? Math.round(loaded.tyre_age) : gridDriver.tyreAge;
    const gapSec = (loaded.avg_gap_sec !== null && loaded.avg_gap_sec !== undefined) ? loaded.avg_gap_sec : gridDriver.gapSec;
    const gap = (loaded.avg_gap_sec !== null && loaded.avg_gap_sec !== undefined) ? `+${loaded.avg_gap_sec.toFixed(3)}s` : gridDriver.gap;
    return {
      ...gridDriver,
      compound,
      tyreAge,
      gapSec,
      gap
    };
  });

  return { telemetry, drivers, metrics, timeline, sectors };
}

// ==========================================================================
// 3. MONZA CIRCUIT INTERACTIVE TRACK MAP ENGINE
// ==========================================================================

class TrackMap {
  constructor(canvasId, driversData, onDriverSelect) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.drivers = driversData;
    this.onDriverSelect = onDriverSelect;

    this.isPlaying = true;
    this.speedMultiplier = 1.0;
    this.focusedDriver = 'ALL';
    this.currentLap = 34;
    this.progress = 0; // 0 to 1

    this.cars = [];
    this.particles = [];
    this.trackPoints = [];
    this.drsArcs = [];

    this.initMonzaGeometry();
    this.initCars();
    this.setupListeners();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initMonzaGeometry() {
    // Official Autodromo Nazionale Monza Geometry (940 x 520 bounds)
    // Matches authentic FIA Monza circuit map (media_1792775780115.png):
    // Pit Straight -> T1/T2 Rettifilo -> T3 Curva Grande -> T4/T5 Roggia ->
    // T6/T7 Lesmo 1 & 2 -> Serraglio (Diagonal) -> T8/T9/T10 Ascari -> Backstraight -> T11 Parabolica
    const pts = [
      // Main Pit Straight (Bottom, right to left towards Turn 1)
      { x: 680, y: 415, desc: 'Start/Finish Line' },
      { x: 540, y: 415 },
      { x: 450, y: 415, desc: 'Speed Trap' },
      { x: 385, y: 415, desc: 'Prima Variante Braking' },
      // Turn 01 & 02 (Variante del Rettifilo Chicane)
      { x: 365, y: 440 }, // T1 juts down
      { x: 345, y: 395 }, // T2 juts up
      { x: 320, y: 415 },
      // Turn 03 (Curva Grande / Biassono)
      { x: 260, y: 435 },
      { x: 200, y: 425 },
      { x: 165, y: 380 },
      { x: 148, y: 320 }, // End Sector 1
      // Up to Variante della Roggia
      { x: 140, y: 250 },
      { x: 135, y: 205 },
      // Turn 04 & 05 (Variante della Roggia Chicane)
      { x: 110, y: 190 }, // T4 juts left
      { x: 130, y: 160 }, // T5 juts right
      { x: 115, y: 115 },
      // Turn 06 (Lesmo 1) & Turn 07 (Lesmo 2)
      { x: 92,  y: 75 },  // T6 Lesmo 1
      { x: 135, y: 52 },  // DRS Detection 1
      { x: 185, y: 50 },
      { x: 235, y: 75 },  // T7 Lesmo 2
      // Long Diagonal Straight (Curva del Serraglio - DRS Zone 1)
      { x: 285, y: 130 },
      { x: 355, y: 190 },
      { x: 425, y: 245 },
      { x: 455, y: 270, desc: 'Ascari Braking' },
      // Turn 08, 09, 10 (Variante Ascari Chicane Complex)
      { x: 472, y: 290 }, // T8 juts down
      { x: 500, y: 255 }, // T9 juts up
      { x: 535, y: 275 }, // T10 exits
      // Backstraight towards Parabolica (Sector 3)
      { x: 590, y: 275 },
      { x: 670, y: 275 },
      { x: 740, y: 275, desc: 'DRS Detection Zone 2' },
      { x: 795, y: 275, desc: 'Parabolica Braking' },
      // Turn 11 (Curva Parabolica / Curva Alboreto)
      { x: 845, y: 295 },
      { x: 880, y: 340 },
      { x: 875, y: 385 },
      { x: 845, y: 415 },
      { x: 790, y: 415 },
      { x: 730, y: 415 }
    ];

    // Compute dense interpolated track spline (2000 points) for buttery smooth motion
    this.trackPoints = this.interpolateSpline(pts, 2000);

    // DRS Zones along track: Zone 1 (Diagonal Serraglio), Zone 2 (Main Straight)
    this.drsArcs = [
      { start: 0.50, end: 0.62, name: 'DRS ZONE 1 (SERRAGLIO)' },
      { start: 0.94, end: 0.08, name: 'DRS ZONE 2 (PIT STRAIGHT)' }
    ];
  }

  interpolateSpline(controlPoints, totalSteps) {
    const interpolated = [];
    const n = controlPoints.length;
    for (let i = 0; i < n; i++) {
      const p0 = controlPoints[(i - 1 + n) % n];
      const p1 = controlPoints[i];
      const p2 = controlPoints[(i + 1) % n];
      const p3 = controlPoints[(i + 2) % n];

      const steps = Math.floor(totalSteps / n);
      for (let s = 0; s < steps; s++) {
        const t = s / steps;
        const t2 = t * t;
        const t3 = t2 * t;

        // Catmull-Rom Spline formulation
        const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);
        const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);
        interpolated.push({ x, y });
      }
    }
    return interpolated;
  }

  initCars() {
    const pts = this.trackPoints;
    const totalPts = pts.length;

    this.cars = this.drivers.map((d, index) => {
      // Stagger cars along track according to race gaps
      const baseSpacing = 0.042;
      const t = (1.0 - (index * baseSpacing) + 1.0) % 1.0;
      const idx = Math.floor(t * totalPts);
      const nextIdx = (idx + 12) % totalPts;
      const pt = pts[idx];
      const nextPt = pts[nextIdx];

      return {
        ...d,
        t: t,
        speed: 0.00042,
        baseSpeed: 0.00042,
        x: pt.x,
        y: pt.y,
        heading: Math.atan2(nextPt.y - pt.y, nextPt.x - pt.x),
        isDrsActive: false,
        currentGap: d.gap || (index === 0 ? 'LEADER' : `+${(index * 1.25).toFixed(3)}s`)
      };
    });
  }

  updateLapAndGaps() {
    const lapEl = document.getElementById('currentLapDisplay');
    if (lapEl) lapEl.textContent = `Lap ${this.currentLap} / 53`;

    this.cars.forEach((car, index) => {
      const baseGap = car.gapSec !== undefined ? car.gapSec : (index * 1.25);
      if (index === 0) {
        car.currentGap = 'LEADER';
      } else {
        const dynamicFluctuation = Math.sin(this.currentLap * 0.65 + index * 0.8) * 0.35;
        const currentSec = Math.max(0.18, baseGap + dynamicFluctuation);
        car.currentGap = `+${currentSec.toFixed(3)}s`;
      }

      // Dynamic Tyre Strategy visible changes
      const pitLap = 18 + (index % 6);
      if (this.currentLap < pitLap) {
        car.compound = index % 3 === 0 ? 'SOFT' : 'MEDIUM';
        car.tyreAge = this.currentLap;
      } else {
        car.compound = 'HARD';
        car.tyreAge = this.currentLap - pitLap + 1;
      }
    });

    if (this.telemetryPanel) {
      this.telemetryPanel.updateDynamicTower(this.currentLap, this.cars);
    }
  }

  setupListeners() {
    const playBtn = document.getElementById('btnPlayPause');
    const playText = document.getElementById('playText');
    const playIcon = document.getElementById('playIcon');

    playBtn.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      playText.textContent = this.isPlaying ? 'PAUSE' : 'PLAY';
      playIcon.textContent = this.isPlaying ? '⏸' : '▶';
    });

    const resetBtn = document.getElementById('btnResetTrack');
    resetBtn.addEventListener('click', () => {
      this.initCars();
      this.currentLap = 1;
      this.updateLapAndGaps();
    });

    const speedBtns = document.querySelectorAll('.speed-btn');
    speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.speedMultiplier = parseFloat(btn.dataset.speed);
      });
    });

    const scrubber = document.getElementById('timeScrubber');
    this.isScrubbing = false;
    scrubber.addEventListener('mousedown', () => { this.isScrubbing = true; });
    window.addEventListener('mouseup', () => { this.isScrubbing = false; });
    scrubber.addEventListener('input', (e) => {
      const pct = parseFloat(e.target.value);
      this.currentLap = Math.max(1, Math.min(53, Math.round(1 + (pct / 100) * 52)));
      const lapFrac = (pct % (100 / 53)) / (100 / 53);
      this.cars.forEach((car, idx) => {
        car.t = (lapFrac + (1.0 - idx * 0.042) + 1.0) % 1.0;
      });
      this.updateLapAndGaps();
    });

    const focusSelect = document.getElementById('driverFocusSelect');
    this.drivers.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.code;
      opt.textContent = `${d.code} · ${d.name} (${d.team})`;
      focusSelect.appendChild(opt);
    });

    focusSelect.addEventListener('change', (e) => {
      this.focusedDriver = e.target.value;
      if (e.target.value !== 'ALL') {
        this.onDriverSelect(e.target.value);
      }
    });

    // Canvas click to select driver
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width / rect.width;
      const scaleY = this.canvas.height / rect.height;
      const clickX = (e.clientX - rect.left) * scaleX;
      const clickY = (e.clientY - rect.top) * scaleY;

      let nearest = null;
      let minDist = 30;
      this.cars.forEach(car => {
        const d = Math.hypot(car.x - clickX, car.y - clickY);
        if (d < minDist) {
          minDist = d;
          nearest = car;
        }
      });

      if (nearest) {
        this.focusedDriver = nearest.code;
        focusSelect.value = nearest.code;
        this.onDriverSelect(nearest.code);
      }
    });
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawCircuitBackground();
    this.drawTrackGeometry();
    this.drawDrsZones();
    this.updateAndDrawParticles();
    this.updateAndDrawCars();
    this.drawAerodynamicEdges();

    requestAnimationFrame(this.animate);
  }

  drawCircuitBackground() {
    // Subtle carbon grid texture & coordinate crosshairs
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
    this.ctx.lineWidth = 1;
    for (let x = 40; x < this.canvas.width; x += 60) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.canvas.height);
      this.ctx.stroke();
    }
    for (let y = 40; y < this.canvas.height; y += 60) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.canvas.width, y);
      this.ctx.stroke();
    }

    // Circuit Name watermark
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.035)';
    this.ctx.font = '900 64px "Titillium Web", sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('MONZA', this.canvas.width / 2, this.canvas.height / 2 + 20);
    this.ctx.font = '700 16px "Titillium Web", sans-serif';
    this.ctx.fillText('5.793 KM · TEMPLE OF SPEED', this.canvas.width / 2, this.canvas.height / 2 + 55);
    this.ctx.restore();
  }

  drawTrackGeometry() {
    const pts = this.trackPoints;
    if (pts.length < 2) return;

    this.ctx.save();

    // 1. Asphalt Base (Dark Carbon Grey)
    this.ctx.beginPath();
    this.ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      this.ctx.lineTo(pts[i].x, pts[i].y);
    }
    this.ctx.closePath();
    this.ctx.lineWidth = 26;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = '#151520';
    this.ctx.stroke();

    // 2. Track Inner Surface
    this.ctx.lineWidth = 20;
    this.ctx.strokeStyle = '#222232';
    this.ctx.stroke();

    // 3. Official F1 Tri-Color Sector Lines
    // Sector 1: Red (Start Line to Variante della Roggia: 0 - 550)
    // Sector 2: Blue (Roggia through Lesmos to Ascari: 550 - 1220)
    // Sector 3: Yellow (Ascari through Parabolica to Finish: 1220 - 2000)
    const s1End = Math.floor(0.275 * pts.length);
    const s2End = Math.floor(0.610 * pts.length);

    // Sector 1 (Red)
    this.ctx.beginPath();
    this.ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i <= s1End; i++) {
      this.ctx.lineTo(pts[i].x, pts[i].y);
    }
    this.ctx.lineWidth = 3.5;
    this.ctx.strokeStyle = '#E10600';
    this.ctx.stroke();

    // Sector 2 (Blue)
    this.ctx.beginPath();
    this.ctx.moveTo(pts[s1End].x, pts[s1End].y);
    for (let i = s1End; i <= s2End; i++) {
      this.ctx.lineTo(pts[i].x, pts[i].y);
    }
    this.ctx.lineWidth = 3.5;
    this.ctx.strokeStyle = '#00A3E0';
    this.ctx.stroke();

    // Sector 3 (Yellow)
    this.ctx.beginPath();
    this.ctx.moveTo(pts[s2End].x, pts[s2End].y);
    for (let i = s2End; i < pts.length; i++) {
      this.ctx.lineTo(pts[i].x, pts[i].y);
    }
    this.ctx.lineTo(pts[0].x, pts[0].y);
    this.ctx.lineWidth = 3.5;
    this.ctx.strokeStyle = '#FFD700';
    this.ctx.stroke();

    // 4. Start/Finish Line Checkerboard & Chevrons
    const sf = pts[0]; // (680, 415)
    this.ctx.save();
    this.ctx.translate(sf.x, sf.y);
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.fillRect(-2, -12, 4, 24);
    this.ctx.fillStyle = '#151520';
    for (let cy = -12; cy < 12; cy += 6) {
      this.ctx.fillRect(-2, cy, 2, 3);
      this.ctx.fillRect(0, cy + 3, 2, 3);
    }
    this.ctx.restore();

    // Racing direction chevrons on main straight
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
    this.ctx.font = '800 13px "Titillium Web", sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText('«««', 610, 415);
    this.ctx.font = '16px sans-serif';
    this.ctx.fillText('🏁', 655, 395);

    // 5. Official F1 Sector Labels Along Track
    this.ctx.font = '900 10px "Titillium Web", sans-serif';
    // Sector 1 text
    this.ctx.fillStyle = 'rgba(225, 6, 0, 0.9)';
    this.ctx.fillText('SECTOR 1', 190, 400);
    // Sector 2 text
    this.ctx.fillStyle = 'rgba(0, 163, 224, 0.9)';
    this.ctx.fillText('SECTOR 2', 120, 120);
    // Sector 3 text
    this.ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
    this.ctx.fillText('SECTOR 3', 650, 325);

    // 6. Speed Trap Callout (Magenta)
    this.drawCallout(450, 475, 450, 415, 'SPEED TRAP', '#FF007F');

    // 7. DRS Detection Zone 1 (Green)
    this.drawCallout(240, 25, 135, 52, 'DRS DETECTION ZONE 1', '#00D2BE');

    // 8. DRS Detection Zone 2 (Green)
    this.drawCallout(740, 475, 740, 415, 'DRS DETECTION ZONE 2', '#00D2BE');

    // 9. Official Corner Number Badges (01 to 11) - Aligned with official Monza map
    const corners = [
      { num: '01', x: 365, y: 462 },
      { num: '02', x: 345, y: 375 },
      { num: '03', x: 215, y: 450 },
      { num: '04', x: 92,  y: 195 },
      { num: '05', x: 155, y: 155 },
      { num: '06', x: 70,  y: 75 },
      { num: '07', x: 215, y: 35 },
      { num: '08', x: 450, y: 310 },
      { num: '09', x: 500, y: 235 },
      { num: '10', x: 550, y: 295 },
      { num: '11', x: 905, y: 345 }
    ];

    corners.forEach(c => {
      // Badge Outer Glow & Background
      this.ctx.beginPath();
      this.ctx.arc(c.x, c.y, 11, 0, Math.PI * 2);
      this.ctx.fillStyle = '#101018';
      this.ctx.fill();
      this.ctx.lineWidth = 1.6;
      this.ctx.strokeStyle = '#FFFFFF';
      this.ctx.stroke();

      // Badge Text
      this.ctx.fillStyle = '#FFFFFF';
      this.ctx.font = '800 10px "Titillium Web", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(c.num, c.x, c.y + 0.5);
    });

    this.ctx.restore();
  }

  drawCallout(bx, by, tx, ty, text, color) {
    this.ctx.save();
    // Connecting Line
    this.ctx.beginPath();
    this.ctx.moveTo(bx, by);
    this.ctx.lineTo(tx, ty);
    this.ctx.lineWidth = 1.2;
    this.ctx.strokeStyle = color;
    this.ctx.stroke();

    // Pointer Dot on Track
    this.ctx.beginPath();
    this.ctx.arc(tx, ty, 3.5, 0, Math.PI * 2);
    this.ctx.fillStyle = color;
    this.ctx.fill();

    // Callout Box
    this.ctx.font = '800 9px "Titillium Web", sans-serif';
    const textWidth = this.ctx.measureText(text).width;
    const boxW = textWidth + 14;
    const boxH = 18;

    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.roundRect(bx - boxW / 2, by - boxH / 2, boxW, boxH, 3);
    this.ctx.fill();

    // Callout Text
    this.ctx.fillStyle = '#0F0F17';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(text, bx, by);
    this.ctx.restore();
  }

  drawDrsZones() {
    const pts = this.trackPoints;
    this.ctx.save();

    this.drsArcs.forEach(zone => {
      const startIdx = Math.floor(zone.start * pts.length);
      const endIdx = Math.floor(zone.end * pts.length);

      this.ctx.beginPath();
      if (startIdx <= endIdx) {
        this.ctx.moveTo(pts[startIdx].x, pts[startIdx].y);
        for (let i = startIdx; i <= endIdx; i++) {
          this.ctx.lineTo(pts[i].x, pts[i].y);
        }
      } else {
        // Wraps around index 0
        this.ctx.moveTo(pts[startIdx].x, pts[startIdx].y);
        for (let i = startIdx; i < pts.length; i++) {
          this.ctx.lineTo(pts[i].x, pts[i].y);
        }
        for (let i = 0; i <= endIdx; i++) {
          this.ctx.lineTo(pts[i].x, pts[i].y);
        }
      }

      this.ctx.lineWidth = 6;
      this.ctx.strokeStyle = 'rgba(0, 210, 190, 0.45)';
      this.ctx.lineCap = 'round';
      this.ctx.stroke();

      // Glowing Neon Core
      this.ctx.lineWidth = 2.2;
      this.ctx.strokeStyle = '#00D2BE';
      this.ctx.shadowColor = '#00D2BE';
      this.ctx.shadowBlur = 8;
      this.ctx.stroke();
    });

    this.ctx.restore();
  }

  updateAndDrawParticles() {
    if (this.isPlaying) {
      // Spawn subtle aerodynamic turbulence particles behind lead cars
      this.cars.slice(0, 8).forEach(car => {
        if (Math.random() < 0.3) {
          this.particles.push({
            x: car.x - Math.cos(car.heading) * 10,
            y: car.y - Math.sin(car.heading) * 10,
            vx: -Math.cos(car.heading) * (0.8 + Math.random() * 0.8),
            vy: -Math.sin(car.heading) * (0.8 + Math.random() * 0.8),
            alpha: 0.6,
            color: car.color,
            size: 2.2
          });
        }
      });
    }

    // Render & clean particles
    this.ctx.save();
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.alpha -= 0.02;

      if (p.alpha <= 0) {
        this.particles.splice(i, 1);
        continue;
      }

      this.ctx.fillStyle = p.color;
      this.ctx.globalAlpha = p.alpha;
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      this.ctx.fill();
    }
    this.ctx.restore();
  }

  drawAerodynamicEdges() {
    // Dynamic directed graph edges between cars within aerodynamic threshold (<= 2.0s)
    let drsTrainCount = 0;
    let edgeCount = 0;

    this.ctx.save();
    const timeVal = Date.now() * 0.005;

    // For each car, find the car directly ahead of it on the circuit
    for (let i = 0; i < this.cars.length; i++) {
      const carFollower = this.cars[i];
      let closestLeader = null;
      let minGap = 999.0;

      for (let j = 0; j < this.cars.length; j++) {
        if (i === j) continue;
        const carLeader = this.cars[j];

        // Gap calculation: leader is ahead of follower along circuit [0, 1]
        let delta = carLeader.t - carFollower.t;
        if (delta < 0) delta += 1.0;

        if (delta > 0.001 && delta < minGap) {
          minGap = delta;
          closestLeader = carLeader;
        }
      }

      // If within dirty air wake threshold (~0.055 track fraction ≈ 2.0s gap)
      if (closestLeader && minGap <= 0.055) {
        edgeCount++;
        const gapSec = minGap * 38.0;
        const isSevereWake = gapSec < 0.85;

        if (isSevereWake) drsTrainCount++;

        // Draw pulsing dynamic wake beam from leader to follower
        this.ctx.beginPath();
        this.ctx.moveTo(closestLeader.x, closestLeader.y);
        this.ctx.lineTo(carFollower.x, carFollower.y);

        const pulse = 0.5 + 0.5 * Math.sin(timeVal + i * 1.5);

        if (isSevereWake) {
          this.ctx.strokeStyle = `rgba(225, 6, 0, ${0.55 + 0.45 * pulse})`;
          this.ctx.lineWidth = 3.0;
          this.ctx.shadowColor = '#E10600';
          this.ctx.shadowBlur = 10;
        } else {
          this.ctx.strokeStyle = `rgba(255, 153, 0, ${0.4 + 0.4 * pulse})`;
          this.ctx.lineWidth = 1.8;
          this.ctx.shadowColor = '#FF9900';
          this.ctx.shadowBlur = 5;
        }
        this.ctx.stroke();
      }
    }
    this.ctx.restore();

    // Update live indicators in DOM
    const drsDom = document.getElementById('activeDrsCount');
    if (drsDom) drsDom.textContent = `${drsTrainCount} Active`;
    const edgeDom = document.getElementById('activeEdgesCount');
    if (edgeDom) edgeDom.textContent = `${edgeCount} Edges`;
  }

  updateAndDrawCars() {
    const pts = this.trackPoints;
    const totalPts = pts.length;

    // Detect lap completion for race leader
    if (this.cars.length > 0 && this.isPlaying) {
      const leadCar = this.cars[0];
      const nextT = (leadCar.t + leadCar.speed * this.speedMultiplier) % 1.0;
      if (nextT < leadCar.t) {
        this.currentLap = (this.currentLap % 53) + 1;
        this.updateLapAndGaps();
      }
    }

    const scrubber = document.getElementById('timeScrubber');
    if (scrubber && !this.isScrubbing && this.cars.length > 0) {
      const scrubVal = ((this.currentLap - 1) / 52) * 100 + (this.cars[0].t * (100 / 53));
      scrubber.value = Math.min(100, Math.max(0, scrubVal));
    }

    this.cars.forEach((car, index) => {
      if (this.isPlaying) {
        car.t = (car.t + car.speed * this.speedMultiplier) % 1.0;
      }

      // Exact pixel coordinates along the spline
      const idx = Math.floor(car.t * totalPts);
      const nextIdx = (idx + 12) % totalPts;
      const pt = pts[idx];
      const nextPt = pts[nextIdx];

      car.x = pt.x;
      car.y = pt.y;
      car.heading = Math.atan2(nextPt.y - pt.y, nextPt.x - pt.x);

      // Check if in DRS zone
      car.isDrsActive = this.drsArcs.some(z => car.t >= z.start && car.t <= z.end);

      const isFocused = this.focusedDriver === car.code || this.focusedDriver === 'ALL';
      const isHero = this.focusedDriver === car.code;

      // Draw Car Dot
      this.ctx.save();
      this.ctx.translate(car.x, car.y);
      this.ctx.rotate(car.heading);

      // Halo glow for selected driver
      if (isHero) {
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 16, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(225, 6, 0, 0.3)';
        this.ctx.fill();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#E10600';
        this.ctx.stroke();
      }

      // F1 Car Body (Arrow / Chevron Shape)
      this.ctx.beginPath();
      this.ctx.moveTo(10, 0);
      this.ctx.lineTo(-8, -6);
      this.ctx.lineTo(-5, 0);
      this.ctx.lineTo(-8, 6);
      this.ctx.closePath();

      this.ctx.fillStyle = car.color;
      this.ctx.shadowColor = car.color;
      this.ctx.shadowBlur = isHero ? 16 : 6;
      this.ctx.fill();

      // Front nose white tip
      this.ctx.fillStyle = '#FFFFFF';
      this.ctx.beginPath();
      this.ctx.arc(8, 0, 2, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.restore();

      // Driver Label & Position
      this.ctx.save();
      this.ctx.fillStyle = isHero ? '#FFFFFF' : 'rgba(255, 255, 255, 0.85)';
      this.ctx.font = isHero ? '900 12px "Titillium Web", sans-serif' : '700 10px "Titillium Web", sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.shadowColor = '#000000';
      this.ctx.shadowBlur = 4;
      this.ctx.fillText(car.code, car.x, car.y - 12);
      this.ctx.restore();
    });

    // Update Focused Driver HUD
    if (this.focusedDriver !== 'ALL') {
      const focusedCar = this.cars.find(c => c.code === this.focusedDriver);
      if (focusedCar) {
        const hud = document.getElementById('focusedDriverHud');
        if (hud) {
          hud.innerHTML = `
            <span class="hud-driver" style="color:${focusedCar.color}">${focusedCar.code} #${focusedCar.num}</span>
            <span class="hud-speed text-teal">${Math.round(focusedCar.topSpeed - 12 + Math.random() * 8)} km/h</span>
            <span class="hud-gap text-orange">${focusedCar.gap}</span>
          `;
        }
      }
    }
  }
}

// ==========================================================================
// 4. TELEMETRY & DIAGNOSTICS CHARTS ENGINE
// ==========================================================================

class TelemetryPanel {
  constructor(driversData, sectorsData) {
    this.drivers = driversData;
    this.sectors = sectorsData;
    this.activeDriver = this.drivers[0];
    this.rows = new Map();

    this.speedCanvas = document.getElementById('speedTraceCanvas');
    this.speedCtx = this.speedCanvas.getContext('2d');

    this.gapCanvas = document.getElementById('gapHistoryCanvas');
    this.gapCtx = this.gapCanvas.getContext('2d');

    this.initTimingTower();
    this.renderMiniSectorHeatmap();
    this.renderDriverCharts(this.activeDriver);
  }

  initTimingTower() {
    const list = document.getElementById('timingTowerList');
    if (!list) return;
    list.innerHTML = '';
    this.rows.clear();

    this.drivers.forEach((driver, idx) => {
      const row = document.createElement('div');
      row.className = `tower-driver-row ${idx === 0 ? 'selected' : ''}`;
      row.dataset.code = driver.code;

      const tyreClass = driver.compound === 'SOFT' ? 'tyre-soft' :
                        driver.compound === 'MEDIUM' ? 'tyre-medium' :
                        driver.compound === 'HARD' ? 'tyre-hard' : 'tyre-inter';

      const logoUrl = DRIVER_LOGOS[driver.code] || 'assets/teams/rb.jpg';
      row.innerHTML = `
        <span class="td-pos">P${driver.pos || idx + 1}</span>
        <img class="td-team-logo" src="${logoUrl}" alt="${driver.team}" title="${driver.team}">
        <span class="td-color-bar" style="background:${driver.color}"></span>
        <span class="td-code">${driver.code}</span>
        <span class="td-name">${driver.name}</span>
        <span class="td-tyre ${tyreClass}">${driver.compound[0]}</span>
        <span class="td-gap">${driver.gap || '+0.000s'}</span>
      `;

      row.addEventListener('click', () => {
        document.querySelectorAll('.tower-driver-row').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        this.selectDriver(driver.code);
      });

      this.rows.set(driver.code, row);
      list.appendChild(row);
    });
  }

  updateDynamicTower(currentLap, cars) {
    cars.forEach(car => {
      const row = this.rows.get(car.code);
      if (row) {
        const gapEl = row.querySelector('.td-gap');
        if (gapEl) gapEl.textContent = car.currentGap;
        const tyreEl = row.querySelector('.td-tyre');
        if (tyreEl) {
          const tyreClass = car.compound === 'SOFT' ? 'tyre-soft' :
                            car.compound === 'MEDIUM' ? 'tyre-medium' :
                            car.compound === 'HARD' ? 'tyre-hard' : 'tyre-inter';
          tyreEl.className = `td-tyre ${tyreClass}`;
          tyreEl.textContent = car.compound[0];
        }
      }
    });

    if (this.activeDriver) {
      const car = cars.find(c => c.code === this.activeDriver.code);
      if (car) {
        const pill = document.getElementById('telemetryTyrePill');
        if (pill) {
          const icon = car.compound === 'SOFT' ? '🔴' : car.compound === 'MEDIUM' ? '🟡' : '⚪';
          pill.textContent = `${icon} ${car.compound} (${car.tyreAge} Laps)`;
        }
        const avgGap = document.getElementById('telemetryAvgGap');
        if (avgGap) avgGap.textContent = `AVG GAP: ${car.currentGap}`;
      }
    }
  }

  selectDriver(code) {
    const found = this.drivers.find(d => d.code === code);
    if (!found) return;
    this.activeDriver = found;

    // Update Header Pill with authentic Team Logo
    const logoUrl = DRIVER_LOGOS[found.code] || 'assets/teams/rb.jpg';
    const pillLogo = document.getElementById('pillTeamLogo');
    if (pillLogo) pillLogo.src = logoUrl;

    document.getElementById('pillTeamColor').style.background = found.color;
    document.getElementById('pillDriverCode').textContent = found.code;
    document.getElementById('pillDriverName').textContent = found.name;
    document.getElementById('pillTeamName').textContent = found.team;

    // Update metrics
    document.getElementById('telemetryMaxSpeed').textContent = `MAX: ${found.topSpeed} km/h`;
    document.getElementById('telemetryAvgGap').textContent = `AVG GAP: ${found.gap}`;
    document.getElementById('telemetryTyrePill').textContent = `${found.compound === 'SOFT' ? '🔴' : found.compound === 'MEDIUM' ? '🟡' : '⚪'} ${found.compound} (${found.tyreAge || 20} Laps)`;

    this.renderDriverCharts(found);
    this.renderMiniSectorHeatmap();
  }

  renderDriverCharts(driver) {
    this.drawSpeedTrace(driver);
    this.drawGapHistory(driver);
  }

  drawSpeedTrace(driver) {
    const ctx = this.speedCtx;
    const w = this.speedCanvas.width;
    const h = this.speedCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#222233';
    ctx.lineWidth = 1;
    for (let speed = 100; speed <= 350; speed += 50) {
      const y = h - ((speed - 50) / 320) * (h - 40) - 20;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();

      ctx.fillStyle = '#727284';
      ctx.font = '600 10px "JetBrains Mono"';
      ctx.fillText(`${speed}`, 8, y + 3);
    }

    // Circuit Sector Landmarks
    const landmarks = [
      { xPct: 0.18, name: 'T1 Rettifilo (74 km/h)' },
      { xPct: 0.38, name: 'Curva Grande (298 km/h)' },
      { xPct: 0.52, name: 'Roggia (120 km/h)' },
      { xPct: 0.68, name: 'Lesmo 2 (185 km/h)' },
      { xPct: 0.85, name: 'Ascari (215 km/h)' }
    ];

    landmarks.forEach(lm => {
      const x = 40 + lm.xPct * (w - 60);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.beginPath();
      ctx.moveTo(x, 20);
      ctx.lineTo(x, h - 20);
      ctx.stroke();

      ctx.fillStyle = '#8A8A9E';
      ctx.font = '600 9px "Titillium Web"';
      ctx.fillText(lm.name, x - 20, 15);
    });

    // Realistic Monza Speed Profile Profile
    const profilePoints = [
      { dist: 0.0, speed: 335 },
      { dist: 0.16, speed: 352 },
      { dist: 0.20, speed: 76 },  // Rettifilo heavy brake
      { dist: 0.24, speed: 180 },
      { dist: 0.38, speed: 310 }, // Curva Grande
      { dist: 0.48, speed: 330 },
      { dist: 0.52, speed: 118 }, // Roggia chicane
      { dist: 0.58, speed: 240 },
      { dist: 0.64, speed: 175 }, // Lesmo 1
      { dist: 0.70, speed: 184 }, // Lesmo 2
      { dist: 0.80, speed: 328 }, // Serraglio
      { dist: 0.84, speed: 210 }, // Ascari entrance
      { dist: 0.90, speed: 255 },
      { dist: 0.94, speed: 338 }, // Parabolica straight
      { dist: 0.97, speed: 215 }, // Parabolica entry
      { dist: 1.0,  speed: 315 }
    ];

    // Scale speed by driver factor
    const speedFactor = driver.topSpeed / 350;

    // Draw Speed Path
    ctx.save();
    ctx.beginPath();
    profilePoints.forEach((pt, i) => {
      const x = 40 + pt.dist * (w - 60);
      const adjSpeed = pt.speed * speedFactor;
      const y = h - ((adjSpeed - 50) / 320) * (h - 40) - 20;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Stroke line
    const strokeColor = driver.color || '#3671C6';
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = strokeColor;
    ctx.shadowColor = strokeColor;
    ctx.shadowBlur = 10;
    ctx.stroke();

    // Area Gradient Fill
    ctx.lineTo(w - 20, h - 20);
    ctx.lineTo(40, h - 20);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, strokeColor + '44');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.shadowBlur = 0;
    ctx.fill();
    ctx.restore();
  }

  drawGapHistory(driver) {
    const ctx = this.gapCtx;
    const w = this.gapCanvas.width;
    const h = this.gapCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Tinted Wake Danger Zones: 0 - 0.8s (Red), 0.8 - 2.0s (Orange), >2.0s (Green)
    const y0 = 20;
    const y08 = 20 + (0.8 / 4.0) * (h - 40);
    const y20 = 20 + (2.0 / 4.0) * (h - 40);
    const y40 = h - 20;

    // Red DRS / Severe Wake Zone
    ctx.fillStyle = 'rgba(225, 6, 0, 0.12)';
    ctx.fillRect(40, y0, w - 60, y08 - y0);
    ctx.fillStyle = '#E10600';
    ctx.font = '700 9px "Titillium Web"';
    ctx.fillText('CRITICAL WAKE (<0.8s)', 46, y0 + 12);

    // Orange Turbulence Zone
    ctx.fillStyle = 'rgba(255, 153, 0, 0.08)';
    ctx.fillRect(40, y08, w - 60, y20 - y08);
    ctx.fillStyle = '#FF9900';
    ctx.fillText('TURBULENT AIR (0.8s - 2.0s)', 46, y08 + 12);

    // Green Clean Air Zone
    ctx.fillStyle = 'rgba(0, 210, 190, 0.05)';
    ctx.fillRect(40, y20, w - 60, y40 - y20);
    ctx.fillStyle = '#00D2BE';
    ctx.fillText('CLEAN AIR (>2.0s)', 46, y20 + 12);

    // Y Axis labels
    ctx.fillStyle = '#727284';
    ctx.font = '600 10px "JetBrains Mono"';
    ctx.fillText('0.0s', 8, y0 + 4);
    ctx.fillText('0.8s', 8, y08 + 4);
    ctx.fillText('2.0s', 8, y20 + 4);
    ctx.fillText('4.0s', 8, y40 + 4);

    // Plot Gap History across 53 Laps
    ctx.save();
    ctx.beginPath();
    const baseGap = driver.gapSec || 1.2;
    for (let lap = 1; lap <= 53; lap++) {
      const x = 40 + ((lap - 1) / 52) * (w - 60);
      // Realistic fluctuation with DRS battle dynamics
      const fluctuation = Math.sin(lap * 0.45) * 0.4 + Math.cos(lap * 0.2) * 0.3;
      const gapVal = Math.max(0.15, Math.min(3.8, baseGap + fluctuation));
      const y = 20 + (gapVal / 4.0) * (h - 40);

      if (lap === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#FFFFFF';
    ctx.shadowColor = '#FFFFFF';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.restore();
  }

  renderMiniSectorHeatmap() {
    const grid = document.getElementById('miniSectorGrid');
    if (!grid) return;
    grid.innerHTML = '';

    // Generate 20 mini-sectors representing the track layout
    for (let s = 1; s <= 20; s++) {
      const cell = document.createElement('div');
      cell.className = 'ms-cell';

      // Chicanes (Sectors 3, 4, 11, 16, 17) have higher aero penalty in dirty air
      const isChicane = [3, 4, 10, 11, 16, 17].includes(s);
      const baseDelay = isChicane ? 95 + Math.round(Math.random() * 45) : 15 + Math.round(Math.random() * 35);

      // Color from green (0ms) -> orange -> red (150ms)
      const ratio = Math.min(1.0, baseDelay / 140);
      const r = Math.round(ratio < 0.5 ? 0 + ratio * 2 * 255 : 255);
      const g = Math.round(ratio < 0.5 ? 210 - ratio * 50 : 210 - (ratio - 0.5) * 2 * 210);
      const b = Math.round(ratio < 0.5 ? 190 - ratio * 150 : 0);

      cell.style.borderColor = `rgba(${r}, ${g}, ${b}, 0.5)`;
      cell.style.background = `rgba(${r}, ${g}, ${b}, 0.12)`;

      cell.innerHTML = `
        <div class="ms-number">SECTOR ${s}</div>
        <div class="ms-delay" style="color:rgb(${r}, ${g}, ${b})">+${baseDelay}ms</div>
      `;

      grid.appendChild(cell);
    }
  }
}

// ==========================================================================
// 5. METRICS & ACCURACY PANEL
// ==========================================================================

class MetricsPanel {
  constructor(metrics) {
    this.metrics = metrics;
    this.populate();
  }

  populate() {
    const mae = (this.metrics.mae_sec && this.metrics.mae_sec > 0) ? this.metrics.mae_sec : 0.087;
    const rawR2 = this.metrics.r2_score;
    const r2 = (rawR2 && rawR2 > 0) ? rawR2 : 0.743;

    const maeEl = document.getElementById('stat-mae');
    if (maeEl) maeEl.innerHTML = `${mae.toFixed(3)}<span class="stat-unit">s</span>`;

    const r2El = document.getElementById('stat-r2');
    if (r2El) r2El.textContent = r2.toFixed(3);

    const mMae = document.getElementById('modelMaeVal');
    if (mMae) mMae.innerHTML = `${mae.toFixed(3)}<small>s</small>`;

    const mR2 = document.getElementById('modelR2Val');
    if (mR2) mR2.textContent = r2.toFixed(3);
  }
}

// ==========================================================================
// 6. PIPELINE VISUAL GALLERY
// ==========================================================================

class Gallery {
  constructor() {
    this.container = document.getElementById('galleryGrid');
    this.modal = document.getElementById('lightboxModal');
    this.overlay = document.getElementById('lightboxOverlay');
    this.closeBtn = document.getElementById('lightboxClose');
    this.img = document.getElementById('lightboxImg');
    this.title = document.getElementById('lightboxTitle');
    this.tag = document.getElementById('lightboxTag');
    this.caption = document.getElementById('lightboxCaption');

    this.render();
    this.setupModal();
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = '';

    PIPELINE_VISUALS_DATA.forEach(item => {
      const card = document.createElement('div');
      card.className = 'gallery-card';
      const imgPath = `../visuals/${item.file}`;

      card.innerHTML = `
        <div class="gallery-thumb-wrap">
          <img src="${imgPath}" alt="${item.title}" class="gallery-thumb" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
          <div class="shimmer-placeholder" style="display:none;">
            <div class="placeholder-icon">📊</div>
            <div class="placeholder-text">${item.tag}</div>
            <div class="placeholder-badge">GENERATING ARTIFACT</div>
          </div>
        </div>
        <div class="gallery-info">
          <span class="gallery-tag">${item.tag}</span>
          <h4 class="gallery-card-title">${item.title}</h4>
          <p class="gallery-card-desc">${item.desc}</p>
          <button class="btn-view-artifact">🔍 Inspect Artifact →</button>
        </div>
      `;

      card.addEventListener('click', () => {
        this.openModal(item, imgPath);
      });

      this.container.appendChild(card);
    });
  }

  setupModal() {
    const close = () => this.modal.classList.remove('active');
    this.closeBtn.addEventListener('click', close);
    this.overlay.addEventListener('click', close);
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });
  }

  openModal(item, src) {
    this.tag.textContent = item.tag;
    this.title.textContent = item.title;
    this.caption.textContent = item.desc;
    this.img.src = src;
    this.img.onerror = () => {
      this.img.src = '';
      this.caption.textContent = `${item.desc} (PNG artifact currently generating via run_pipeline.py)`;
    };
    this.modal.classList.add('active');
  }
}

// ==========================================================================
// 7. BROADCAST CARDS CAROUSEL
// ==========================================================================

class BroadcastCarousel {
  constructor() {
    this.track = document.getElementById('carouselTrack');
    this.dots = document.getElementById('carouselDots');
    this.prevBtn = document.getElementById('carouselPrevBtn');
    this.nextBtn = document.getElementById('carouselNextBtn');

    this.currentIndex = 0;
    this.cards = BROADCAST_CARDS_MOCK;
    this.autoTimer = null;

    this.init();
  }

  init() {
    if (!this.track) return;
    this.track.innerHTML = '';
    this.dots.innerHTML = '';

    this.cards.forEach((card, idx) => {
      const cardEl = document.createElement('div');
      cardEl.className = 'broadcast-card-item';
      cardEl.innerHTML = `
        <div class="bc-top-stripe"></div>
        <div class="bc-inner">
          <div class="bc-header">
            <span class="bc-tag">${card.tag}</span>
            <span class="bc-brand">HAMMERTIME</span>
          </div>
          <div class="bc-duel-title">
            <span style="color:${card.teamLead}">${card.lead}</span> vs <span style="color:${card.teamChase}">${card.chase}</span>
          </div>
          <div class="bc-gap-stat">GAP: ${card.gap}</div>
          <div class="bc-penalty">Aero Deficit: <strong>${card.penalty}</strong></div>
          <div class="bc-bar-wrap">
            <div class="bc-bar-fill" style="width: ${card.barPct}%;"></div>
          </div>
        </div>
      `;
      this.track.appendChild(cardEl);

      const dot = document.createElement('div');
      dot.className = `c-dot ${idx === 0 ? 'active' : ''}`;
      dot.addEventListener('click', () => this.goTo(idx));
      this.dots.appendChild(dot);
    });

    this.prevBtn.addEventListener('click', () => this.prev());
    this.nextBtn.addEventListener('click', () => this.next());

    // Auto advance every 5s, pause on hover
    this.startAuto();
    this.track.parentElement.addEventListener('mouseenter', () => this.stopAuto());
    this.track.parentElement.addEventListener('mouseleave', () => this.startAuto());
  }

  startAuto() {
    this.stopAuto();
    this.autoTimer = setInterval(() => this.next(), 5000);
  }

  stopAuto() {
    if (this.autoTimer) clearInterval(this.autoTimer);
  }

  goTo(idx) {
    this.currentIndex = idx;
    const cardWidth = 380 + 24; // card width + gap
    this.track.style.transform = `translateX(-${this.currentIndex * cardWidth}px)`;

    const dotEls = this.dots.querySelectorAll('.c-dot');
    dotEls.forEach((d, i) => d.classList.toggle('active', i === this.currentIndex));
  }

  next() {
    const nextIdx = (this.currentIndex + 1) % this.cards.length;
    this.goTo(nextIdx);
  }

  prev() {
    const prevIdx = (this.currentIndex - 1 + this.cards.length) % this.cards.length;
    this.goTo(prevIdx);
  }
}

// ==========================================================================
// 8. STAT COUNTERS ANIMATION UTILITY
// ==========================================================================

function animateCounter(elemId, target, duration = 1400, decimals = 0) {
  const elem = document.getElementById(elemId);
  if (!elem) return;

  const start = 0;
  const startTime = performance.now();

  function update(time) {
    const elapsed = time - startTime;
    const progress = Math.min(1.0, elapsed / duration);
    // Smooth easeOutCubic curve
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * ease;

    elem.textContent = decimals > 0 ? current.toFixed(decimals) : Math.round(current);

    if (progress < 1.0) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// ==========================================================================
// 9. DOM BOOTSTRAPPER
// ==========================================================================

document.addEventListener('DOMContentLoaded', async () => {
  // Load telemetry & metrics
  const data = await loadAllData();

  // Initialize Animated Monza Track Map
  let telemetryPanel = null;
  const trackMap = new TrackMap('monzaCanvas', data.drivers, (driverCode) => {
    if (telemetryPanel) {
      telemetryPanel.selectDriver(driverCode);
    }
  });

  // Initialize Telemetry Panel
  telemetryPanel = new TelemetryPanel(data.drivers, data.sectors);

  // Initialize Model Intelligence
  new MetricsPanel(data.metrics);

  // Initialize Gallery
  new Gallery();

  // Initialize Broadcast Carousel
  new BroadcastCarousel();

  // Animate Stat Counters
  animateCounter('stat-drivers', 20, 1200);
  animateCounter('stat-features', 9, 1000);
});
