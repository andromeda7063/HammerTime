import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SCRIPT_DIR, "../..");
const SKILL_DIR = "/home/andromeda/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations";
const RUNTIME_NODE_MODULES = "/home/andromeda/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_NODE_MODULES ||= RUNTIME_NODE_MODULES;
const { importRuntimeModule } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/runtime_helpers.mjs")).href);
const { Presentation, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const BUILD = path.join(ROOT, ".local/research-history/2026-11-02/presentation-build");
const OUTPUT = path.join(SCRIPT_DIR, "HammerTime_Project_Overview_2026_Registry.pptx");
const PREVIEW = path.join(BUILD, "slides");
const W = 1280;
const H = 720;
const C = {
  ink: "#0B1014",
  ink2: "#151C21",
  paper: "#F3F2ED",
  white: "#F5F5F2",
  text: "#20272B",
  muted: "#677177",
  mutedDark: "#A5B0B4",
  rule: "#D4D8D6",
  red: "#E10600",
  redSoft: "#FF5147",
  teal: "#00D2BE",
  amber: "#E4AE47",
  green: "#21966E",
  fail: "#B83F38",
};
const FONT = "Fira Sans";
const MONO = "JetBrains Mono";
const { applyPresentationChartFont } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function shape(slide, geometry, position, fill = "none", line = { style: "solid", fill: "none", width: 0 }, name = undefined) {
  return slide.shapes.add({ geometry, name, position, fill, line });
}

function text(slide, value, position, options = {}) {
  const box = shape(slide, "textbox", position, "none", { style: "solid", fill: "none", width: 0 }, options.name);
  box.text = value;
  box.text.style = {
    typeface: options.typeface ?? FONT,
    fontSize: options.size ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.text,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "top",
    lineSpacing: options.leading ?? 1.05,
    autoFit: "shrinkText",
    wrap: options.wrap ?? "square",
    insets: 0,
  };
  return box;
}

function title(slide, heading, deckLabel = "HAMMERTIME  /  RESEARCH OVERVIEW", dark = false) {
  text(slide, deckLabel, { left: 72, top: 30, width: 800, height: 22 }, {
    size: 15, typeface: MONO, color: dark ? C.mutedDark : C.muted, bold: true, name: "section-label",
  });
  text(slide, heading, { left: 72, top: 70, width: 1125, height: 70 }, {
    size: 45, bold: true, color: dark ? C.white : C.text, name: "slide-title",
  });
}

function footer(slide, page, dark = false, context = "FASTF1 2023  /  SAVED EVALUATION  /  NOV 2026") {
  const color = dark ? "#718086" : "#8A9396";
  text(slide, context, { left: 72, top: 675, width: 860, height: 20 }, {
    size: 12, typeface: MONO, color, name: "footer-source-context",
  });
  text(slide, String(page).padStart(2, "0"), { left: 1142, top: 672, width: 64, height: 24 }, {
    size: 15, typeface: MONO, color, align: "right", name: "page-number",
  });
}

function newSlide(presentation, dark = false) {
  const slide = presentation.slides.add();
  slide.background.fill = dark ? C.ink : C.paper;
  return slide;
}

function note(slide, lines) {
  slide.speakerNotes.text = lines.join("\n");
}

function styleTable(table, rows, cols, fontSize = 21) {
  table.borders.assign({ style: "solid", fill: C.rule, width: 1 });
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? C.ink2 : (r % 2 === 0 ? "#E9ECE8" : C.paper);
      cell.text.style = {
        typeface: r === 0 ? MONO : FONT,
        fontSize: r === 0 ? 17 : fontSize,
        bold: r === 0 || c === 0,
        color: r === 0 ? C.white : C.text,
        insets: { top: 8, right: 10, bottom: 8, left: 10 },
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
}

async function addDashboardImage(slide, imagePath, position) {
  const bytes = await fs.readFile(imagePath);
  const arrayBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  slide.images.add({
    blob: arrayBuffer,
    contentType: "image/png",
    alt: "HammerTime archived FastF1 season dashboard preview showing saved evaluation metrics, split counts, and prediction explorer",
    fit: "contain",
    position,
  });
}

async function main() {
  await fs.mkdir(BUILD, { recursive: true });
  await fs.mkdir(PREVIEW, { recursive: true });
  const seasonData = JSON.parse(await fs.readFile(path.join(ROOT, "hammertime_100/frontend/data/season_model.json"), "utf8"));
  const registryData = JSON.parse(await fs.readFile(path.join(ROOT, "hammertime_100/frontend/data/experiments.json"), "utf8"));
  if (seasonData.run_id !== "769a76aab164b514cc0a") throw new Error("Unexpected current season run ID in tracked season_model.json");
  const seasonMetrics = seasonData.metrics;
  const zeroMetrics = seasonMetrics.zero_baseline;
  const raceMetrics = seasonData.race_metrics;
  const physicsCleanAir = seasonMetrics.physics_tests.find(item => item.test.startsWith("Clean Air"));
  const physicsDrs = seasonMetrics.physics_tests.find(item => item.test.startsWith("DRS Train"));
  if (raceMetrics.length !== 4 || !physicsCleanAir || !physicsDrs) throw new Error("Tracked season JSON is missing race or physics evidence");
  const hungary = registryData.groups.find(item => item.id === "openf1-hungary-circuit-screen");
  const stgnnRegistryRuns = registryData.groups.flatMap(item => item.runs).filter(item => item.family === "ST-GNN");
  const scoredRegistryCount = registryData.groups.flatMap(item => item.runs).filter(item => item.mae_sec !== null).length;
  const noScoreRegistryCount = registryData.groups.flatMap(item => item.runs).filter(item => item.mae_sec === null).length;
  const stgnnScoredCount = stgnnRegistryRuns.filter(item => item.mae_sec !== null).length;
  const stgnnNoScoreCount = stgnnRegistryRuns.length - stgnnScoredCount;
  if (!hungary || registryData.groups.length !== 9 || scoredRegistryCount !== 21 || noScoreRegistryCount !== 2 || stgnnRegistryRuns.length !== 14) {
    throw new Error("Unexpected experiment registry counts in tracked experiments.json");
  }
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Cover
  {
    const slide = newSlide(presentation, true);
    text(slide, "HAMMERTIME", { left: 72, top: 70, width: 840, height: 90 }, {
      size: 72, bold: true, color: C.white, name: "cover-title",
    });
    text(slide, "Race telemetry and\ngraph learning", { left: 76, top: 213, width: 890, height: 150 }, {
      size: 39, color: C.white, leading: 1.06, name: "cover-subtitle",
    });
    text(slide, "Research prototype  /  2023 FastF1 season evaluation", { left: 76, top: 400, width: 840, height: 32 }, {
      size: 21, typeface: MONO, color: C.teal, name: "cover-context",
    });
    text(slide, "A saved ST-GNN evaluation with race-level holdout, a matched zero predictor, and explicit limits on what the result shows.", { left: 76, top: 475, width: 810, height: 74 }, {
      size: 23, color: C.mutedDark, leading: 1.2, name: "cover-description",
    });
    text(slide, "EVIDENCE THROUGH 2026-11-02", { left: 76, top: 625, width: 620, height: 24 }, {
      size: 14, typeface: MONO, color: "#718086", name: "cover-date",
    });
    note(slide, [
      "Sources: README.md; EXECUTIVE_ONE_PAGER.md; hammertime_100/frontend/data/season_model.json.",
      "Run identifier: 769a76aab164b514cc0a. This is a project research prototype, not a production race-strategy system.",
    ]);
  }

  // 2. Target definition
  {
    const slide = newSlide(presentation);
    title(slide, "The model estimates a calibrated sector residual");
    text(slide, "OBSERVED", { left: 80, top: 204, width: 300, height: 24 }, {
      size: 16, typeface: MONO, color: C.muted, bold: true,
    });
    text(slide, "Mini-sector\nduration", { left: 80, top: 241, width: 285, height: 90 }, {
      size: 37, bold: true, color: C.text,
    });
    text(slide, "−", { left: 385, top: 245, width: 52, height: 64 }, {
      size: 45, color: C.red, align: "center",
    });
    text(slide, "REFERENCE", { left: 468, top: 204, width: 340, height: 24 }, {
      size: 16, typeface: MONO, color: C.muted, bold: true,
    });
    text(slide, "Same-session\nclean-air pace", { left: 468, top: 241, width: 340, height: 90 }, {
      size: 37, bold: true, color: C.text,
    });
    text(slide, "=", { left: 825, top: 245, width: 52, height: 64 }, {
      size: 45, color: C.red, align: "center",
    });
    text(slide, "PREDICT", { left: 900, top: 204, width: 300, height: 24 }, {
      size: 16, typeface: MONO, color: C.muted, bold: true,
    });
    text(slide, "Signed\nresidual", { left: 900, top: 241, width: 300, height: 90 }, {
      size: 37, bold: true, color: C.text,
    });
    text(slide, "Positive and negative labels are possible. The reference is calibrated within each race because sector times do not transfer across circuits.", { left: 82, top: 402, width: 1055, height: 60 }, {
      size: 24, color: C.text, leading: 1.18,
    });
    text(slide, "Interpretation limit", { left: 82, top: 515, width: 230, height: 28 }, {
      size: 18, typeface: MONO, color: C.red, bold: true,
    });
    text(slide, "The current target uses observed clean sectors from the full session, including future held-out observations. This is retrospective session calibration, not a strict prospective forecast target.", { left: 82, top: 551, width: 1080, height: 68 }, {
      size: 22, color: C.muted, leading: 1.18,
    });
    footer(slide, 2);
    note(slide, [
      "Sources: hammertime_100/REPORT.md, Data and model flow; ASSUMPTIONS.md, A03 and A11; OPEN_QUESTIONS.md, item 2.",
      "The archived season run labels the target as an existing full-session-calibrated residual. The report distinguishes session-scoped label calibration from train-only model feature scaling.",
    ]);
  }

  // 3. Representation and model flow
  {
    const slide = newSlide(presentation, true);
    title(slide, "Race telemetry becomes a time-varying graph", "HAMMERTIME  /  MODEL FLOW", true);
    text(slide, "Each time step aligns observed car states. A directed traffic graph carries proximity context into a temporal graph model.", { left: 75, top: 153, width: 1100, height: 60 }, {
      size: 23, color: C.mutedDark, leading: 1.2,
    });
    const y = 284;
    const boxes = [
      { x: 80, w: 290, head: "RACE OBSERVATIONS", body: "Speed, throttle,\ntyre state, track progress" },
      { x: 495, w: 290, head: "TRAFFIC GRAPH", body: "Cars as nodes\nWake proximity as edges" },
      { x: 910, w: 290, head: "ST-GNN OUTPUT", body: "Predicted sector\nresidual at a future step" },
    ];
    for (const [i, item] of boxes.entries()) {
      const box = shape(slide, "rect", { left: item.x, top: y, width: item.w, height: 196 }, C.ink2, { style: "solid", fill: i === 2 ? C.red : "#364148", width: 1 }, `flow-node-${i + 1}`);
      text(slide, item.head, { left: item.x + 22, top: y + 26, width: item.w - 44, height: 26 }, {
        size: 15, typeface: MONO, bold: true, color: i === 2 ? C.redSoft : C.teal,
      });
      text(slide, item.body, { left: item.x + 22, top: y + 69, width: item.w - 44, height: 100 }, {
        size: 25, bold: true, color: C.white, leading: 1.1,
      });
      if (i < boxes.length - 1) {
        shape(slide, "rightArrow", { left: item.x + item.w + 24, top: y + 76, width: 78, height: 42 }, C.red, { style: "solid", fill: "none", width: 0 }, `flow-arrow-${i + 1}`);
      }
      void box;
    }
    text(slide, "The audited implementation masks pit, invalid, and safety-car-contaminated samples. Those rules describe data eligibility, not measured aerodynamic effects.", { left: 82, top: 548, width: 1090, height: 62 }, {
      size: 21, color: C.mutedDark, leading: 1.18,
    });
    footer(slide, 3, true);
    note(slide, [
      "Sources: EXECUTIVE_ONE_PAGER.md; hammertime_100/REPORT.md, Data and model flow; hammertime_100/src/graph.py; hammertime_100/src/dataset.py.",
      "This slide describes the pipeline concept. It does not assert that a graph edge is a measured aerodynamic wake effect.",
    ]);
  }

  // 4. Split design
  {
    const slide = newSlide(presentation);
    title(slide, "The 2023 evaluation holds out whole races");
    const cols = [
      { x: 90, n: "15", label: "training races", note: "model fit" },
      { x: 465, n: "3", label: "validation races", note: "checkpoint selection" },
      { x: 845, n: "4", label: "test races", note: "held out by race" },
    ];
    for (const col of cols) {
      text(slide, col.n, { left: col.x, top: 190, width: 245, height: 118 }, {
        size: 92, typeface: MONO, color: col.x === 845 ? C.red : C.text, bold: true,
      });
      text(slide, col.label, { left: col.x + 3, top: 311, width: 300, height: 35 }, {
        size: 23, bold: true,
      });
      text(slide, col.note, { left: col.x + 3, top: 350, width: 300, height: 30 }, {
        size: 17, typeface: MONO, color: C.muted,
      });
    }
    text(slide, "HELD-OUT TEST RACES", { left: 90, top: 431, width: 360, height: 24 }, {
      size: 16, typeface: MONO, bold: true, color: C.muted,
    });
    const raceNames = {"2023-AbuDhabi-R":"Abu Dhabi","2023-LasVegas-R":"Las Vegas","2023-Mexico-R":"Mexico","2023-SãoPaulo-R":"São Paulo"};
    text(slide, seasonData.split.test_races.map(id => raceNames[id] ?? id).join("     "), { left: 90, top: 468, width: 1050, height: 44 }, {
      size: 28, color: C.text, bold: true,
    });
    text(slide, `${seasonData.split.test_window_count.toLocaleString("en-US")} test graph windows`, { left: 90, top: 551, width: 450, height: 30 }, {
      size: 21, typeface: MONO, color: C.text,
    });
    text(slide, `${seasonMetrics.sample_count.toLocaleString("en-US")} valid node-target predictions`, { left: 585, top: 551, width: 600, height: 30 }, {
      size: 21, typeface: MONO, color: C.red, bold: true,
    });
    text(slide, `${seasonData.split.method} split · 15 training / 3 validation / ${seasonData.split.test_races.length} test races. Graph windows and valid prediction rows are different units.`, { left: 90, top: 596, width: 1070, height: 35 }, {
      size: 18, color: C.muted,
    });
    footer(slide, 4);
    note(slide, [
      "Source: hammertime_100/frontend/data/season_model.json, split and count_note. Archived source paths and fingerprints appear under provenance.",
      "Test race IDs, test windows and target-row counts come from the tracked season_model.json split and count_note fields.",
    ]);
  }

  // 5. Main result with native chart
  {
    const slide = newSlide(presentation);
    title(slide, "Matched test MAE differs by about 2 ms");
    text(slide, "TEST MAE  /  SECONDS  /  ZERO-BASED AXIS", { left: 80, top: 163, width: 610, height: 26 }, {
      size: 16, typeface: MONO, color: C.muted, bold: true,
    });
    const maeChart = slide.charts.add("bar", {
      position: { left: 78, top: 202, width: 630, height: 342 },
      categories: ["ST-GNN", "Zero predictor"],
      series: [{
        name: "MAE (seconds)",
        // Keep the editable workbook snapshot within Excel's 15-significant-digit limit.
        values: [Number(seasonMetrics.mae_sec.toPrecision(12)), Number(zeroMetrics.mae_sec.toPrecision(12))],
        valuesFormatCode: "0.00000",
        points: [
          { idx: 0, fill: C.red },
          { idx: 1, fill: "#7B8589" },
        ],
      }],
      barOptions: { direction: "bar", grouping: "clustered", gapWidth: 80 },
      hasLegend: false,
      chartFill: C.paper,
      chartLine: { style: "solid", fill: C.paper, width: 0 },
      plotAreaFill: C.paper,
      plotAreaLine: { style: "solid", fill: C.paper, width: 0 },
      xAxis: {
        textStyle: { fill: C.muted, fontSize: 15 },
        line: { style: "solid", fill: C.rule, width: 1 },
      },
      yAxis: {
        min: 0,
        max: 0.1,
        majorUnit: 0.025,
        numberFormatCode: "0.000",
        textStyle: { fill: C.text, fontSize: 20, bold: true },
        majorGridlines: { style: "solid", fill: C.rule, width: 1 },
        line: { style: "solid", fill: C.rule, width: 1 },
      },
      dataLabels: {
        showValue: true,
        position: "outEnd",
        numberFormatCode: "0.00000",
        textStyle: { fill: C.text, fontSize: 19, bold: true },
      },
    });
    applyPresentationChartFont(maeChart, { fontFamily: MONO });
    text(slide, "ST-GNN", { left: 780, top: 220, width: 310, height: 28 }, {
      size: 16, typeface: MONO, bold: true, color: C.red,
    });
    text(slide, `${seasonMetrics.mae_sec.toFixed(5)} s`, { left: 780, top: 253, width: 360, height: 52 }, {
      size: 40, typeface: MONO, bold: true, color: C.text,
    });
    text(slide, `R²  ${seasonMetrics.r2_score.toFixed(5)}`, { left: 780, top: 321, width: 370, height: 35 }, {
      size: 24, typeface: MONO, color: C.text,
    });
    text(slide, "Zero predictor", { left: 780, top: 399, width: 310, height: 28 }, {
      size: 16, typeface: MONO, bold: true, color: C.muted,
    });
    text(slide, `${zeroMetrics.mae_sec.toFixed(5)} s`, { left: 780, top: 432, width: 360, height: 52 }, {
      size: 34, typeface: MONO, bold: true, color: C.text,
    });
    text(slide, `R²  ${zeroMetrics.r2_score.toFixed(5)}`, { left: 780, top: 490, width: 370, height: 35 }, {
      size: 24, typeface: MONO, color: C.text,
    });
    const maeGainPct = ((zeroMetrics.mae_sec - seasonMetrics.mae_sec) / zeroMetrics.mae_sec * 100).toFixed(2);
    text(slide, `Same ${seasonMetrics.sample_count.toLocaleString("en-US")} target rows. MAE falls by ${maeGainPct}%; the R² acceptance gate still fails.`, { left: 80, top: 579, width: 1100, height: 43 }, {
      size: 22, color: C.text,
    });
    footer(slide, 5);
    note(slide, [
      "Source: hammertime_100/frontend/data/season_model.json. Both values use the same held-out target rows. Relative MAE gain is computed from the tracked test and matched zero-predictor metrics. The ST-GNN MAE gate passes; its R² gate fails.",
    ]);
  }

  // 6. Race-level test detail
  {
    const slide = newSlide(presentation);
    title(slide, "Held-out MAE varies across the four races");
    text(slide, "Race-level MAE and target counts", { left: 78, top: 151, width: 600, height: 32 }, {
      size: 23, color: C.muted,
    });
    const table = slide.tables.add({
      rows: 5,
      columns: 3,
      left: 80,
      top: 211,
      width: 1118,
      height: 310,
      columnWidths: [520, 250, 348],
      values: [
        ["HELD-OUT RACE", "TEST MAE (s)", "VALID NODE-TARGETS"],
        ...raceMetrics.map(item => [
          ({"2023-AbuDhabi-R":"Abu Dhabi","2023-LasVegas-R":"Las Vegas","2023-Mexico-R":"Mexico","2023-SãoPaulo-R":"São Paulo"})[item.race_id] ?? item.race_id,
          item.mae_sec.toFixed(5),
          item.sample_count.toLocaleString("en-US"),
        ]),
      ],
    });
    styleTable(table, 5, 3, 22);
    text(slide, "Across-race values sum to the archived test prediction count. Each race stays on its own row; no per-race ranking is adjusted for target coverage.", { left: 82, top: 554, width: 1085, height: 56 }, {
      size: 21, color: C.muted, leading: 1.15,
    });
    footer(slide, 6);
    note(slide, [
      "Source: hammertime_100/frontend/data/season_model.json, race_metrics. Counts are valid node-target predictions; values remain separated by race.",
    ]);
  }

  // 7. Acceptance gates
  {
    const slide = newSlide(presentation);
    title(slide, "Acceptance checks return mixed results");
    const table = slide.tables.add({
      rows: 5,
      columns: 3,
      left: 80,
      top: 183,
      width: 1118,
      height: 342,
      columnWidths: [390, 230, 498],
      values: [
        ["CHECK", "STATUS", "OBSERVED EVIDENCE"],
        ["Test MAE target", seasonMetrics.target_mae_pass ? "PASS" : "FAIL", `${seasonMetrics.mae_sec.toFixed(5)} s`],
        ["Test R² target", seasonMetrics.target_r2_pass ? "PASS" : "FAIL", seasonMetrics.r2_score.toFixed(5)],
        ["Clean-air calibration", physicsCleanAir.passed ? "PASS" : "FAIL", `${physicsCleanAir.observed_mae.toFixed(5)} s MAE across ${physicsCleanAir.sample_count.toLocaleString("en-US")} samples`],
        ["DRS train check", physicsDrs.complete_laps === 0 ? "UNMEASURABLE" : (physicsDrs.passed ? "PASS" : "FAIL"), physicsDrs.complete_laps === 0 ? "No complete qualifying held-out lap" : `${physicsDrs.complete_laps} complete held-out laps`],
      ],
    });
    styleTable(table, 5, 3, 20);
    table.getCell(1, 1).text.style = { typeface: MONO, fontSize: 20, bold: true, color: C.green, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
    table.getCell(2, 1).text.style = { typeface: MONO, fontSize: 20, bold: true, color: C.fail, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
    table.getCell(3, 1).text.style = { typeface: MONO, fontSize: 20, bold: true, color: C.green, alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
    table.getCell(4, 1).text.style = { typeface: MONO, fontSize: 17, bold: true, color: "#8A5C13", alignment: "center", verticalAlignment: "middle", autoFit: "shrinkText" };
    text(slide, "A missing qualifying DRS lap is not a pass or a measured aerodynamic effect. The check remains open until coverage supports a complete-lap result.", { left: 82, top: 559, width: 1080, height: 57 }, {
      size: 21, color: C.muted, leading: 1.18,
    });
    footer(slide, 7);
    note(slide, [
      "Source: hammertime_100/frontend/data/season_model.json, metrics.physics_tests and target gate flags.",
      `Clean-air threshold and result come from the tracked season JSON: expected ${physicsCleanAir.expected}, observed MAE ${physicsCleanAir.observed_mae} s over ${physicsCleanAir.sample_count} samples. DRS complete-lap count is ${physicsDrs.complete_laps}; zero makes the check unmeasurable.`,
    ]);
  }

  // 8. Model families in separate evidence tiers
  {
    const slide = newSlide(presentation, true);
    title(slide, "Model-family comparisons need matching evidence", "HAMMERTIME  /  COMPARISON DESIGN", true);
    text(slide, "FASTF1 2023 HELD-OUT TEST", { left: 80, top: 165, width: 500, height: 24 }, {
      size: 16, typeface: MONO, color: C.teal, bold: true,
    });
    text(slide, "ST-GNN vs zero predictor", { left: 80, top: 202, width: 515, height: 42 }, {
      size: 29, bold: true, color: C.white,
    });
    text(slide, `${seasonMetrics.mae_sec.toFixed(5)} s  /  ${zeroMetrics.mae_sec.toFixed(5)} s MAE\nSame held-out races and target rows\n${seasonMetrics.sample_count.toLocaleString("en-US")} valid predictions`, { left: 80, top: 268, width: 500, height: 104 }, {
      size: 22, typeface: MONO, color: C.white, leading: 1.25,
    });
    text(slide, "OPENF1 HUNGARY 2023 VALIDATION", { left: 676, top: 165, width: 530, height: 24 }, {
      size: 16, typeface: MONO, color: C.redSoft, bold: true,
    });
    text(slide, "Separate model-family suite", { left: 676, top: 202, width: 530, height: 42 }, {
      size: 29, bold: true, color: C.white,
    });
    const table = slide.tables.add({
      rows: 4,
      columns: 2,
      left: 676,
      top: 258,
      width: 526,
      height: 223,
      columnWidths: [335, 191],
      values: [
        ["MODEL", "VALIDATION MAE (s)"],
        ["ST-GNN legacy graph", hungary.runs.find(item => item.model === "legacy_input_legacy_graph").mae_sec.toFixed(5)],
        ["LSTM", hungary.runs.find(item => item.model === "lstm").mae_sec.toFixed(5)],
        ["XGBoost", hungary.runs.find(item => item.model === "xgboost").mae_sec.toFixed(5)],
      ],
    });
    table.borders.assign({ style: "solid", fill: "#3B464B", width: 1 });
    for (let r = 0; r < 4; r += 1) {
      for (let c = 0; c < 2; c += 1) {
        const cell = table.getCell(r, c);
        cell.fill = r === 0 ? "#273136" : (r % 2 === 0 ? "#182126" : "#121A1F");
        cell.text.style = {
          typeface: r === 0 ? MONO : (c === 1 ? MONO : FONT),
          fontSize: r === 0 ? 14 : 18,
          bold: r === 0 || c === 0,
          color: r === 0 ? C.mutedDark : C.white,
          insets: { top: 7, right: 8, bottom: 7, left: 8 },
          verticalAlignment: "middle",
          autoFit: "shrinkText",
        };
      }
    }
    text(slide, `Same ${hungary.runs.find(item => item.model === "lstm").sample_count.toLocaleString("en-US")}-target validation split within this suite.`, { left: 680, top: 496, width: 510, height: 28 }, {
      size: 17, color: C.mutedDark,
    });
    text(slide, "The OpenF1 table is validation-only. It uses a different provider, split, and target from the FastF1 season test. Do not compare its absolute scores with the left-hand result.", { left: 82, top: 558, width: 1098, height: 66 }, {
      size: 21, color: C.mutedDark, leading: 1.16,
    });
    footer(slide, 8, true);
    note(slide, [
      "FastF1 source: hammertime_100/frontend/data/season_model.json.",
      "Separate OpenF1 validation source: hammertime_100/frontend/data/experiments.json, group openf1-hungary-circuit-screen. The group records the OpenF1 2023 Hungary validation screen; variants are not compared with the season test.",
      "These are separate evidence tiers. Only ST-GNN and zero predictor share the FastF1 season test target set.",
    ]);
  }

  // 9. Frontend
  {
    const slide = newSlide(presentation);
    title(slide, "The dashboard presents saved evidence, not live inference");
    text(slide, "The frontend connects the archived season run to a research view with race filters, held-out predictions, race-level summaries, and gate context.", { left: 80, top: 152, width: 1040, height: 55 }, {
      size: 22, color: C.muted, leading: 1.16,
    });
    await addDashboardImage(slide, path.join(SCRIPT_DIR, "assets/dashboard-preview.png"), {
      left: 80, top: 232, width: 680, height: 377,
    });
    text(slide, "RUN 769a76aab164b514cc0a", { left: 810, top: 253, width: 390, height: 26 }, {
      size: 15, typeface: MONO, color: C.red, bold: true,
    });
    text(slide, "Archived metrics\nlinked to checkpoint\n\nA deterministic sample\nof held-out predictions\n\nRace, driver and\nmini-sector views", { left: 810, top: 292, width: 380, height: 218 }, {
      size: 22, color: C.text, leading: 1.17,
    });
    text(slide, "The browser displays saved outputs. It does not load the checkpoint or run fresh model inference.", { left: 810, top: 540, width: 386, height: 60 }, {
      size: 19, color: C.muted, leading: 1.15,
    });
    footer(slide, 9);
    note(slide, [
      "Screenshot: reports/HammerTime_Project_Overview_2026/assets/dashboard-preview.png. The page presents archived FastF1 evaluation metrics and saved predictions; it does not run checkpoint inference in the browser.",
      "Sources: hammertime_100/frontend/data/season_model.json; hammertime_100/frontend/data/export_season_model.py; hammertime_100/frontend/app.js; frontend audit notes.",
      "The browser charts a deterministic sample of archived held-out predictions. It does not run checkpoint inference. Independent Chromium smoke verified the exact metrics, race and driver filters, the 20-row mini-sector table, mobile width without horizontal overflow, and explicit fetch-error state.",
    ]);
  }

  // 10. Experiment registry
  {
    const slide = newSlide(presentation);
    title(slide, "Experiment records stay inside their evidence groups");
    text(slide, "The registry keeps test, validation, pilot and report-only results apart, with source paths and hashes attached to each group.", { left: 80, top: 150, width: 1100, height: 58 }, {
      size: 22, color: C.muted, leading: 1.16,
    });
    await addDashboardImage(slide, path.join(SCRIPT_DIR, "assets/experiment-registry-preview.png"), {
      left: 78, top: 226, width: 650, height: 380,
    });
    text(slide, String(scoredRegistryCount + noScoreRegistryCount), { left: 788, top: 232, width: 150, height: 68 }, {
      size: 54, typeface: MONO, bold: true, color: C.red,
    });
    text(slide, "archived records", { left: 788, top: 301, width: 360, height: 28 }, {
      size: 18, color: C.muted,
    });
    text(slide, String(registryData.groups.length), { left: 788, top: 355, width: 150, height: 64 }, {
      size: 48, typeface: MONO, bold: true, color: C.teal,
    });
    text(slide, "evaluation groups", { left: 788, top: 416, width: 360, height: 28 }, {
      size: 18, color: C.muted,
    });
    text(slide, "Filters cover model family, evidence class, evaluation group and text search. Filtered CSV retains protocol and source hashes.", { left: 788, top: 483, width: 390, height: 99 }, {
      size: 21, color: C.text, leading: 1.17,
    });
    text(slide, `ST-GNN family has ${stgnnScoredCount} scored records and ${stgnnNoScoreCount} failed attempts with no scores.`, { left: 788, top: 602, width: 388, height: 55 }, {
      size: 18, color: C.muted, leading: 1.15,
    });
    footer(slide, 10, false, "HAMMERTIME  /  MODEL EVIDENCE REGISTRY  /  NOV 2026");
    note(slide, [
      "Screenshot: reports/HammerTime_Project_Overview_2026/assets/experiment-registry-preview.png, captured from the registry browser smoke at 1440x1200.",
      `Data source: hammertime_100/frontend/data/experiments.json, generated by hammertime_100/frontend/data/export_experiments.py. The JSON contains ${scoredRegistryCount + noScoreRegistryCount} records across ${registryData.groups.length} evidence groups: ${scoredRegistryCount} scored and ${noScoreRegistryCount} failed with null metric fields.`,
      `The ST-GNN family count is ${stgnnRegistryRuns.length} records: ${stgnnScoredCount} with scores and ${stgnnNoScoreCount} failed/no-score pilot attempts. Filtered CSV preserves each evidence group's dataset, protocol, unit and source paths/hashes. The registry does not rank incompatible evidence groups.`,
    ]);
  }

  // 11. Limits and next evidence
  {
    const slide = newSlide(presentation, true);
    title(slide, "The next evaluation must close four evidence gaps", "HAMMERTIME  /  WHAT REMAINS", true);
    const gaps = [
      ["01", "Explained variance", `R² is ${seasonMetrics.r2_score.toFixed(5)} and misses its configured gate.`],
      ["02", "Prospective target", "Full-session references use future clean-air observations."],
      ["03", "DRS validation", "No complete qualifying held-out lap was available."],
      ["04", "Model-family test", "XGBoost and LSTM need the same FastF1 held-out protocol before test-score comparison."],
    ];
    let top = 174;
    for (const [num, label, body] of gaps) {
      text(slide, num, { left: 80, top, width: 60, height: 38 }, {
        size: 20, typeface: MONO, color: C.redSoft, bold: true,
      });
      text(slide, label, { left: 155, top, width: 325, height: 35 }, {
        size: 24, bold: true, color: C.white,
      });
      text(slide, body, { left: 500, top: top + 1, width: 675, height: 54 }, {
        size: 21, color: C.mutedDark, leading: 1.15,
      });
      top += 103;
    }
    text(slide, `Current reading: MAE ${seasonMetrics.mae_sec.toFixed(5)} s versus ${zeroMetrics.mae_sec.toFixed(5)} s matched zero predictor, R² ${seasonMetrics.r2_score.toFixed(5)}, clean-air ${physicsCleanAir.passed ? "pass" : "fail"}, DRS ${physicsDrs.complete_laps === 0 ? "unmeasurable" : (physicsDrs.passed ? "pass" : "fail")}.`, { left: 80, top: 602, width: 1100, height: 48 }, {
      size: 22, color: C.white, bold: true, leading: 1.15,
    });
    footer(slide, 11, true);
    note(slide, [
      "Sources: hammertime_100/frontend/data/season_model.json; OPEN_QUESTIONS.md, items 2, 8, and 11; RESEARCH_CHANGELOG.md, 2026-11-01 full-season and model enhancement records.",
      "This slide records unresolved research and evidence gaps. No new threshold, split, or acceptance criterion is proposed.",
    ]);
  }

  // Export editable PPTX plus private rendered previews for review.
  for (const [index, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(PREVIEW, `slide-${String(index + 1).padStart(2, "0")}.png`), png);
  }
  const montage = await presentation.export({ format: "webp", montage: true });
  await writeBlob(path.join(BUILD, "montage.webp"), montage);
  const draft = path.join(BUILD, "candidate.pptx");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(draft);
  await fs.writeFile(path.join(BUILD, "deck-inspect.ndjson"), (await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes",
    maxChars: 200000,
  })).ndjson);
  const validation = await finalizePresentation({
    workspaceDir: ROOT,
    candidatePath: draft,
    finalPath: OUTPUT,
    pythonExecutable: "/home/andromeda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", `${W * 9525},${H * 9525}`, "--validate-bullet-geometry", "--validate-heading-fit", ...[6, 7, 8].flatMap(number => ["--require-native-table-slide", String(number)])],
    requiredNativeTableOwnerSlides: [6, 7, 8],
    requiredNativeChartOwnerSlides: [5],
    explicitTotalSlideCount: 11,
    materializeLiteralChartWorkbooks: true,
    fontPolicy: { basis: "design", families: [FONT, MONO] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(BUILD, "HammerTime_Project_Overview_2026_Registry.validation.json"),
  });
  console.log(JSON.stringify({ draft, output: OUTPUT, preview: PREVIEW, slides: presentation.slides.items.length, validation }));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
