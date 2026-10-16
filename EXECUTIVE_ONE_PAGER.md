# HammerTime — Executive Overview

HammerTime is a research prototype for estimating Formula 1 traffic delay. It represents cars as nodes in time-indexed graphs and predicts mini-sector duration relative to a clean-air reference calculated from the same race session.

## What the 100-tier implementation contains

The `hammertime_100/` implementation has nine modules. It loads real FastF1 sessions, aligns telemetry, builds session-scoped targets and graph snapshots, creates temporal or race-held-out samples, trains a spatio-temporal graph neural network, evaluates it, and runs XGBoost and LSTM baselines. A dashboard and nine visual generators consume pipeline artifacts.

The graph uses directed leader-to-follower edges for valid gaps up to the configured two-second limit. Its edge weights use a Gaussian radial basis function. Node inputs include speed, throttle, tyre compound, tyre age, and mini-sector position. Safety-car samples are masked from targets and windows that contain safety-car snapshots are excluded. Pit and invalid samples are masked.

## What the pipeline reports

For each run, the report contains held-out MAE and R², baseline availability and scores, physics-test results, split metadata, and row-level predictions. The runner also records whether the configured quality gates passed and which visuals or cards were unavailable.

The configured clean-air and DRS checks are acceptance criteria. They are not measured results until a run produces them. No 100-tier performance numbers or superiority claims are published here because this repository state has not been executed for this review.

## Operating conditions

The pipeline requires accessible FastF1 data or a usable FastF1 cache. A single session uses a temporal split. Race-held-out evaluation requires at least three configured sessions with distinct race identifiers. The clean-air reference is computed per session; model features and scaling are fit using the training partition.

Start with [`README.md`](README.md) and review [`hammertime_100/REPORT.md`](hammertime_100/REPORT.md) for implementation details, outputs, and verification status.
