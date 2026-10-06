# HammerTime 50%: Core System Architecture & Evaluation Report

**Client:** Formula 1 · Broadcast Insights Division  
**Subject:** 50% Prototype Execution, Real FastF1 Telemetry, Dynamic Graph Construction & Analytics  
**Standard:** ASD-STE100 Plain English  
**Detailed Technical Report:** 👉 **[`hammertime_50/REPORT.md`](file:///m:/HammerTIme/hammertime_50/REPORT.md)**  

---

## 1. Executive Summary

This report certifies the successful execution of **HammerTime 50%**, demonstrating end-to-end functionality on actual race telemetry from the **2021 Abu Dhabi Grand Prix** (`VER` vs `HAM`):

1. **Real-World Telemetry Ingestion (`core_pipeline.py`):**
   - 416,879 synchronized telemetry rows across 19 drivers resampled at 4.0 Hz.
   - 19,768 true clean-air baseline mini-sector delay labels ($\Delta t_{\text{sector}}$).
2. **Dynamic Graph Construction & Serialization (`core_graph.py`):**
   - 21,941 continuous graph snapshots constructed and serialized to [`data/cache/core_graph_snapshots.pt`](file:///m:/HammerTIme/data/cache/core_graph_snapshots.pt).
   - 137,355 directed disturbance edges governed by thresholded Gaussian RBF weights:
     $$e_{ij} = \exp\left(-\frac{\Delta t_{ij}^2}{2\sigma^2}\right) \quad (\sigma = 1.0\text{ s}, \quad 0 < \Delta t_{ij} \le 2.0\text{ s})$$
3. **Multi-Tier Aerodynamic Wake Visualization (`core_analytics.py`):**
   - Physical arrow direction: **Leader ($j$) $\to$ Follower ($i$)** (the car ahead casting dirty air wake onto trailing cars).
   - 🔴 **Severe Dirty Air / DRS Train:** $\Delta t < 0.8\text{ s}$ ($e_{ij} \ge 0.726$)
   - 🟠 **Moderate Wake:** $0.8\text{ s} \le \Delta t < 1.4\text{ s}$ ($0.375 \le e_{ij} < 0.726$)
   - 🟢 **Light Wake:** $1.4\text{ s} \le \Delta t \le 2.0\text{ s}$ ($0.135 \le e_{ij} < 0.375$)
   - Circular network topology with driver team rings and full $19 \times 19$ disturbance matrix rendered to [`core_graph_visualization.png`](file:///m:/HammerTIme/core_graph_visualization.png).
4. **ST-GNN Model Training (`core_model.py`):**
   - Trained over 4,112 real race windows (Laps 1–43) across 10 epochs.
   - Evaluated on 1,371 unseen closing race windows (Laps 44–58).
5. **Traffic Delay Distribution Visualizer:**
   - 9,804 DRS-train disturbance events detected and charted in [`core_analytics.png`](file:///m:/HammerTIme/core_analytics.png).

---

## 2. Interactive CLI Runner Commands

```bash
# 1. Fast Graph Construction & Multi-Tier Visualization (23 seconds)
uv run python hammertime_50/demo_core.py --mode graph

# 2. Inspect Specific Snapshot (e.g., Snapshot 3000, t=750s / Lap 8)
uv run python hammertime_50/demo_core.py --mode graph --snapshot-idx 3000

# 3. Full Pipeline Execution (Graph + Training + Evaluation + Analytics)
uv run python hammertime_50/demo_core.py --mode full --epochs 10
```

---

## 3. Visual Artifacts

- **Dynamic Graph Topology & Adjacency Matrix:** [`core_graph_visualization.png`](file:///m:/HammerTIme/core_graph_visualization.png)
- **Model Regression Parity & Delay Distribution:** [`core_analytics.png`](file:///m:/HammerTIme/core_analytics.png)
- **Serialized Graph Dataset:** [`data/cache/core_graph_snapshots.pt`](file:///m:/HammerTIme/data/cache/core_graph_snapshots.pt)

---

## 4. Full Technical Report

For mathematical derivations, layer formulations, safety-car diagnostics, and the transition roadmap to the 100% production suite (`hammertime_100`), please see:
👉 **[`hammertime_50/REPORT.md`](file:///m:/HammerTIme/hammertime_50/REPORT.md)**
