# HammerTime: Dynamic ST-GNN for F1 Traffic Congestion and Lap-Time Slowdown Prediction

**Formula 1 Broadcast Insights Division — Fast-Track Prototype**  
**Authors:** Adithya Sai Pratheek, Abhinav Sriram  
**Language Standard:** ASD-STE100 Plain English  

---

## 1. System Overview

Standard Formula 1 lap-time prediction models evaluate each car in isolation. In reality, Grand Prix racing is dominated by spatio-temporal aerodynamic interactions: dirty air in corners, DRS trains on straights, and blue-flag delays while lapping backmarkers.

**HammerTime** models the race as a dynamic, edge-varying graph:
- **Nodes ($N=20$):** Every Formula 1 car on the circuit.
- **Edges ($j \to i$):** Directed leader-to-follower aerodynamic disturbance exists when time gap $0 < \Delta t_{ij} \le 2.0\text{ s}$.
- **Edge Weight ($e_{ij}$):** Gaussian Radial Basis Function (RBF) kernel:
  $$e_{ij} = \exp\left(-\frac{\Delta t_{ij}^2}{2\sigma^2}\right), \quad \sigma = 1.0\text{ s}$$
- **Target ($\Delta t_{\text{sector}}$):** Traffic delay relative to clean-air baseline ($t_{\text{actual}} - t_{\text{clean\_air\_baseline}}$).

---

## 2. Quickstart Runbook

HammerTime uses `uv` for reproducible, lightning-fast dependency synchronization and execution:

```bash
# 1. Sync dependencies
uv sync

# 2. Run full 100% pipeline verification
uv run python hammertime_100/run_pipeline.py

# 3. Launch interactive broadcast dashboard
uv run streamlit run hammertime_100/dashboard/app.py

# 4. Run 50% core dynamic graph builder & visualization (23 seconds)
uv run python hammertime_50/demo_core.py --mode graph

# 5. Run 50% complete pipeline (Graph + Real Training + Analytics)
uv run python hammertime_50/demo_core.py --mode full --epochs 10
```

---

## 3. Reports & Documentation

- 📊 **50% Core Architecture & Evaluation Report:** [`REPORT_50.md`](file:///m:/HammerTIme/REPORT_50.md) / [`hammertime_50/REPORT.md`](file:///m:/HammerTIme/hammertime_50/REPORT.md)
- 🕸️ **Multi-Tier Dynamic Graph Visualizer:** [`core_graph_visualization.png`](file:///m:/HammerTIme/core_graph_visualization.png)
- 📈 **Traffic Delay Distribution & Parity:** [`core_analytics.png`](file:///m:/HammerTIme/core_analytics.png)
- 📖 **Assumption Ledger & Tuning Parameters:** [`ASSUMPTIONS.md`](file:///m:/HammerTIme/ASSUMPTIONS.md)

---

## 4. Repository Structure

```
HammerTime/
├── hammertime_100/              # 100% Complete Production ST-GNN Pipeline
│   ├── configs/default.yaml     # Zero magic numbers configuration
│   ├── src/                     # Modules M1 - M9
│   ├── visuals/                 # V1 - V9 Broadcast Graphics
│   ├── dashboard/               # Interactive Streamlit Broadcast Dashboard
│   ├── broadcast/               # Rendered Broadcast Insight Cards
│   └── run_pipeline.py          # End-to-end verification pipeline
│
├── hammertime_50/               # 50% Core Functions & Analytics Baseline
│   ├── configs/core.yaml
│   ├── src/
│   └── demo_core.py
│
├── ASSUMPTIONS.md               # Assumption ledger tracking all tuned parameters
├── pyproject.toml               # uv project manifest
└── README.md                    # System documentation
```
