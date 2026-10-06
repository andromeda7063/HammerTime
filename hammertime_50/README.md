# HammerTime 50%: Core Functions, Dynamic Graph Builder & Real Analytics

**Fast-Track Core Prototype for Formula 1 Traffic Delay Estimation**  
**Race Data:** 2021 Formula 1 Abu Dhabi Grand Prix (`VER` vs `HAM`)  
**Language Standard:** ASD-STE100 Plain English  

---

## 1. Overview & Capabilities

`hammertime_50` is the functional core prototype of the HammerTime Spatio-Temporal Graph Neural Network (ST-GNN). It ingests official Formula 1 telemetry, constructs dynamic spatio-temporal aerodynamic interaction graphs, visualizes multi-tier aerodynamic disturbance flow, trains a message-passing ST-GNN on real mini-sector delay targets, and detects DRS-train bottlenecks.

### Core Modules (`src/`)

1. **[`core_pipeline.py`](file:///m:/HammerTIme/hammertime_50/src/core_pipeline.py):**
   - Ingests real 2021 Abu Dhabi Grand Prix telemetry from FastF1 (416,879 rows, 19 drivers).
   - Resamples telemetry onto a uniform 4.0 Hz temporal grid.
   - Calculates dynamic time gaps ($\Delta t_{ij} = \Delta s_{ij} / v_i$).
   - Computes clean-air reference baselines ($t_{\text{clean\_air\_baseline}} = \text{median}(t_{\text{sector}} \mid \Delta t > 3.0\text{ s})$) and true delay labels ($\Delta t_{\text{sector}} = t_{\text{actual}} - t_{\text{clean\_air\_baseline}}$).

2. **[`core_graph.py`](file:///m:/HammerTIme/hammertime_50/src/core_graph.py):**
   - Builds 21,941 continuous graph snapshots across the 58-lap Grand Prix.
   - Connects cars with directed disturbance edges governed by thresholded Gaussian Radial Basis Function (RBF) weights:
     $$e_{ij} = \exp\left(-\frac{\Delta t_{ij}^2}{2\sigma^2}\right) \quad \text{for } 0 < \Delta t_{ij} \le 2.0\text{ s}, \quad \sigma = 1.0\text{ s}$$
   - Generates 137,355 directed disturbance edges (average 6.26 active edges per snapshot).
   - Normalizes node features: Speed ($v / 320$), Throttle ($T / 100$), Tyre Age ($L / 55$), Mini-Sector ($s / 20$).

3. **[`core_model.py`](file:///m:/HammerTIme/hammertime_50/src/core_model.py):**
   - Implements `CoreSTGNN`, a message-passing neural network.
   - Aggregates aerodynamic disturbance messages weighted by $e_{ij}$ from leading cars.
   - Performs temporal pooling across rolling historical windows ($W = 5$ snapshots).
   - Projects through a linear regression head to predict traffic delay ($\Delta t_{\text{sector}}$).

4. **[`core_analytics.py`](file:///m:/HammerTIme/hammertime_50/src/core_analytics.py):**
   - Renders circular aerodynamic network topology with team livery colors and driver codes.
   - Visualizes the $19 \times 19$ weighted disturbance adjacency matrix ($A[j, i] = e_{ij}$).
   - Detects DRS-train chains ($\ge 3$ consecutive cars within 1.0 s gap).
   - Plots prediction parity scatter and delay distribution histograms.

---

## 2. Dynamic Graph Topology & Aerodynamic Tiers

### 2.1 Edge Direction Rule (Physics-First)
All directed graph arrows point from **Leader ($j$) $\to$ Follower ($i$)** (e.g., `16 LEC -> 22 TSU`).
- **Physical Meaning:** The leading car generates aerodynamic turbulence (dirty air wake) that washes over and disrupts the following car, causing downforce loss and tyre degradation.

### 2.2 Aerodynamic Disturbance Tiers

| Tier | Color | Time Gap ($\Delta t$) | Gaussian Weight ($e_{ij}$) | Aerodynamic Impact |
|---|---|---|---|---|
| **Severe Dirty Air** | **Red (`#E10600`)** | $\Delta t < 0.8\text{ s}$ | $e_{ij} \ge 0.726$ | Heavy downforce loss ($\approx 35\%$), critical understeer, DRS-train bottleneck |
| **Moderate Wake** | **Orange (`#FF9900`)** | $0.8\text{ s} \le \Delta t < 1.4\text{ s}$ | $0.375 \le e_{ij} < 0.726$ | Noticeable wake disruption ($\approx 15\%-20\%$), tyre overheating in braking zones |
| **Light Wake** | **Green (`#10B981`)** | $1.4\text{ s} \le \Delta t \le 2.0\text{ s}$ | $0.135 \le e_{ij} < 0.375$ | Minor aerodynamic disturbance ($\approx 5\%-10\%$), marginal acoustic slipstream |
| **Clean Air** | *(No Edge)* | $\Delta t > 2.0\text{ s}$ | $e_{ij} = 0$ | Undisturbed laminar air flow |

---

## 3. Running `demo_core.py`

`demo_core.py` is the unified CLI runner. It supports full end-to-end execution, graph-only execution, snapshot inspection, and model training.

### 3.1 Fast Graph Building & Multi-Tier Visualization (23 Seconds)
To build all 21,941 snapshots, save the graph dataset, and render the multi-tier graph visualizer:
```bash
uv run python hammertime_50/demo_core.py --mode graph
```
**Output:**
- Persists graph snapshots to: [`data/cache/core_graph_snapshots.pt`](file:///m:/HammerTIme/data/cache/core_graph_snapshots.pt)
- Renders graph topology & adjacency matrix to: [`core_graph_visualization.png`](file:///m:/HammerTIme/core_graph_visualization.png)
- Prints real-time interaction inspection table:
  ```
  --- DYNAMIC GRAPH INTERACTION INSPECTION (WHO DISTURBS WHOM) ---
  Timestamp: 750.00 s | Nodes: 19 cars | Active Disturbance Edges: 11
  -----------------------------------------------------------------------------------------
  Leader Car (j)   ->   Follower Car (i)    Time Gap (Δt)   Weight (e_ij)   Aerodynamic Tier
  -----------------------------------------------------------------------------------------
  16 LEC           ->   22 TSU              0.53 s          0.8679          [SEVERE DIRTY AIR]
  99 GIO           ->   10 GAS              0.19 s          0.9815          [SEVERE DIRTY AIR]
  22 TSU           ->   77 BOT              1.75 s          0.2158          [LIGHT WAKE]
  3 RIC            ->   14 ALO              0.56 s          0.8549          [SEVERE DIRTY AIR]
  77 BOT           ->   31 OCO              1.73 s          0.2241          [LIGHT WAKE]
  4 NOR            ->   16 LEC              1.92 s          0.1569          [LIGHT WAKE]
  31 OCO           ->   3 RIC               1.30 s          0.4300          [MODERATE WAKE]
  5 VET            ->   18 STR              1.73 s          0.2252          [LIGHT WAKE]
  6 LAT            ->   47 MSC              1.44 s          0.3544          [LIGHT WAKE]
  47 MSC           ->   63 RUS              1.32 s          0.4200          [MODERATE WAKE]
  18 STR           ->   7 RAI               0.35 s          0.9411          [SEVERE DIRTY AIR]
  -----------------------------------------------------------------------------------------
  ```

### 3.2 Inspect Any Specific Snapshot
You can visualize any of the 21,941 snapshots across the Grand Prix:
```bash
# Snapshot 3000 (Lap 8, t=750s)
uv run python hammertime_50/demo_core.py --mode graph --snapshot-idx 3000

# Snapshot 1000 (Lap 3, t=250s)
uv run python hammertime_50/demo_core.py --mode graph --snapshot-idx 1000
```

### 3.3 Full End-to-End Pipeline (Graph + Training + Evaluation + Analytics)
```bash
uv run python hammertime_50/demo_core.py --mode full --epochs 10
```
**Steps executed:**
1. Ingests & aligns 416,879 rows of 2021 Abu Dhabi Grand Prix telemetry.
2. Constructs 21,941 graph snapshots and saves to `data/cache/core_graph_snapshots.pt`.
3. Renders multi-tier graph visualizer to `core_graph_visualization.png`.
4. Trains Core ST-GNN over 4,112 windows (Laps 1–43) and evaluates on unseen closing laps (Laps 44–58).
5. Detects 9,804 DRS-train disturbance events and renders `core_analytics.png`.

---

## 4. Visual Artifacts Produced

| Visual Artifact | Path | Description |
|---|---|---|
| **Graph Topology & Matrix** | [`core_graph_visualization.png`](file:///m:/HammerTIme/core_graph_visualization.png) | High-resolution circular network with curved Leader $\to$ Follower disturbance arrows (Red/Orange/Green), driver badges, team rings, and $19 \times 19$ heatmap. |
| **Traffic Delay Analytics** | [`core_analytics.png`](file:///m:/HammerTIme/core_analytics.png) | Ground truth vs. predicted parity scatter plot and delay distribution histogram across the Grand Prix. |
| **Cached Graph Dataset** | [`data/cache/core_graph_snapshots.pt`](file:///m:/HammerTIme/data/cache/core_graph_snapshots.pt) | Serialized PyTorch tensor dataset containing all 21,941 graph snapshots and node mappings for rapid reuse. |

---

## 5. Technical Report

For the in-depth architectural breakdown, mathematical equations, real telemetry evaluation diagnostics, and transition roadmap to 100%, see:
👉 **[`hammertime_50/REPORT.md`](file:///m:/HammerTIme/hammertime_50/REPORT.md)**
