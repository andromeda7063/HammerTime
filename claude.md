# HammerTime: Dynamic GNN for F1 Traffic Congestion and Lap-Time Slowdown Prediction

**Document type:** Technical Report, Blueprint, and User Manual
**Authors:** Adithya Sai Pratheek (24BAI1445), Abhinav Sriram (24BAI1209)
**Language standard:** ASD-STE100

---

## 0. Document Control

| Item | Value |
|---|---|
| Project name | HammerTime |
| System type | Dynamic Spatio-Temporal Graph Neural Network (ST-GNN) |
| Data source | FastF1 open-source API |
| Frameworks | Python, PyTorch, PyTorch Geometric |
| Status | Design and implementation plan |

**Note:** Values marked **[ASSUMED]** are design choices. The source proposal does not give these values. Tune them during testing.

---

## 1. Scope and Purpose

### 1.1 Purpose

HammerTime predicts how much lap time a Formula 1 car loses in a mini-sector because of traffic. The system uses telemetry and timing data.

### 1.2 Problem

Race strategy depends on traffic effects. These effects are DRS trains, dirty air, and blue-flag delays. Standard models treat each car alone. They do not model how cars affect each other.

### 1.3 Solution

HammerTime models the race as a dynamic graph. Each car is a node. An edge shows that a leading car disturbs a trailing car. The model learns from a window of graphs and predicts the future delay of each car.

### 1.4 Out of Scope

- HammerTime does not plan pit stops.
- HammerTime does not plan overtakes.
- HammerTime does not use data from sources other than FastF1.

---

## 2. Terms and Definitions

| Term | Meaning |
|---|---|
| Mini-sector | One of M equal parts of the circuit |
| Dirty air | Disturbed air behind a car. It reduces downforce of the car behind |
| DRS | Drag Reduction System. A movable rear wing flap |
| DRS train | A group of cars that follow each other closely |
| Gap (Δt) | Time distance between a trailing car and the car ahead |
| Clean air | A gap of more than 3 s |
| Snapshot | The state of all cars at one time step |
| Window (W) | Number of past snapshots that the model reads |
| Horizon (K) | Number of future steps that the model predicts |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |
| RBF | Radial Basis Function |
| GAT | Graph Attention Network |

---

## 3. Blueprint (System Architecture)

### 3.1 Data Flow

```
FastF1 API
    |
    v
[1] Data Ingestion  -->  Parquet cache
    |
    v
[2] Preprocessing   -->  Aligned telemetry (all cars, same time grid)
    |
    v
[3] Graph Builder   -->  Snapshots G_t (nodes, edges, weights)
    |
    v
[4] Target Builder  -->  Delta t_sector per car
    |
    v
[5] ST-GNN Model    -->  GATConv (space) + Conv1d (time) + MLP head
    |
    v
[6] Evaluation      -->  MAE, R2, clean-air test, DRS-train test
```

### 3.2 Module List

| Module | Function | Output |
|---|---|---|
| Ingestion | Downloads and caches race data | Parquet files |
| Preprocessing | Aligns all cars on one time grid | Tables per car |
| Graph Builder | Creates nodes and weighted edges | Snapshot list |
| Target Builder | Calculates the delay target | Label table |
| Model | Learns space and time patterns | Predictions |
| Evaluation | Measures accuracy and checks physics | Report |
| Baselines | Runs XGBoost and LSTM | Comparison table |

---

## 4. Data Specification

### 4.1 Input Signals

Use these signals from FastF1:

- GPS position
- Speed
- Throttle
- Brake
- DRS status
- Tyre compound
- Tyre age
- Lap and pit-interval timing

### 4.2 Node Features

Each car i at time t has a feature vector X_i(t) in R^d. The vector contains:

| Feature | Symbol |
|---|---|
| Velocity | v_i |
| Throttle percentage | θ_i |
| Tyre compound | C_i |
| Tyre age | A_i |
| Mini-sector index | s_i |

**Note:** Encode the tyre compound as a one-hot vector. Scale all numeric features to zero mean and unit variance. Fit the scaler on the training set only.

### 4.3 Scale

- Number of nodes per graph: N = 20.
- Number of active edges per snapshot: about 30.
- Number of graphs per race: about 5,000.
- Expected training time: less than 15 minutes.

---

## 5. Graph Construction

### 5.1 Edge Rule

Create a directed edge from leading car j to trailing car i. Do this only when:

```
0 < Δt_ij ≤ 2.0 s
```

### 5.2 Edge Weight

Use the thresholded Gaussian RBF kernel:

```
e_ij(t) = exp( -Δt_ij^2 / (2 σ^2) )   if 0 < Δt_ij ≤ 2.0 s
e_ij(t) = 0                           otherwise
```

Set σ = 1.0 s as the initial value **[ASSUMED]**.

### 5.3 Gap Calculation

1. Use the FastF1 function `add_driver_ahead()` for each car.
2. Read the distance to the car ahead.
3. Divide this distance by the speed of the trailing car.
4. Use the result as Δt_ij.

**Note:** This is one method. Any method that gives a correct time gap is acceptable.

---

## 6. Target Definition

### 6.1 Formula

```
Δt_sector = t_actual − t_clean_air_baseline
```

### 6.2 Meaning

The target isolates the time lost to traffic. It removes the effect of car and engine performance differences.

### 6.3 Baseline Calculation **[ASSUMED]**

1. Select the laps of one driver where the gap is more than 3 s.
2. Group these laps by mini-sector and tyre compound.
3. Calculate the median sector time of each group.
4. Use this median as t_clean_air_baseline.

---

## 7. Model Specification

### 7.1 Objective

The model f_Θ reads graphs G(t−W : t). It predicts the delay Δy_i(t+K) for each car i.

Minimize this loss:

```
L = (1/N) Σ_i | Δy_i(t+K) − ŷ_i(t+K) |  +  λ ||Θ||_2^2
```

### 7.2 Layers

| Stage | Layer | Purpose |
|---|---|---|
| 1 | Input projection (Linear) | Maps features to hidden size |
| 2 | GATConv, 2 layers, edge weight as edge feature | Learns car-to-car influence |
| 3 | Conv1d across W snapshots | Learns time patterns |
| 4 | MLP head | Outputs ŷ for each car |

### 7.3 Initial Hyperparameters **[ASSUMED]**

| Parameter | Value |
|---|---|
| Hidden size | 64 |
| GAT heads | 4 |
| Window W | 10 snapshots |
| Horizon K | 5 snapshots |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Weight decay (λ) | 0.0001 |
| Batch size | 32 |
| Dropout | 0.1 |
| Maximum epochs | 100 |
| Early stopping patience | 10 epochs |

---

## 8. Implementation Details

### 8.1 Project Structure

```
hammertime/
├── data/
│   ├── cache/
│   └── processed/
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── graph.py
│   ├── target.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── baselines.py
├── configs/default.yaml
├── requirements.txt
└── README.md
```

### 8.2 Dependencies

```
fastf1
pandas
numpy
pyarrow
torch
torch-geometric
scikit-learn
xgboost
pyyaml
```

### 8.3 Ingestion (`ingest.py`)

```python
import fastf1

fastf1.Cache.enable_cache("data/cache")

def load_race(year, gp):
    session = fastf1.get_session(year, gp, "R")
    session.load(telemetry=True, laps=True)
    return session
```

### 8.4 Edge Builder (`graph.py`)

```python
import numpy as np
import torch

SIGMA = 1.0
MAX_GAP = 2.0

def build_edges(gaps):
    """gaps: dict {(i, j): dt} where j leads i."""
    src, dst, w = [], [], []
    for (i, j), dt in gaps.items():
        if 0 < dt <= MAX_GAP:
            src.append(j)   # leader
            dst.append(i)   # follower
            w.append(np.exp(-dt**2 / (2 * SIGMA**2)))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(w, dtype=torch.float).unsqueeze(1)
    return edge_index, edge_attr
```

### 8.5 Model (`model.py`)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class HammerTime(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, window=10, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads,
                            edge_dim=1, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads,
                            edge_dim=1, dropout=dropout)
        self.tconv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, snapshots):
        # snapshots: list of W Data objects (x, edge_index, edge_attr)
        outs = []
        for g in snapshots:
            h = torch.relu(self.proj(g.x))
            h = torch.relu(self.gat1(h, g.edge_index, g.edge_attr))
            h = torch.relu(self.gat2(h, g.edge_index, g.edge_attr))
            outs.append(h)                      # [N, hidden]
        z = torch.stack(outs, dim=2)            # [N, hidden, W]
        z = torch.relu(self.tconv(z))           # [N, hidden, W]
        z = z[:, :, -1]                         # last time step
        return self.head(z).squeeze(-1)         # [N]
```

### 8.6 Training Loop (`train.py`)

```python
import torch
import torch.nn as nn

def train(model, loader, val_loader, epochs=100, lr=1e-3, wd=1e-4, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.L1Loss()
    best, wait = float("inf"), 0
    for ep in range(epochs):
        model.train()
        for snaps, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(snaps), y)
            loss.backward()
            opt.step()
        val = evaluate_mae(model, val_loader, loss_fn)
        if val < best:
            best, wait = val, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= patience:
                break

@torch.no_grad()
def evaluate_mae(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for snaps, y in loader:
        total += loss_fn(model(snaps), y).item() * len(y)
        n += len(y)
    return total / n
```

**Note:** The L2 term of the loss is applied through `weight_decay` in Adam.

### 8.7 Data Split Rule

**WARNING:** Do not split snapshots at random. Neighbor snapshots are almost identical. A random split causes data leakage and false high scores.

Split by race or by time:

- Train: first part of the races.
- Validation: next part of the races.
- Test: last part of the races, or whole races not seen in training.

---

## 9. Evaluation Plan

### 9.1 Metrics and Targets

| Metric | Target |
|---|---|
| R² | 0.68 to 0.82 |
| MAE | Less than 0.15 s per mini-sector |
| Baseline XGBoost R² (expected) | About 0.50 |

### 9.2 Baselines

Train these models on the same splits:

1. Isolated single-car XGBoost.
2. Isolated single-car LSTM.

### 9.3 Physical Validation

| Test | Condition | Expected result |
|---|---|---|
| Clean air | Gap more than 3 s | Predicted penalty near zero |
| DRS train | Gap less than 0.8 s | Cumulative penalty of 0.4 s to 0.8 s per lap |

If a test fails, the model has not learned the correct traffic physics. Check the graph builder and the target first.

---

## 10. User Manual

### 10.1 Prerequisites

- Python 3.10 or later.
- Internet access for the first data download.
- A GPU is optional. The workload is small.

### 10.2 Installation

1. Create a virtual environment.
2. Run `pip install -r requirements.txt`.
3. Make sure PyTorch Geometric matches your PyTorch version.

### 10.3 Procedure: Download Race Data

1. Open `configs/default.yaml`.
2. Set the year and the Grand Prix name.
3. Run `python src/ingest.py`.
4. Wait until the download is complete.
5. Make sure that Parquet files exist in `data/cache`.

**Note:** The first download is slow. Later runs use the cache.

### 10.4 Procedure: Build Graphs and Targets

1. Run `python src/preprocess.py`.
2. Run `python src/target.py`.
3. Run `python src/graph.py`.
4. Make sure that the snapshot count is about 5,000 per race.

### 10.5 Procedure: Train the Model

1. Run `python src/train.py --config configs/default.yaml`.
2. Monitor the validation MAE.
3. Wait for early stopping or for the maximum epoch count.
4. Make sure that the file `best.pt` exists.

### 10.6 Procedure: Evaluate

1. Run `python src/evaluate.py --weights best.pt`.
2. Read the MAE and R² values.
3. Read the clean-air and DRS-train results.
4. Run `python src/baselines.py` for the comparison table.

### 10.7 Procedure: Predict on New Data

1. Load a new race with `ingest.py`.
2. Build its snapshots with `graph.py`.
3. Load `best.pt`.
4. Give W snapshots to the model.
5. Read the predicted delay for each car.

---

## 11. Troubleshooting

| Problem | Possible cause | Action |
|---|---|---|
| No edges in snapshots | Gap calculation wrong or units wrong | Check that Δt is in seconds |
| Very high R² | Data leakage | Use a race-based split |
| Predictions near zero for all cars | Too few traffic samples | Select races with more traffic |
| Missing telemetry values | Sensor gaps | Interpolate short gaps. Remove long gaps |
| Training does not converge | Features not scaled | Scale the features |
| GAT import error | Version mismatch | Install matching PyTorch Geometric |

---

## 12. Literature Gap Summary

| Area | Existing work | Limit |
|---|---|---|
| Traffic ST-GNN (STGCN, DCRNN, STG4Traffic) | Static sensor graphs | Cannot model moving cars |
| Attention (GAT) | Learns edge weights | Not tested on racing |
| Multi-agent trajectory (GATraj, dynamic graph convolution) | Moving agents | Road traffic only, not racing |
| F1 strategy and lap time (RL, DNN) | Single-car or scalar traffic penalty | No multi-car graph |
| FastF1 | Data access | No ML tools |

**Gap:** No known work models F1 traffic as a dynamic, edge-varying graph on a closed circuit. HammerTime fills this gap.

---

## 13. Limitations and Risks

- The 2.0 s edge threshold is fixed. It may not suit all circuits.
- The clean-air baseline depends on enough clean laps per driver.
- Safety car laps and pit laps add noise. Remove them or mark them.
- The accuracy targets are expected values. They are not proven results.
- Telemetry sample rates differ between cars. Align them before you build graphs.

---

## 14. References

1. Luo, Zhu, Zhang, Li (2023). STG4Traffic. arXiv.
2. Yu, Yin, Zhu (2018). STGCN. IJCAI.
3. Li, Yu, Shahabi, Liu (2018). DCRNN. ICLR.
4. Veličković et al. (2018). Graph Attention Networks. ICLR.
5. Cheng et al. (2022). GATraj. IEEE RA-L / arXiv.
6. Balerna et al. (2025). Towards Learning-Based Formula 1 Race Strategies. arXiv.
7. O'Kelly et al. (2022). F1TENTH overtaking algorithm.
8. Deep Neural Network-Based Lap Time Forecasting of Formula 1 Racing (2024).
9. A Formula 1 Strategy Predictor Using Reinforcement Learning (2023). IJRASET.
10. Li, Gao, et al. (2021). Spatial-Temporal Dynamic Graph Convolution. IEEE T-ITS.
11. Sharma et al. (2023). GNN Real-Time Traffic Speed Estimation. Sustainability.
12. Roembke et al. (2021–2024). FastF1.

---