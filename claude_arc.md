# HammerTime: Architecture and Module Responsibilities

**Language standard:** ASD-STE100

---

## 1. System Overview

HammerTime has 8 modules. Each module has one task. Each module reads defined input and writes defined output. A module does not read the internal data of another module.

```
[M1 Ingestion] -> [M2 Preprocessing] -> [M3 Target Builder]
                         |                     |
                         v                     v
                  [M4 Graph Builder] -> [M5 Dataset and Split]
                                               |
                                               v
                                        [M6 Model]
                                               |
                                               v
                                        [M7 Training]
                                               |
                                               v
                           [M8 Evaluation] <- [M9 Baselines]
```

**Note:** M9 is a separate module. It uses the same splits as M7. The total count is 9 modules, including M9.

---

## 2. Module Summary

| ID | Module | File | Main task |
|---|---|---|---|
| M1 | Ingestion | `ingest.py` | Download and cache race data |
| M2 | Preprocessing | `preprocess.py` | Align and clean telemetry |
| M3 | Target Builder | `target.py` | Calculate the delay target |
| M4 | Graph Builder | `graph.py` | Create nodes, edges, and weights |
| M5 | Dataset and Split | `dataset.py` | Build windows and split data |
| M6 | Model | `model.py` | Define the ST-GNN |
| M7 | Training | `train.py` | Train and save the model |
| M8 | Evaluation | `evaluate.py` | Measure accuracy and check physics |
| M9 | Baselines | `baselines.py` | Run XGBoost and LSTM |

---

## 3. Module Specifications

### M1. Ingestion

**Responsibility:** Get raw race data from FastF1. Store it in a local cache.

| Item | Description |
|---|---|
| Input | Year, Grand Prix name |
| Output | Session object, Parquet files in `data/cache` |
| Does | Download telemetry, laps, and tyre data. Reuse the cache. |
| Does not | Clean, align, or change the data |
| Failure case | No network and no cache. Raise an error. |

---

### M2. Preprocessing

**Responsibility:** Make all car data usable on one time grid.

| Item | Description |
|---|---|
| Input | Raw telemetry and lap tables from M1 |
| Output | One aligned table per car, same time grid |
| Does | Resample to a common rate. Interpolate short gaps. Mark long gaps. Mark safety car laps and pit laps. Compute the gap Δt to the car ahead. |
| Does not | Create graphs or targets |
| Failure case | Gap longer than the allowed limit. Mark the rows as invalid. |

---

### M3. Target Builder

**Responsibility:** Calculate the traffic delay for each car and mini-sector.

| Item | Description |
|---|---|
| Input | Aligned tables from M2 |
| Output | Label table: car, time step, Δt_sector |
| Does | Find clean-air laps (gap more than 3 s). Calculate the median clean-air time per mini-sector and tyre compound. Subtract this baseline from the actual time. |
| Does not | Build edges or features |
| Failure case | Too few clean laps. Mark the baseline as missing. Exclude those rows. |

**Formula:**

```
Δt_sector = t_actual − t_clean_air_baseline
```

---

### M4. Graph Builder

**Responsibility:** Create one graph for each time step.

| Item | Description |
|---|---|
| Input | Aligned tables from M2 |
| Output | List of snapshots. Each has `x`, `edge_index`, `edge_attr` |
| Does | Create 20 nodes. Create the feature vector for each node. Create a directed edge from leader to follower when 0 < Δt ≤ 2.0 s. Calculate the RBF weight. |
| Does not | Calculate targets. Split data. |
| Failure case | Car not on track (retired or in pit). Keep the node. Mask its features. |

**Node features:** velocity, throttle, tyre compound (one-hot), tyre age, mini-sector index.

**Edge weight:**

```
e_ij = exp( -Δt_ij^2 / (2 σ^2) )   if 0 < Δt_ij ≤ 2.0 s
e_ij = 0                           otherwise
```

---

### M5. Dataset and Split

**Responsibility:** Prepare training samples. Prevent data leakage.

| Item | Description |
|---|---|
| Input | Snapshots from M4, labels from M3 |
| Output | Train, validation, and test loaders |
| Does | Build windows of W snapshots. Attach the target at step t+K. Split by race or by time. Fit the scaler on the training set only. |
| Does not | Train the model |
| Failure case | Window crosses a race boundary. Discard the window. |

**WARNING:** Do not split at random. Neighbor snapshots are almost identical.

---

### M6. Model

**Responsibility:** Define the ST-GNN network.

| Item | Description |
|---|---|
| Input | Window of W snapshots |
| Output | Predicted delay ŷ for each of 20 cars |
| Does | Project features. Pass messages with 2 GATConv layers. Learn time patterns with Conv1d. Output values with an MLP head. |
| Does not | Load data or compute the loss |
| Failure case | Snapshot has no edges. The GAT layer still works. The node keeps its own features. |

**Layer order:** Linear, GATConv, GATConv, Conv1d, MLP.

---

### M7. Training

**Responsibility:** Fit the model weights.

| Item | Description |
|---|---|
| Input | Loaders from M5, model from M6, config |
| Output | `best.pt`, training log |
| Does | Run the Adam optimizer. Compute the L1 loss. Apply weight decay. Run validation each epoch. Apply early stopping. Save the best weights. |
| Does not | Test on the test set |
| Failure case | Loss is NaN. Stop and report. Check the feature scaling. |

---

### M8. Evaluation

**Responsibility:** Measure model quality. Check traffic physics.

| Item | Description |
|---|---|
| Input | `best.pt`, test loader |
| Output | Report with MAE, R², and physical test results |
| Does | Calculate MAE and R². Run the clean-air test (gap more than 3 s, penalty near zero). Run the DRS-train test (gap less than 0.8 s, penalty 0.4 s to 0.8 s per lap). Compare with M9. |
| Does not | Change the model weights |
| Failure case | A physical test fails. Report it. Check M3 and M4 first. |

---

### M9. Baselines

**Responsibility:** Give a fair comparison.

| Item | Description |
|---|---|
| Input | Same splits as M5 |
| Output | MAE and R² for each baseline |
| Does | Train isolated single-car XGBoost. Train isolated single-car LSTM. |
| Does not | Use any edge or neighbor data |
| Failure case | Different split from M7. Results are not valid. Use the same split. |

---

## 4. Interface Contracts

| From | To | Data | Format |
|---|---|---|---|
| M1 | M2 | Raw telemetry, laps | Parquet or DataFrame |
| M2 | M3 | Aligned car tables | DataFrame |
| M2 | M4 | Aligned car tables | DataFrame |
| M3 | M5 | Labels | DataFrame (car, t, Δt_sector) |
| M4 | M5 | Snapshots | List of `Data` objects |
| M5 | M7 | Windows and targets | DataLoader |
| M6 | M7 | Model | `nn.Module` |
| M7 | M8 | Weights | `best.pt` |
| M5 | M9 | Same splits | DataFrame |
| M9 | M8 | Baseline scores | Dictionary |

---

## 5. Dependency Rules

1. A module may depend only on the modules before it in the data flow.
2. M6 must not import M7 or M8.
3. M4 and M3 must not import each other. Both read M2 output.
4. M8 and M9 must use the same split object from M5.
5. All parameters come from `configs/default.yaml`. Do not hard-code them.

---

## 6. Responsibility Matrix

| Task | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 |
|---|---|---|---|---|---|---|---|---|---|
| Download data | R | | | | | | | | |
| Align time grid | | R | | | | | | | |
| Compute gap Δt | | R | | C | | | | | |
| Clean-air baseline | | | R | | | | | | |
| Create edges | | | | R | | | | | |
| Scale features | | | | | R | | | | |
| Split data | | | | | R | | | | C |
| Define network | | | | | | R | | | |
| Fit weights | | | | | | | R | | |
| Report metrics | | | | | | | | R | C |
| Run baselines | | | | | | | | | R |

**Key:** R = Responsible. C = Consumes the result.

---

## 7. Work Split for Two Authors

| Author | Modules | Reason |
|---|---|---|
| Author 1 (Adithya Sai Pratheek) | M1, M2, M3, M4, M5 | Data pipeline |
| Author 2 (Abhinav Sriram) | M6, M7, M8, M9 | Model and evaluation |

**Note:** This split is a suggestion **[ASSUMED]**. The two authors must agree on the interface contracts in Section 4 first. After that, each author can work alone.

---

## 8. Build Order

1. Build M1 and M2. Check the aligned data.
2. Build M3 and M4 in parallel.
3. Build M5. Test the split rule.
4. Build M6. Test with random input.
5. Build M7. Train on one race.
6. Build M9. Get baseline scores.
7. Build M8. Run all tests.
8. Scale to more races.

**Note:** I corrected the module count. The first diagram says 8 modules. The correct count is 9.