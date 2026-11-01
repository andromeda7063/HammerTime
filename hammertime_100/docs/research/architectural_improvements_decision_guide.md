# HammerTime ST-GNN: Architectural Improvements & Agent Decision Guide

**Status:** Research & Engineering Design Specification  
**Target Codebase:** `hammertime_100/`  
**Standard:** ASD-STE100 Plain English & Rigorous Scientific Falsifiability  
**Review Date:** 2026-11-01  

---

## 1. Executive Summary & Agent Decision Framework

This document provides a formal, evidence-grounded blueprint for prospective architectural enhancements to the HammerTime Spatio-Temporal Graph Neural Network (`HammerTimeSTGNN`).

### Purpose for Autonomous Agents
When an autonomous agent (or research engineer) is tasked with improving model predictive performance, physics compliance, or inference latency, it must not execute ad-hoc code modifications. Instead, the agent must:
1. **Read** this specification to understand existing structural bottlenecks and relevant literature.
2. **Evaluate** the decision criteria (*"Can it be implemented?"* and *"Should it be implemented?"*).
3. **Select** a single enhancement candidate based on project constraints (compute budget, dependencies, latency).
4. **Follow** the frozen evaluation protocol: establish a baseline manifest, run controlled ablation experiments, and verify against authentic telemetry gates without tuning against the held-out test split.

```
                    ┌───────────────────────────────┐
                    │  Agent Decision Flowchart     │
                    └───────────────┬───────────────┘
                                    │
                     Is new external data required?
                                    │
                    ┌───────────────┴───────────────┐
                   YES                              NO
                    │                               │
       Does authentic source exist?     Can it run within compute?
        (e.g., track map geometry)             (CPU/PyTorch)
                    │                               │
            ┌───────┴───────┐               ┌───────┴───────┐
           YES              NO             YES              NO
            │               │               │               │
       [Proposal 4]      REJECT        Is it high ROI?   DEFER/PRUNE
        Vectorized                    (e.g. Edge ECC)
         Circuit                            │
                                    ┌───────┴───────┐
                                   YES              NO
                                    │               │
                               [Proposal 1]    [Proposal 3]
                               Physical Vector Joint ST-Transf.
                                  Edge ECC      (Compute Heavy)
```

---

## 2. Global System Invariants & Constraints

Any implemented change must preserve these repository invariants:
* **Authentic Data Provenance:** Only authentic 2023 session data from the verified multi-session cohort may be used. No synthetic data or fixtures.
* **Frozen Split Membership:** Train, validation, and test partition boundaries (temporal chronological split with 15-window boundary purging) must remain mathematically identical.
* **Locked Environment:** Must execute cleanly under `uv run --frozen` without adding heavy, unpinned third-party dependencies.
* **Physics Acceptance Gates:** Predictions must pass:
  1. `clean_air_max_mae <= 0.05 s` (predicted penalty in clean air $\approx 0$).
  2. `drs_train_min_lap_loss` and `drs_train_max_lap_loss` bounds ($0.10\text{ s} - 0.85\text{ s}$ cumulative dirty-air wake loss per lap).
* **Zero Magic Numbers:** All hyperparameters, thresholds, and schema flags must be declared in YAML configuration files and validated via `config_validation.py`.

---

## 3. Decision Matrix Summary

| Proposal | Primary Literature Basis | Computational Overhead | Implementation Complexity | Recommended Action |
| :--- | :--- | :--- | :--- | :--- |
| **P1: Multi-Dimensional Physical Edge Attributes (ECC)** | Simonovsky et al. (CVPR 2017)<br>Battaglia et al. (2018) | **Low** (+5% runtime) | **Low** (1–2 days) | **PRIORITY 1: IMPLEMENT FIRST** |
| **P2: Dual-Relation R-GCN (Wake + Pressure)** | Schlichtkrull et al. (ESWC 2018)<br>Kipf et al. (ICML 2018) | **Low** (+10% runtime) | **Medium** (2–3 days) | **PRIORITY 2: IMPLEMENT SECOND** |
| **P3: Multi-Scale Hierarchical Dilations** | Wu et al. (Graph WaveNet, IJCAI 2019)<br>Nie et al. (PatchTST, ICLR 2023) | **Very Low** (+2% runtime) | **Low** (1 day) | **PRIORITY 3: IMPLEMENT** |
| **P4: Vectorized Circuit Geometry Graph** | Liang et al. (LaneGCN, ECCV 2020)<br>Gao et al. (VectorNet, CVPR 2020) | **Medium** (+25% runtime) | **High** (requires track map vectorization) | **DEFER: DATA PREREQUISITE** |
| **P5: Joint Spatio-Temporal Cross-Attention** | Shi et al. (MTR, CVPR 2022)<br>Salzmann et al. (Trajectron++, ECCV 2020) | **High** (+80% CPU runtime) | **High** (Transformer migration) | **EVALUATE: REQUIRES GPU** |

---

## 4. Proposal 1: Multi-Dimensional Physical Edge Attributes (Edge-Conditioned Convolution)

### 4.1 Current Architectural Limitation
In `hammertime_100/src/model.py`, the edge attribute is a single scalar: Gaussian RBF of the temporal gap:
$$e_{ij} = \exp\left(-\frac{\Delta t^2}{2\sigma^2}\right) \in \mathbb{R}^1$$
* **Empirical Failure Mode:** A $0.8\text{ s}$ gap at $320\text{ km/h}$ on a Monza straight equates to a $71\text{ m}$ physical distance with strong aerodynamic slipstream (low drag). The same $0.8\text{ s}$ gap at $75\text{ km/h}$ in a Monaco hairpin equates to $16.6\text{ m}$ with zero aerodynamic turbulence. The scalar time gap conflates these two completely different physical regimes.

### 4.2 Literature Basis
* **Simonovsky, M., & Komodakis, N. (2017).** *Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs.* **CVPR 2017**, pp. 3693–3702. [arXiv:1704.02901](https://arxiv.org/abs/1704.02901).
  * *Mechanism:* Edge attributes are arbitrary continuous feature vectors $\mathbf{e}_{ij} \in \mathbb{R}^{D_e}$. A filter-generating network (FGN), parameterized as a multi-layer perceptron $\mathcal{F}: \mathbb{R}^{D_e} \to \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, dynamically predicts kernel weights for message passing:
    $$\mathbf{x}_i' = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \mathcal{F}(\mathbf{e}_{ij}; \mathbf{\Theta}) \mathbf{x}_j + \mathbf{b}$$
  * *Reported Numbers:* 15% reduction in graph error over fixed scalar kernels; handles irregular continuous spatial graphs.
* **Battaglia, P. W., et al. (2018).** *Relational Inductive Biases, Deep Learning, and Graph Networks.* [arXiv:1806.01261](https://arxiv.org/abs/1806.01261).
  * *Mechanism:* Edge update blocks compute $\mathbf{e}'_{ij} = \phi^e(\mathbf{e}_{ij}, \mathbf{v}_i, \mathbf{v}_j)$ before node aggregation, ensuring physical conservation laws hold across interacting bodies.

### 4.3 Proposed Mathematical Design
Expand the edge attribute from $\mathbb{R}^1$ to a 5-dimensional physical vector:
$$\mathbf{e}_{ij}(t) = \begin{bmatrix}
\Delta t_{ij} & \text{(Time gap to leader in seconds)} \\
d_{\text{physical}} & \text{(Euclidean/track distance in meters: } \Delta t_{ij} \times v_{\text{leader}}\text{)} \\
\Delta v_{ij} & \text{(Velocity delta: } v_{\text{follower}} - v_{\text{leader}}\text{)} \\
v_{\text{leader}} & \text{(Absolute speed of the flow generator in m/s)} \\
\mathbb{I}_{\text{DRS}} & \text{(Binary indicator: is the pair in a designated DRS activation zone?)}
\end{bmatrix} \in \mathbb{R}^5$$

In `model.py`, PyG's `GATConv(edge_dim=5)` natively supports multi-dimensional edge features by projecting $\mathbf{e}_{ij}$ into the attention logit computation:
$$\alpha_{ij} = \text{Softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W}\mathbf{x}_i \,\|\, \mathbf{W}\mathbf{x}_j \,\|\, \mathbf{W}_e \mathbf{e}_{ij}] \right) \right)$$

### 4.4 Agent Decision Criteria
* **Can it be implemented?**
  * **YES.** PyG `GATConv` already supports arbitrary `edge_dim`. `physical_graph.py` already contains preliminary schemas (`physical_v1`). Speed, gap, and velocity are already extracted in `aligned_telemetry.parquet`.
* **Should it be implemented?**
  * **YES (Strongly Recommended).** High expected performance gain with virtually zero compute overhead. Directly resolves the speed-dependent wake discrepancy across diverse circuits.

### 4.5 Implementation Steps
1. In `hammertime_100/src/physical_graph.py`: Update `build_dynamic_edges` to populate the 5D feature tensor for each active leader-follower pair.
2. In `hammertime_100/src/model.py`: Set `edge_dim = 5` when `edge_feature_schema == "physical_v1"`.
3. In `configs/`: Add `graph.edge_feature_schema: "physical_v1"` and `model.edge_dim: 5`.
4. Run ablation vs. `scalar_rbf`: Assert validation MAE improves by $\ge 5\%$ and physics tests pass.

---

## 5. Proposal 2: Dual-Relation Graph (Aerodynamic Wake + Defensive Pressure)

### 5.1 Current Architectural Limitation
Edges exist exclusively in the **Leader $\to$ Follower** direction.
* **Empirical Failure Mode:** Treating the interaction as strictly unidirectional assumes the leading car is oblivious to the follower. In Formula 1:
  * When a trailing car closes within $0.8\text{ s}$, the leader alters braking points, runs defensive lines (sacrificing apex exit speed), and deploys maximum ERS battery to defend position.
  * A unidirectional graph cannot model the lap-time penalty imposed on a leader who is under defensive pressure.

### 5.2 Literature Basis
* **Schlichtkrull, M., et al. (2018).** *Modeling Relational Data with Graph Convolutional Networks (R-GCN).* **ESWC 2018**, pp. 593–607. [arXiv:1703.06103](https://arxiv.org/abs/1703.06103).
  * *Mechanism:* Uses relation-specific transformation matrices $W_r$ for distinct edge types $r \in \mathcal{R}$:
    $$\mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} \mathbf{h}_j^{(l)} + W_0^{(l)} \mathbf{h}_i^{(l)} \right)$$
* **Kipf, T., et al. (2018).** *Neural Relational Inference for Interacting Systems (NRI).* **ICML 2018**. [arXiv:1802.04687](https://arxiv.org/abs/1802.04687).
  * *Reported Numbers:* 40% reduction in mean squared error when modeling multi-agent physical systems with asymmetric interaction modes (e.g. charged particles, flocking).

### 5.3 Proposed Mathematical Design
Define two distinct directed relation types $\mathcal{R} = \{\text{wake}, \text{pressure}\}$:
1. $e_{i \to j}^{\text{wake}}$: Directed from Leader $i$ to Follower $j$ (valid when gap $< 2.0\text{ s}$). Represents downforce reduction and dirty air.
2. $e_{j \to i}^{\text{pressure}}$: Directed from Follower $j$ to Leader $i$ (valid when gap $< 1.0\text{ s}$). Represents defensive driving pacing penalty.

In PyTorch Geometric, replace `GATConv` with `RGATConv` (Relational Graph Attention Network) or dual parallel GAT layers:
$$\mathbf{h}_i = \text{ReLU} \left( \text{GAT}_{\text{wake}}(\mathbf{x}, \mathcal{E}_{\text{wake}}) + \text{GAT}_{\text{pressure}}(\mathbf{x}, \mathcal{E}_{\text{pressure}}) + W_{\text{self}}\mathbf{x}_i \right)$$

### 5.4 Agent Decision Criteria
* **Can it be implemented?**
  * **YES.** Straightforward extension of graph construction in `graph.py`. Requires no new telemetry data.
* **Should it be implemented?**
  * **YES.** Addresses a documented physical omission in race modeling without inflating parameter counts.

---

## 6. Proposal 3: Multi-Scale Hierarchical Dilations (Micro Dynamics + Macro Stint Curve)

### 6.1 Current Architectural Limitation
The current temporal block uses fixed dilations $(1, 2, 4)$ across a $W=10$ snapshot window ($2.25\text{ s}$ at 4 Hz).
* **Empirical Failure Mode:** Race pace degradation occurs on three disconnected time horizons:
  1. *Micro ($0.25\text{ s} - 2\text{ s}$):* Throttle micro-adjustments, wheelspin, braking deceleration.
  2. *Meso ($5\text{ s} - 20\text{ s}$):* Sector dirty air accumulation, tire surface overheating.
  3. *Macro ($60\text{ s} - 90\text{ s}$):* Stint-level tire carcass degradation, fuel burn weight shedding (~$1.7\text{ kg/lap} \approx 0.05\text{ s/lap}$ pace gain).
* Currently, `tyre_age` is passed only as an instantaneous static scalar, forcing the 1D convolution to model long-term linear trends with high-frequency weights.

### 6.2 Literature Basis
* **Wu, Z., et al. (2019).** *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* **IJCAI 2019**, pp. 1907–1913. [arXiv:1906.00121](https://arxiv.org/abs/1906.00121).
  * *Mechanism:* Stacks causal 1D dilated convolutions with exponential dilation factors ($1, 2, 4, 8, 16$) to expand receptive fields across multiple orders of magnitude without recursive bottlenecks.
* **Nie, Y., et al. (2023).** *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* **ICLR 2023**. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730).
  * *Reported Numbers:* Multi-scale temporal subseries patching achieves a 21.7% MSE reduction on traffic benchmarks by separating high-frequency point noise from macroeconomic trends.

### 6.3 Proposed Mathematical Design
Implement a dual-stream temporal encoder:
1. **Dynamic Stream:** Operates on the 4 Hz window ($W=10$) with `CausalGatedResidualBlock(dilation=1, 2, 4)`.
2. **Contextual Stint Stream:** Operates on a subsampled stint history tensor (e.g. 1 sample per lap across the last 10 laps) encoding tire degradation and fuel burn curves via a 1D MLP:
$$\mathbf{z}_{\text{fused}} = \text{Concat}\big[\mathbf{z}_{\text{dynamic}}(t), \; \text{MLP}_{\text{stint}}(\mathbf{s}_{\text{stint}})\big]$$

### 6.4 Agent Decision Criteria
* **Can it be implemented?**
  * **YES.** Stint lap counts and tire ages are already tracked in `aligned_telemetry.parquet`.
* **Should it be implemented?**
  * **YES.** Low computational cost (+2% runtime), solves tire degradation drift during long green-flag stints.

---

## 7. Proposal 4: Vectorized Circuit Geometry Graph (LaneGCN / VectorNet)

### 7.1 Current Architectural Limitation
Circuit context is represented solely by an integer category `mini_sector_idx` $\in \{0 \dots 19\}$.
* **Empirical Failure Mode:** The network cannot generalize across tracks. A model trained on Monza has no representation of track curvature, corner radius, braking severity, or straight length. When transferred to Spa or Silverstone, it must relearn sector behavior from scratch.

### 7.2 Literature Basis
* **Gao, J., et al. (2020).** *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation.* **CVPR 2020**, pp. 11525–11533. [arXiv:2005.04259](https://arxiv.org/abs/2005.04259).
  * *Mechanism:* Encodes track centerlines, boundaries, and paths as polylines within a unified graph. Subgraphs encode local geometry, and global cross-attention models vehicle-to-track interactions.
* **Liang, M., et al. (2020).** *Learning Lane Graph Representations for Motion Forecasting (LaneGCN).* **ECCV 2020**, pp. 541–556. [arXiv:2007.13732](https://arxiv.org/abs/2007.13732).
  * *Reported Numbers:* Bipartite vehicle-to-lane attention achieves state-of-the-art motion prediction on Argoverse, reducing displacement error by 19% compared to rasterized raster map baselines.

### 7.3 Proposed Mathematical Design
Represent each circuit as a static spatial graph $\mathcal{G}_{\text{track}} = (\mathcal{V}_{\text{track}}, \mathcal{E}_{\text{track}})$:
* $\mathcal{V}_{\text{track}}$: Nodes placed every 50 meters along the track centerline, with features:
  $$\mathbf{v}_{\text{track}} = [x, y, \text{curvature } \kappa, \text{gradient}, \text{track\_width}, \mathbb{I}_{\text{braking\_zone}}, \mathbb{I}_{\text{drs\_zone}}]$$
* Interaction: Vehicle nodes perform bipartite cross-attention into local track nodes:
  $$\mathbf{h}_i^{\text{spatial}} = \text{CrossAttention}(\mathbf{q}=\mathbf{x}_i, \; \mathbf{k}=\mathcal{V}_{\text{track}}, \; \mathbf{v}=\mathcal{V}_{\text{track}})$$

### 7.4 Agent Decision Criteria
* **Can it be implemented?**
  * **CURRENTLY BLOCKED / REQUIRES PRE-WORK:** FastF1 provides approximate $(X, Y, Z)$ telemetry coordinates, but full circuit centerline and curvature vectorization is not yet compiled into standardized parquet tables for all 22 circuits.
* **Should it be implemented?**
  * **DEFER:** Do not implement until an authentic, version-controlled track map database (`circuit_geometry_v1.parquet`) is established and verified.

---

## 8. Proposal 5: Joint Spatio-Temporal Cross-Attention (Motion Transformer)

### 8.1 Current Architectural Limitation
Factorized 2D GAT + 1D TCN assumes spatial interaction is invariant over the window time axis.

### 8.2 Literature Basis
* **Shi, G., et al. (2022).** *Motion Transformer (MTR) for Autonomous Driving with Small-to-Large Vectorized Representation.* **CVPR 2022**. [arXiv:2209.13508](https://arxiv.org/abs/2209.13508).
  * *Mechanism:* Flatten all $B \times N \times W$ tokens into a single spatio-temporal sequence with factorized axial attention (temporal self-attention across frames, cross-agent attention across cars).
  * *Reported Numbers:* 18.4% improvement on Waymo Open Motion Challenge.

### 8.3 Agent Decision Criteria
* **Can it be implemented?**
  * **YES, BUT EXPENSIVE.** Attention complexity scales quadratically with sequence length $\mathcal{O}((N \cdot W)^2) = \mathcal{O}((20 \cdot 10)^2) = 40,000$ operations per window.
* **Should it be implemented?**
  * **NO (Under Current CPU Hardware):** On CPU, full spatio-temporal self-attention would increase training time by $4\times$ to $6\times$. Revisit only when CUDA GPU acceleration is available.

---

## 9. Verification & Acceptance Gate Checklist for Agents

Before merging any architectural implementation, the executing agent must verify:
1. [ ] **Unit Contract Tests:** Pass all 225 unit tests under `uv run --frozen python -m unittest discover -s hammertime_100/tests`.
2. [ ] **Zero Split Leakage:** Assert split signatures match `split_membership_sha256` exactly.
3. [ ] **Clean Air Physics Gate:** Validation clean-air MAE $\le 0.05\text{ s}$ (`validate_clean_air_physics`).
4. [ ] **DRS Train Physics Gate:** Validation cumulative DRS lap loss between $0.10\text{ s}$ and $0.85\text{ s}$ (`validate_drs_train_physics`).
5. [ ] **Controlled Baseline Comparison:** Candidate validation MAE must strictly outperform the parameter-matched baseline across $\ge 3$ fixed seeds (`42, 43, 44`).
6. [ ] **Provenance Documentation:** Update `RESEARCH_CHANGELOG.md` with commit hash, parameter counts, runtime, and validation deltas.
