# HammerTime — Assumption Ledger

**Document Standard:** ASD-STE100  
**Authority:** Law 10 (Master Directive)  

Every parameter tuned or assumed from `claude.md` and `claude_arc.md` is recorded in this ledger.

| ID | Parameter | Original Value | Final Value | File Location | Rationale |
|---|---|---|---|---|---|
| A01 | Gaussian RBF kernel scale ($\sigma$) | 1.0 s | 1.0 s | `configs/default.yaml` (`graph.rbf_sigma`) | Specified in `claude.md` §5.2. Smoothly decays edge weights between 0 s and 2.0 s. |
| A02 | Dirty air maximum gap cutoff | 2.0 s | 2.0 s | `configs/default.yaml` (`graph.edge_threshold_sec`) | Law 1 immutable rule. Empirical aerodynamic disturbance threshold for modern ground-effect cars. |
| A03 | Clean air threshold | > 3.0 s | 3.0 s | `configs/default.yaml` (`target.clean_air_gap_sec`) | Law 2 baseline rule. Unhindered baseline where aerodynamic wake decay is negligible. |
| A04 | Number of graph nodes ($N$) | 20 | 20 | `configs/default.yaml` (`graph.num_nodes`) | Represents the standard 20-car F1 grid. Degraded/retired cars are retained and masked. |
| A05 | Rolling history window ($W$) | 10 snapshots | 10 snapshots | `configs/default.yaml` (`data.window_size`) | Captures dynamic approach and wake disturbance build-up over 2.5 seconds (at 4 Hz). |
| A06 | Prediction horizon ($K$) | 5 snapshots | 5 snapshots | `configs/default.yaml` (`data.horizon`) | Predicts delay 1.25 seconds ahead into the next mini-sector entry. |
| A07 | Telemetry resample frequency | [ASSUMED] Not specified | 4 Hz (0.25 s) | `configs/default.yaml` (`preprocess.sample_hz`) | Balances high spatial fidelity across mini-sectors with ~5,000 snapshots per race. |
| A08 | Circuit mini-sector count ($M$) | [ASSUMED] Not specified | 20 equal segments | `configs/default.yaml` (`track.mini_sectors`) | Divides lap distance into discrete sectors for localized baseline and feature extraction. |
| A09 | GAT hidden channels & heads | 64 / 4 heads | 64 / 4 heads | `configs/default.yaml` (`model.hidden_dim`, `model.gat_heads`) | Balances expressiveness for 20 nodes with fast convergence (<15 min CPU budget). |
| A10 | Optimizer and learning rate | Adam / 0.001 | Adam / 0.001 | `configs/default.yaml` (`train.lr`) | Standard convergence setting with weight decay $10^{-4}$ (L2 regularization). |
| A11 | Baseline fallback hierarchy | [ASSUMED] Driver only | Driver -> Teammate -> Grid | `src/target.py` | Prevents NaN when clean laps are scarce for a specific driver in heavy traffic. |
| A12 | Missing telemetry gap tolerance | [ASSUMED] Not specified | 1.0 s maximum | `src/preprocess.py` (`preprocess.max_interpolate_sec`) | Gaps $\le 1.0$ s are linearly interpolated; larger gaps are marked invalid. |
