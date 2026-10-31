# Validation-only ST-GNN ablation runner

`src/ablation.py` implements the proposed graph and temporal ablations without
accepting a test loader. It consumes already-created training and validation
loaders, so the caller must construct the existing split and fit the existing
training-only scaler once before invoking it. Target values, window size,
horizon, split membership, scaler, thresholds, and gates remain fixed. The
runner requires the reference `W=10`, `K=5`, and temporal kernel 3 configuration.
It accepts explicitly selected FastF1 or OpenF1 provenance, verifies the
prepared artifact and manifest hashes, and rejects cohorts mixing providers.

The fixed candidate order is:

1. `current_k3`: configured leader/follower graph and kernel 3.
2. `self_loop_only_k3`: remove stored inter-car edges; HammerTime inserts one
   self-loop per node before each GAT layer. Both `GATConv` layers have internal
   self-loop insertion disabled.
3. `temporal_kernel_1`: legacy temporal mode, current graph, and kernel 1.
4. `temporal_kernel_10`: legacy temporal mode, current graph, and kernel 10.
5. `constant_edge_attribute_k3`: legacy temporal mode; preserve edge topology and set each stored
   edge attribute to 1.0; the model's self-loop attribute is also 1.0.
6. `gated_dilated_full_history`: replace only the temporal module with three
   gated residual causal convolutions using dilations 1, 2, and 4. Their
   receptive field is 15 frames, covering the fixed 10-frame input. Train with
   the default masked MAE objective.
7. `gated_dilated_mse`: keep the same full-history model and graph as candidate
   6, but train with masked mean squared error. Candidates 1–6 explicitly use
   masked MAE.

Runs are registered before fitting. The manifest records the effective config,
Git revision and dirty state, source provenance, hashes of training and
validation tensor contents, a separate train/validation membership hash, split metadata, `uv.lock`,
relevant source modules, package versions, device, seeds, candidate order,
epoch/run limits, test exposure, and metric/selection definitions. The tensor
hash is computed from dtype and contiguous tensor bytes rather than serialized
Python objects. It also records and hashes the shared training-fitted feature
scaler. The runner records existing validation MAE and R², trainable
parameter count, runtime, epochs, and checkpoint path. It reloads the best
validation checkpoint before reporting metrics and verifies the saved MAE.
The `train.loss` setting accepts `mae` (the default) or `mse`; the MSE option
emphasizes large residuals and can worsen MAE. Checkpoint selection and early
stopping remain based on validation MAE for both objectives. R² remains a
reported metric, not an objective gate, so MSE training does not guarantee a
higher R².
The zero-prediction baseline uses those same validation labels and is context
only; it is not a candidate or selection input.

Prepared config compatibility allows only `model.temporal_mode`,
`model.temporal_dilations`, and isolated cache/output/checkpoint paths to differ
from the source artifact. Data, targets, scaler, split, and training settings
must match the prepared config.

A capped pilot can register a specific candidate index, is always marked
incomplete, and has no selected candidate. A
complete study requires every candidate across at least three matched seeds.
The loader split and training-fitted scaler are prepared once from the frozen
configuration. Run seeds control model and optimizer RNG after loader creation;
they may differ from `data.random_seed` without changing the split. Batched
forward passes use one disjoint PyG graph batch for all windows and frames,
with no cross-graph edges; evaluation outputs match the per-window reference.
Batching changes dropout RNG draw order during training, so it does not preserve
the exact old stochastic training trajectory or imply a predictive gain. The
predeclared selection metric is mean validation MAE, with candidate order
breaking ties. Complete reports include each seed's validation MAE, mean and
spread, runtime, and parameter count; seed repetitions are not independent race
replications.

Prior project reports expose held-out test outcomes. This runner never receives,
iterates, or scores that test loader. No test-set performance claim or
validation-selected checkpoint evaluation is produced by the ablation runner.

The local record contains a one-epoch, seed-42 CPU `current_k3` throughput pilot
only. It is incomplete, has no candidate selection, and was scored on validation
data only; see `.local/research-history/2026-10-31/ablation-pilot.log`. No
comparative ablation result is established. Record future capped pilots as
throughput evidence only.
