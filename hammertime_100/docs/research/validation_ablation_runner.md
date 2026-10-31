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
3. `temporal_kernel_1`: current graph and kernel 1.
4. `temporal_kernel_10`: current graph and kernel 10.
5. `constant_edge_attribute_k3`: preserve edge topology and set each stored
   edge attribute to 1.0; the model's self-loop attribute is also 1.0.

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
The zero-prediction baseline uses those same validation labels and is context
only; it is not a candidate or selection input.

A capped pilot is always marked incomplete and has no selected candidate. A
complete study requires every candidate across at least three matched seeds.
The predeclared selection metric is mean validation MAE, with candidate order
breaking ties. Complete reports include each seed's validation MAE, mean and
spread, runtime, and parameter count; seed repetitions are not independent race
replications.

Prior project reports expose held-out test outcomes. This runner never receives,
iterates, or scores that test loader. No test-set performance claim or
validation-selected checkpoint evaluation is produced by the ablation runner.

No authentic ablation has been run by this implementation note. Before an
experiment, verify authentic source hashes and prepared-data membership, freeze
the generated preregistration manifest, and record any incomplete pilot as
throughput evidence only.
