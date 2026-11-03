# GNNv3 CPU overnight candidate

This is one coordinated validation run for the Linux 16 GiB / Intel i5-13420H host. It keeps the prior GNNv2 prepared cohort, race-held-out split, target, coverage rules, metrics, gates and physics thresholds fixed. It does not promise an overnight completion time: the prior three-hour estimate is for another model/runtime and CPU threads do not scale linearly. The cap is 45 epochs with validation-MAE selection and patience 12.

## Protect and identify the production model first

The production GNNv2 checkpoint must be identified explicitly before the candidate starts. Set `HAMMERTIME_OLD_RUN` to the exact old run output and inspect it read-only:

```bash
uv run --frozen python hammertime_100/tools/inspect_production_checkpoints.py "$HAMMERTIME_OLD_RUN"
```

Review each `best_checkpoint` and metadata against the reported production run (best epoch 18, completed through epoch 28, R² near 0.09). This helper only scans `training/*/best.pt` under the one supplied directory; it reads checkpoint metadata with restricted `weights_only=True` loading and nearby small JSON metrics. If the candidate is ambiguous or unavailable, stop and resolve the exact best checkpoint path. Set `HAMMERTIME_PROD_BEST` to that exact `best.pt`. If its sibling `best.last.pt` exists, the runner discovers and snapshots that exact sibling automatically; an explicit path can also be supplied. The runner requires the best file, copies it into the new run output and verifies SHA256 before cache checks or fitting; it also preserves the final/last checkpoint when present. It records a manifest of source path, backup path, byte count and hash. It refuses a conflicting existing backup. The original checkpoint is read only.

## Candidate changes

`gnnv3_cpu_overnight_overrides.yaml` is applied only after the runner validates and loads the exact prepared cohort made with the GNNv2 config. It selects an attention-free, per-frame physical-edge message-passing encoder; keeps the 64-wide representation, causal dilated TCN, own-car branch, future-speed auxiliary head and objective; lowers node/edge dropout (edge dropout to zero); halves batch size to 128 for memory headroom; retains LR 5e-4; reduces weight decay from 5e-4 to 1e-4; and extends the epoch cap to 45/patience 12. This is a combined candidate, not an attribution experiment. It cannot identify which change caused any gain.

The edge-temporal encoder projects each node at each of ten observed frames, then applies the existing two-relation, edge-conditioned operator to that frame's physical-v2 edges. It keeps the frame sequence and passes it to the existing causal temporal stack. It skips three GAT attention passes and their per-edge softmax while retaining changing leader/follower and physical-edge inputs. Active-node masking, no self-relations, final-frame own-car correction, auxiliary future-speed output, loss and evaluation remain in place. The typed-message gates are learnable; inspect whether the interaction path learns before attributing a poor result to absent physics.

## Separate the source cache from all new outputs

The current run stores the normalized sessions and prepared cohort under `2023-gnnv2-cpu`, while its authentic raw sessions are in the sibling `2023-fastf1-season` run. Each normalized stage manifest records those raw source paths. The launcher therefore asks for both roots separately; its defaults are the verified paths on this machine. The runner hashes the authentic raw tables, validates all 22 normalized stage fingerprints, checks the immutable prepared-cohort manifest and artifact hash, and requires an exact cohort hit. It refuses to prepare a replacement cohort. All new files go to a separate `.local/research-history/` output.

The training overlay is strictly limited to selected `model` and `train` values. Do not pass CLI overrides for batch size, patience, GAT type, node/edge schema, neighbor mode, seed or window stride. The base `--epochs 30 --window-stride 2` values reproduce the original cohort config; after loading that cohort, the profile applies the 45-epoch cap. The old run root is read-only; all new config, logs, checkpoints, evaluation and backup files go under `HAMMERTIME_NEW_RUN`.

## Run

From this worktree root, first run the no-training preflight. It verifies the actual raw, normalized and prepared cache hashes and split/config identity, constructs the candidate model, and exits without unpickling the 2.9 GB cohort or preprocessing. It also makes and verifies a backup copy of the production best and sibling last checkpoint under a new ignored preflight output. The source checkpoints remain unchanged.

```bash
bash hammertime_100/tools/train_gnnv3_cpu_overnight.sh --preflight-only
```

The verified defaults on this machine are:

```text
run root:       /home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-gnnv2-cpu
raw sessions:   /home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-fastf1-season/race_cache/raw_sessions
production best:/home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-gnnv2-cpu/training/5408dd82d53ec69fbb65/best.pt
```

After a successful preflight, start the overnight run with the same launcher and no argument. It prompts with these values prefilled, prints the read-only checkpoint inventory, and backs up the production best and sibling last checkpoint before fitting. Twelve logical CPU threads fit this 12-thread CPU. PyTorch inter-op remains at one; data-loader workers remain zero to avoid duplicating the graph dataset; normalization remains a verified cache hit capped at one process.

```bash
bash hammertime_100/tools/train_gnnv3_cpu_overnight.sh
```

Do not omit the best checkpoint argument. A sibling `best.last.pt` is automatically preserved; pass `--preserve-production-last-checkpoint` only when its path differs from that sibling. If the exact caches are absent, do not remove `--require-existing-caches` to force a rebuild. A failed preflight should leave the old source run and production model unchanged. The profile writes the verified model copy and a checksum manifest under `production_model_backup/` before source validation or training. The launcher writes its log beside (not inside) the new run directory, so the runner can enforce that output is new/empty.

No duration or score is guaranteed. The profile cap of 45 epochs is not evidence that the old GNNv2 underfit; the reported production run's best epoch was 18. It only allows the materially different encoder enough time to reach its own validation plateau. The training history contains `train_online_mae`, computed from existing training batches before the optimizer step with dropout active; use it as a directional gap diagnostic, not an eval-mode training score.

## Interpret the result

Compare against the prior run only after confirming authentic race inventory, prepared-cohort SHA256, train/validation/test membership, target/scaler contract, and unchanged evaluation rules. Report old and candidate MAE/R², baseline scores, best epoch, train objective, train-online MAE, runtime, peak RSS/available RAM, CPU utilization, and exact run fingerprints. Do not select on test metrics or change existing gates. A lower training objective with flat validation error suggests generalization/feature/data limitations; high train error and validation error tracking together suggests optimization/capacity or target-noise limitations. These are diagnostic patterns, not proof of a unique cause.

Fit constant baselines using **training targets only**: train-mean predictions are an interpretable R² reference; train-median predictions are an MAE reference. Apply each unchanged to validation and test. The scored partition's own mean is only the mathematical R² denominator reference, not a deployable baseline.

`DeltaT_Sector` is measured in seconds. The target builder subtracts the clean-air mini-sector reference from future sector duration and applies the existing clip. R² = 0.09 means 91% of scored-target sum-of-squares remains relative to that partition's mean; it is not 9% accuracy. The existing target reference is built from session observations before the race split, so the task predicts a calibrated residual under the current protocol rather than a fully prospective pre-session baseline.

SmoothL1 beta 0.02 is in target seconds. It is quadratic for residual magnitudes below beta and linear outside; it does not directly optimize MAE. Keep the objective fixed in this one run. If the training/validation traces show optimization while MAE remains weak, a separate preregistered MAE or MSE comparison could use validation with unchanged metrics and gates. Overlapping temporal windows and repeated targets for adjacent cars/times are correlated; effective independent sample count is below raw window count. Do not reweight observations without a separate statistical design.

If this remains weak, the next low-cost diagnostic is a no-graph own-history predictor or boosted-tree baseline on the same split. This tests whether interactions add value. GCN is a simpler normalized-neighbor baseline, but may smooth this small directed dynamic graph. PNA could help if degree and aggregation statistics prove important. PairNorm is relevant only if embedding collapse is measured. Do not add these, wider channels, sample reweighting, a new loss or a hyperparameter sweep to this overnight run.

## References

- Veličković et al., [Graph Attention Networks](https://arxiv.org/abs/1710.10903).
- Brody, Alon and Yahav, [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).
- Kipf and Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
- Corso et al., [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718).
- Battaglia et al., [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222).
- Zhao and Akoglu, [PairNorm: Tackling Oversmoothing in GNNs](https://arxiv.org/abs/1909.12223).
- PyTorch, [SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html).
