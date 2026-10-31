# ST-GNN primary literature and controlled study design

**Status:** Proposed design; not registered and no experiments reported.  
**Review date:** 2026-10-31.  
**Study registration condition:** Freeze a manifest after authentic preprocessing has been verified. Until then, source hashes, code revision, split membership, usable sample counts, seeds, runtime budget, and results are intentionally pending.

This document translates four primary spatiotemporal graph papers into falsifiable HammerTime hypotheses. Their published benchmark results are not evidence of F1 performance. It does not authorize a dependency, target, split, metric, threshold, or acceptance-gate change.

## Current HammerTime contract and limits

The active model in `src/model.py` projects each snapshot's node features, applies two `GATConv` layers independently at each time, stacks the embeddings, applies one causal `Conv1d`, and predicts from its last output. The default configuration uses a 10-snapshot window (`W=10`), five-snapshot horizon (`K=5`), 4 Hz samples, and temporal kernel 3. Left padding preserves causality, but this one convolution's final output depends only on the most recent three embeddings. Those samples span two 0.25-second intervals (0.5 s from earliest to latest), even though the supplied window covers 2.25 seconds between its first and last timestamp. The distinction motivates an ablation; it does not establish that longer context improves the target.

`src/graph.py` constructs directed leader-to-follower edges for observed gaps within the configured cutoff. Its edge attribute is the Gaussian transform of the gap. The model passes this scalar to PyG `GATConv(edge_dim=1)`. In PyG's documented operator, edge features contribute a learned term to the attention logits; softmax normalizes the coefficients over incoming neighbors, and those coefficients then weight source-node messages. Thus the scalar is an attention feature, not a direct multiplication of the message by the raw Gaussian value. A constant-edge ablation tests the incremental value of that feature while preserving the topology.

The configured horizon forecasts a future sector-entry timestamp. The label is sector traversal duration minus a clean-air reference, with the duration outcome completing at sector exit. The current clean-air reference uses observations from the whole session, including held-out periods; it is session-calibrated and does not support a strict prospective holdout claim. Window purging does not itself prove that duration outcomes remain within a partition; this remains an audit question. A single race can support initial within-race development, not race-held-out generalization. These questions must be reported and not silently resolved by this model study.

## What the papers contribute

### DCRNN

Li et al., “Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting,” ICLR 2018. [Primary paper](https://arxiv.org/abs/1707.01926).

**Mechanism.** Diffusion convolution represents spatial dependence on a directed road graph using forward and reverse random walks. A recurrent sequence-to-sequence encoder-decoder models temporal dependence; scheduled sampling mixes ground truth and prior predictions during training to reduce autoregressive train/inference mismatch. The paper compares removing diffusion convolution and using one-way versus bidirectional walks, and compares fixed-history prediction with seq2seq and scheduled sampling.

**Fit and mismatch.** HammerTime predicts one per-car sector residual from a short fixed history rather than autoregressively emitting a long traffic sequence. Cars are moving nodes with changing order, and the implemented graph represents only leader-to-follower proximity. Reverse random walks on a road network are not the same as traffic flow from follower to leader; blindly adding reverse edges can communicate context against the intended edge semantics. The paper's large road-sensor datasets and speed forecasts do not validate F1 sector-time residuals.

**Falsifiable hypothesis.** If graph directionality contains useful predictive information beyond node histories, an explicitly defined direction-aware graph operator should improve validation performance over a parameter-matched self-history model, without requiring reverse messages.

**Controlled test and decision evidence.** First run the simpler self-loop-only ablation below. Only if neighbor information helps, preregister a separate direction experiment with parameter-matched alternatives and frozen topology semantics. Retain direction-specific complexity only if its declared validation objective improves across matched seeds and the gain is not explained by parameter count or compute. Reject or defer it if the evidence is inconsistent, data coverage is inadequate, or the study is underpowered. Do not infer causal traffic or wake effects.

### STGCN

Yu, Yin, and Zhu, “Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting,” IJCAI 2018. [Primary paper and proceedings](https://ijcai.org/proceedings/2018/505), [DOI](https://doi.org/10.24963/ijcai/2018/505).

**Mechanism.** STGCN stacks spatiotemporal blocks with causal gated temporal convolutions (gated linear units) around a spatial graph-convolution layer. Its spatial operator uses a fixed weighted road graph; temporal convolutions use multiple observations and are structured for parallel computation.

**Fit and mismatch.** This motivates measuring the temporal receptive field before adding a larger temporal block: HammerTime currently supplies ten frames but its final prediction uses only three embeddings. However, STGCN's road graph is fixed, its sensor observations are sampled at five-minute intervals, and its targets are traffic speed. HammerTime has a changing proximity graph, 20 cars at 4 Hz, and a sector-duration residual. The paper's reported accuracy and efficiency do not establish a beneficial F1 architecture.

**Falsifiable hypothesis.** With all inputs, windows, horizon, targets, splits, and training rules fixed, allowing the final operator to use longer available history (kernel 10 versus kernel 3) will reduce validation error if relevant temporal information lies outside the current 0.5-second span.

**Controlled test and decision evidence.** Compare temporal kernel sizes 1, 3, and 10 using the same causal implementation and matched seeds. Record parameter counts and runtime, since kernel changes affect capacity. Retain a longer kernel only under the preregistered validation selection rule and consistent matched-seed evidence. A gated STGCN-like block is a later, separately registered study only if these results show that longer history is useful; kernel 10 alone does not validate the full STGCN design.

### Graph WaveNet

Wu et al., “Graph WaveNet for Deep Spatial-Temporal Graph Modeling,” IJCAI 2019. [Primary paper](https://arxiv.org/abs/1906.00121).

**Mechanism.** Graph WaveNet learns an adaptive dependency matrix from node embeddings to represent spatial relationships absent from the supplied graph. It pairs graph modeling with stacked dilated causal convolutions; exponentially increasing dilations expand the temporal receptive field without recurrent iteration.

**Fit and mismatch.** The dilation mechanism motivates a later test if the kernel study supports longer context. Learned node dependencies could also expose missing relationships, but HammerTime's 20 node slots are assigned from observed driver ordering and may encode driver identity or race-specific structure. The observed graph changes each snapshot. Graph WaveNet's road-sensor sequence benchmark is not evidence that learned dense dependencies transfer to a small, dynamic F1 field.

**Falsifiable hypothesis.** A causal dilated block with a frozen receptive field and a capacity-matched control may improve validation error if dependencies extend beyond the present three snapshots.

**Controlled test and decision evidence.** Do not add an adaptive adjacency in the initial ablation. If temporal kernels show a reliable longer-history benefit, register a separate design comparing a dilated temporal block against a nondilated control with matched receptive field, parameter budget, data, and seeds. Any adaptive graph test must separately test its contribution against the observed dynamic graph and guard against slot/driver identity effects. Retain only if validation evidence satisfies the frozen rule, with runtime and parameter costs reported; otherwise reject or leave unmeasured. No test-set tuning.

### Graph Attention Networks

Veličković et al., “Graph Attention Networks,” ICLR 2018. [Primary paper](https://arxiv.org/abs/1710.10903).

**Mechanism.** Masked self-attention computes learned coefficients over each node's supplied neighborhood; multi-head attention uses independent attention projections and combines their outputs. The paper studies graph node classification on citation and protein-interaction benchmarks.

**Fit and mismatch.** It supports learned neighbor aggregation as a general operator family. It does not evaluate temporal forecasting, F1 telemetry, leader/follower direction, Gaussian gap features, or sector duration. The present model uses PyG `GATConv`; PyG documents edge-feature contribution to the attention logits, followed by neighborhood softmax and attention-weighted messages. This is a narrower statement than saying the RBF attribute directly scales a message. The attention paper does not establish that HammerTime's topology or edge values help.

**Falsifiable hypothesis.** Other-car messages improve prediction over self-history, and the gap-derived edge feature contributes beyond the same directed topology.

**Controlled test and decision evidence.** The initial self-loop-only and constant-edge-attribute controls test these two claims separately. Retain graph aggregation or the gap feature only when the predeclared validation selection rule supports it across matched seeds; otherwise retain the simpler control for future work or report the result as inconclusive. Attention coefficients are not causal effects.

## Proposed controlled ablation

This is a study design, not a registered experiment. **Before any fitting**, verify authentic preprocessing, then freeze and persist a manifest containing:

- raw-source identity and content hashes, source/provider provenance, coverage, label counts, and preprocessing outputs;
- Git commit and clean/dirty state, effective config and its hash, software/package versions, and checkpoint initialization policy;
- exact train/validation/test sample membership and split method, plus label/outcome partition-boundary audit findings;
- candidate list, matched seed list, training and early-stop rules, validation objective, tie-break rule, parameter/runtime measurement method, and total compute budget;
- prior test-set exposure status and a statement of which test claims are therefore permissible.

Do not start the study until this manifest is frozen. Keep the same authentic samples, node features, target construction, `W=10`, `K=5`, chronological or already-approved race partition membership, scaler, metrics, gates, and physics thresholds for every candidate. Do not use fixtures as race results. Use validation results only for candidate selection.

| Candidate | Change from current model | Question |
|---|---|---|
| Current model | Two GAT layers plus causal temporal kernel 3 | Reference implementation |
| Self-loop-only | Remove all inter-car edges; preserve node features, self-loops, temporal block and head | Do other-car messages help? |
| Temporal kernel 1 | Change only causal temporal kernel to 1 | Is same-frame embedding sufficient? |
| Temporal kernel 3 | Current configured kernel | Reference for temporal-context comparison |
| Temporal kernel 10 | Change only causal temporal kernel to 10 | Does usable longer history help? |
| Constant edge feature | Preserve directed edges and counts, replace each non-self edge attribute with one fixed constant | Does the gap-derived scalar add value beyond topology? |

Specify the constant and the treatment of self-loop attributes in the frozen manifest; keep them identical across runs. Keep the 10-frame input window even when kernel 1 or 3 reads a shorter suffix, so only the model's temporal receptive field changes. Match seeds across all candidates. Seed repetitions quantify optimization variability, not independent race replications.

Report each candidate's validation objective, all existing validation metrics, matched-seed results and spread, trainable parameter count, runtime under a stated hardware/software setup, convergence/failure status, and any missingness/coverage differences. Do not introduce new gates or change existing thresholds. The initial decision is based only on the predeclared validation objective and tie-break rule. A candidate is retained only if it satisfies that rule, its matched-seed evidence is not an isolated favorable run, and the added model/compute cost is documented and acceptable under the frozen budget. Otherwise reject it or mark the evidence inconclusive; never tune against test results.

Evaluate a selected checkpoint on the test set once only if the frozen manifest confirms the set has not already informed model or protocol decisions and the evaluation is appropriate to the stated claim. If exposed, report that limitation and do not call the test untouched. With one race, describe results as within-race development evidence, not race-held-out generalization. Report the session-wide clean-air calibration and unresolved outcome-boundary issue when describing temporal holdout results. Predictive associations do not establish causal traffic effects.

## FastF1 archive access: dated verification

As checked on **2026-10-31**, FastF1's official GitHub release page lists v3.9.0, released 2026-10-24. Its release notes say Formula 1 added authentication requirements to data APIs in October 2026 and FastF1 now supports F1TV Access or higher account login. The official authentication documentation says an active F1TV Access/Pro/Premium subscription is required for most available data, describes local token storage, and says tokens are usually valid for four days. The official PyPI release page also lists 3.9.0 as latest on the check date.

The project lockfile still resolves FastF1 3.8.3. This research note does not upgrade the dependency or lockfile. Before a separate reviewed upgrade, test the installed 3.9.0 APIs, cache behavior, and offline operation using the locked project workflow; keep credentials out of logs and source control. Existing caches remain source artifacts and must not be overwritten during any compatibility test.

Sources: [FastF1 v3.9.0 release notes](https://github.com/theOehrly/Fast-F1/releases), [official authentication documentation source](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/accounts_auth.rst), [FastF1 3.9.0 on PyPI](https://pypi.org/project/fastf1/3.9.0/).

## References

- Li, Y., Yu, R., Shahabi, C., and Liu, Y. (2018). “Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.” ICLR. [arXiv:1707.01926](https://arxiv.org/abs/1707.01926).
- Yu, B., Yin, H., and Zhu, Z. (2018). “Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.” IJCAI, 3634–3640. [Proceedings](https://ijcai.org/proceedings/2018/505), [DOI](https://doi.org/10.24963/ijcai/2018/505).
- Wu, Z., Pan, S., Long, G., Jiang, J., and Zhang, C. (2019). “Graph WaveNet for Deep Spatial-Temporal Graph Modeling.” IJCAI. [arXiv:1906.00121](https://arxiv.org/abs/1906.00121).
- Veličković, P. et al. (2018). “Graph Attention Networks.” ICLR. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).
- PyTorch Geometric. [`GATConv` API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html).
