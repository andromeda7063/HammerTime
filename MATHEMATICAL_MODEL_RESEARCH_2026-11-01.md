# A stronger mathematical model for the current race-traffic predictor

**Research design note · 1 November 2026**

## Executive proposal

The current GNNv2 is a useful, carefully controlled graph-and-history baseline. A more coherent next model would treat each car as a partially observed dynamical system whose latent pace and interaction state evolve through sparse, directed messages from nearby cars. It would predict short-horizon state changes, especially speed increments, and derive a sector-duration distribution by integrating positive slowness over progress when an authentic future-sector label is available.

This is a research blueprint, not an implementation claim or a published F1 result. The near-term proposal is a deterministic recurrent graph state model with one or two sparse message-passing steps per time step. Keep the existing primary residual target, data filtering, splits, thresholds, metric definitions, and acceptance anchors fixed. Add speed-increment supervision first; test latent exposure and duration supervision in later, isolated ablations. Do not change deployment interpretation of the current residual target until its full-session baseline has been replaced by an as-of-available baseline under a separately reviewed protocol.

## What the current pipeline actually models

The committed GNNv2 model consumes windows of `W=10` graph snapshots and forecasts one primary value per node at the end of a `K=5`-step horizon. At 4 Hz, the last input is 0.25 seconds before the first future frame, while the target anchor is 1.25 seconds after the last input (3.5 seconds after the first input). The model does not receive future snapshots or future target-sector metadata as covariates. Dataset samples retain target MiniSector and exit-time metadata, but the loss uses the residual target and, when enabled, five future speed labels.

Each snapshot is projected to hidden width, passed through two GAT/GATv2 layers, optionally a typed edge-conditioned value operator, and then through a temporal module. The gated dilated causal option uses kernel size 3 and dilations `[1,2,4]`, for a receptive field of 15 frames; the input window is 10 frames. The final representation feeds the primary regression head. The optional future-speed head predicts five future speeds. The existing primary loss is masked SmoothL1 on `DeltaT_Sector` with beta 0.02 seconds; the speed auxiliary loss is masked SmoothL1 on speed divided by 100 kph with beta 1 and weight 0.05 in the GNNv2 configuration.

The current label is `actual_duration - baseline`, clipped to `[-2,10]` seconds. The baseline hierarchy is estimated using the full race session: same-race driver/compound/MiniSector, then team/compound/MiniSector, then grid/compound/MiniSector. The raw target source has `Duration`, `Baseline`, `SectorExitTimeSec`, `MiniSector`, `TimeSec`, and `DeltaT_Sector`; the dataset currently materializes the residual target plus some context metadata, not an independently trained duration target. The exit time is used for boundary/leakage diagnostics and is not a model input or duration loss.

The full-session baseline is retrospective. It can condition a research target but is not known at an arbitrary live cutoff. Consequently the current residual prediction must not be described as an operational estimate of `actual_duration - baseline` unless the experiment explicitly supplies that retrospective baseline as an oracle. The code’s current target is also clipped, so a later probabilistic model must not treat the boundary values as ordinary uncensored residual observations.

Inputs have their own as-of limitation. Telemetry is aligned to a 4 Hz grid using linear interpolation with bracketing samples; therefore some aligned historical inputs may use a raw sample later than the nominal grid time. The present pipeline is not strict online inference. Speed, throttle, tyre compound, tyre life, braking, DRS, accelerations, lap phase, curvature, and reference-derived features are among the inputs. The historical speed reference is indexed by circuit and MiniSector; held-out circuits can have no reference support, in which case the configured ratio/delta fallback is used. The position graph links the next car in circular track distance, with a directed leader-to-follower edge retained only for estimated time gaps in `(0,2]` seconds. Its five physical attributes are RBF, gap distance, leader-minus-follower speed, position coverage, and a wake-corridor indicator. The typed operator deliberately uses the same pair attributes for its reverse message stream; those attributes represent pair context, not generic source-minus-destination coordinates.

The cohort node count is not always 20: configuration starts at 20, while the season runner can pad to the number of distinct cohort drivers (22 in the audited 2023 cohort). All formulas below use `N` for this variable padded cohort size. These contracts are visible in `hammertime_100/src/model.py`, `dataset.py`, `physical_graph.py`, `prepare_cohort.py`, `train.py`, and `run_2023_fastf1_season.py` at commit `eb2515887b97fe8a31ef8fc5e87aa5d726522ee1` on `codex/gnnv2`.

## Proposed state-space formulation

Let `i=1,…,N` index cars and `t` index aligned observations. For each car, let `x_{i,t}` contain only telemetry available under a specified observation-latency policy. Let `s_{i,t}` be the registered progress coordinate used by the graph and let `C_i(s)` contain map covariates only if their provenance and pre-race availability are verified. Keep a binary observation/activity mask `m_{i,t}` distinct from the value of any measurement: an absent car or missing telemetry is unknown, not a physical zero.

Use three compact latent components per car:

- `b_{i,t}`: slowly changing pace/stint state;
- `h_{i,t}`: faster car-and-traffic interaction state;
- `w_{i,t}`: a decaying interaction-exposure proxy.

These are statistical states, not direct measurements of tyre temperature, wake turbulence, or fuel mass. A practical deterministic first version is preferable to full Bayesian filtering. For each directed physical edge `j→i`, form a relation-specific message

`r_{ji,t} = φ_r([h_{j,t}, h_{i,t}, e_{ji,t}, x_{j,t}, x_{i,t}, C_j, C_i])`,

where `r` identifies the directed relation/orientation and `e` is the current five-field physical edge context. A learned gate can control each message,

`a_{ji,t} = sigmoid(ψ_r([h_{i,t}, h_{j,t}, e_{ji,t}]))`,

and the receiver aggregate is a **sum**

`M_{i,t} = Σ_r Σ_{j∈N_r(i)} a_{ji,t} r_{ji,t}`.

The current physical graph retains the next car in circular progress order, so each node has at most one incoming neighbor in each directed relation. For a relation with at most one neighbor, sum and mean aggregation are identical. A sum may preserve multiplicity if a later graph deliberately admits multiple same-relation neighbors, but changing topology or aggregation must be evaluated as a separate ablation. Use sparse edges, explicit edge direction/type, and one or two message passes initially. Retain the current 2-second candidate cutoff in the first comparison so the experiment changes temporal integration, not the candidate graph and physics threshold at the same time. The signed leader/follower and reverse-pair semantics must stay explicit; do not infer a sign inversion merely because an edge is reversed when its values are stored as pair context.

A simple slow-state update can use a gated drift driven by current evidence,

`b_{i,t+1} = b_{i,t} + α_{i,t} tanh(f_b([b_{i,t}, x_{i,t}, C_i(s), pit_{i,t}]))`,

with `α` constrained to a modest range or learned through a slow transition. Do not force pace to move monotonically. A pit/stint reset may be used only when a pit event is observed as of that time; no future pit information may reset a past state.

For the exposure proxy, use a separate nonnegative edge-exposure signal, rather than applying softplus to the general signed message aggregate. For each relation define

`E_{i,t} = Σ_r Σ_{j∈N_r(i)} a_{ji,t} softplus(ψ_E^r([h_i,h_j,e_{ji},x_i,x_j]))`,

with `a≥0` and an empty sum defined as zero. Then use a stable exponential-memory update,

`ρ_{i,t}=exp(-Δt/τ_i)`, `τ_i=softplus(θ_i)+τ_min`,

`w_{i,t+1}=ρ_{i,t}w_{i,t}+(1-ρ_{i,t})E_{i,t}`, with `w_{i,0}≥0`.

This gives `0<ρ<1` and a nonnegative accumulated exposure state with learnable decay time. `τ` can be shared or predicted from current context, rather than tied to a driver identity, to support substitutions. When the neighborhood is fully observed and empty, `E=0` and the state decays exactly. If position/telemetry is missing, evidence is unavailable: carry or mask state using a declared missing-data rule; do not treat missing positions as an observed empty neighborhood or a measured zero wake. The state is an exposure summary only. Calling it wake, tyre thermal state, or a physical dose would require additional sensors and validation.

Update the fast interaction state with a recurrent cell,

`h_{i,t+1}=GRU_h(h_{i,t}, [x_{i,t}, M_{i,t}, w_{i,t}, b_{i,t}, C_i(s_{i,t})])`.

For an initial benchmark, use the same 10 observed frames and 5 predicted frames as GNNv2, and consider hidden widths 32–64 as a proposed small-model search range, not an established project range or runtime claim. Reset recurrent state at race/fold boundaries. If windows are shuffled for training, construct state only within each window; never carry a hidden state across unrelated samples. With only ten observed frames, a recurrent state does not by itself provide long-stint history; any longer memory would need an as-of-compatible initialization from a coarse history summary or a separately validated persistent-state protocol. During future rollout, use predicted states and controls or a declared control scenario, never measured future speed/throttle/edge features. Compare open-loop rollout against persistence and the current model; teacher-forced validation alone would conceal rollout error.

## Progress, speed, and duration

Let `u_{i,t}=ds_{i,t}/dt` denote speed in the registered progress coordinate. It need not equal the telemetry channel called `Speed`: registered distance may differ from the centerline arc length or from the car’s driven path. If the registration is validated, integrate predicted progress consistently:

`Δs_t = ∫_{t}^{t+Δt} u(τ)dτ ≈ (Δt/2)(u_t+u_{t+1})`.

The trapezoidal rule is a discrete approximation to the progress integral, not an exact identity. Validate the mapping between registered `Distance` and the chosen speed coordinate before weighting this consistency term strongly. Distance can itself be an integrated or processed telemetry product, so speed and distance losses should not be treated as statistically independent likelihood observations without a measurement model.

For the existing five-frame future speed labels, an output can be parameterized as an increment `Δv̂_{i,k}` with prediction `v̂_{i,t+k}=v_{i,t}+Δv̂_{i,k}` (or analogously in log speed). With the same scale and a translation-invariant loss such as SmoothL1, predicting this increment is mathematically a reparameterization of predicting absolute future speed; it supplies no extra labels and no automatic objective or accuracy gain. Its possible value is an explicit persistence initialization, an inductive bias if the architecture/loss uses it, and a clearer change-versus-persistence diagnostic. Test it against the equivalent absolute-speed head and persistence forecast. Train only with authentic, finite labels and the current activity/validity masks. Future MiniSector metadata remains a label only. At inference the model may receive a next-sector query only if it can be determined from current progress and an available map; otherwise it must jointly predict sector identity or produce a distribution for each candidate sector. Selecting the true future sector from the target table would leak information.

A duration decoder can express time per unit progress as a positive function,

`q_i(s)=dt/ds=q_ref exp(g_free(b_i,C_i(s))+g_int(h_i,w_i,M_i,C_i(s)))`,

where `q_ref` is a fixed seconds-per-distance reference unit and `g` is dimensionless. For a sector spanning progress `[s_k,s_{k+1}]`,

`T_{i,k}=∫_{s_k}^{s_{k+1}} q_i(s) ds ≈ Σ_ℓ Δs_ℓ q_i(s_ℓ) > 0`.

Differences between two durations can be signed, but the duration itself remains positive. These slow-pace and traffic components are not uniquely identifiable from the same observed duration labels: offsets can trade between `g_free` and `g_int`. Clean-air anchoring and slow-state priors are regularizers that may stabilize a decomposition; they do not prove a causal aerodynamic effect. This integral is a candidate decoder, not a newly discovered physical law or guarantee that current telemetry’s `Distance` is the correct path coordinate. Importantly, `E[1/u]` is not generally `1/E[u]`; uncertainty-aware duration predictions should integrate expected slowness or sample plausible paths and integrate each, rather than invert mean speed.

Authentic `Duration` must first be propagated from the current target-generation table into each training sample with exact sector identity, coverage, and valid exit. Some exits will lie beyond the present five-frame horizon. Do not create pseudo-duration labels from speed-through-entry or incomplete sector fragments. For a future duration head, train only on sectors whose actual exit is observed and whose progress interval and duration are covered. At inference, use a query known at the cutoff or predict sector identity jointly.

## Losses, uncertainty, and staged experiments

Keep the current residual SmoothL1 objective and all existing primary metrics as the compatibility anchor. A proposed first composite objective is

`L = L_residual + λ_v L_Δv + λ_s L_progress + λ_T L_duration + λ_c L_consistency + λ_b L_slow`,

where every term has its own valid-label mask and unit normalization. `L_progress` and `L_consistency` are conditional on validated registration; `L_duration` starts only after authentic duration labels are materialized. Choose new weights on validation data only. Do not double-count correlated speed, distance, and duration as if they were independent measurements. Keep the existing held-out race split, coverage rules, physics thresholds, tests, and acceptance anchors unchanged; report auxiliary metrics separately and do not select a new model by test performance.

Run a staged ablation with the same eligible data and fixed evaluation protocol:

| Stage | Change | Question |
|---|---|---|
| A | Existing GNNv2 | Compatibility anchor |
| B | Recurrent graph messages, with a residual output parameterization held fixed | Does recurrent graph state help open-loop prediction? |
| C | B plus slow pace and exponential exposure states | Do explicit latent components help beyond the recurrent state? |
| D | C plus duration integration with registered sector queries | Does authentic complete-duration supervision improve time prediction? |
| E | Separate prospective baseline/label protocol | Does an operational as-of residual task behave differently? |

Stages B–D preserve the current residual target and acceptance gates. Stage B should hold the output parameterization fixed so the graph-recurrence effect is identifiable; speed increments can be assessed as a separate equivalent-head ablation and diagnostic. Stage E is a separately reviewed benchmark: it would define `B_asof` from training/prerace or completed sectors only and predict `T-B_asof` (and raw `T` as needed). Changing the baseline changes the label meaning, so do not silently mix those scores with the existing retrospective residual benchmark. If a later residual likelihood is used, the current clipped labels require censored-tail treatment. For raw positive durations, a log-normal or another explicitly positive distribution is appropriate; for signed, **unclipped** residuals, a Student-t with `ν>2` is one option for finite variance. Do not put a Student-t on log-duration and claim a finite duration mean: exponentiating Student-t tails does not provide that guarantee. Point estimates should be chosen for the evaluation functional: conditional median aligns with MAE, while conditional mean aligns with squared-error risk; no model should be promised to improve both without evidence.

A lightweight deterministic cell with 1–2 sparse message passes and fixed-step rollout is the sensible first laptop-budget experiment. Avoid full dense covariance filters, expensive adaptive ODE solvers, and sampling-heavy Bayesian inference until a measured gain justifies their cost. This is a design recommendation, not an epoch-time estimate: no training, test, or benchmark has been run for this proposal.

## Research context and transfer limits

Cappello and Hoegh’s final 2026 paper studies latent tire pace, fuel, pit resets, and skew-t residuals across 19 races using rolling-origin validation; RMSPE was lower in 15 of 19 races. Fuel is derived rather than directly measured, and this result is not evidence for graph-conditioned multi-car or microsector prediction. It motivates latent stint state and race-wise validation, while leaving our graph and duration questions open. [Final 2026 article](https://journals.sagepub.com/doi/10.1177/22150218261446170).

Graph Network-based Simulators show that learned message-passing updates can model interacting physical systems and that rollout quality depends on training and message-passing design, but their evidence is particle simulation, not Formula 1 telemetry. [Sanchez-Gonzalez et al., ICML 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html). Neural Relational Inference learns latent interaction structure in synthetic systems and tracking tasks; inferred edges are useful hypotheses, not causally identified interactions. [Kipf et al., ICML 2018](https://arxiv.org/abs/1802.04687). These support sparse relational modeling as a method to test, not a claim that the present leader/follower graph captures aerodynamics.

Recurrent Kalman Networks show how factorized uncertainty can support scalable latent filtering, while Deep State Space Models use recurrently parameterized linear state dynamics. These are possible later extensions; the initial deterministic state should remain simpler and easier to audit. [Becker et al., ICML 2019](https://proceedings.mlr.press/v97/becker19a.html); [Rangapuram et al., NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html). Latent ODEs address irregular sampling with continuous-time latent dynamics, but solver cost and sensitivity make them a later option for this fixed 4 Hz grid. [Rubanova et al., NeurIPS 2019](https://arxiv.org/abs/1907.03907). Universal Differential Equations combine known physics with learned terms; without force, mass, tire-grip, and control measurements, a full race-car force model would be under-identified here. [Rackauckas et al., 2020](https://arxiv.org/abs/2001.04385).

Space-domain race-car control provides a useful mathematical precedent for integrating time through `dt/ds=1/v` when the path and velocity coordinate are defined. The cited minimum-lap-time work concerns an electric race car on a known path, not public F1 multi-car forecasting; our registered-progress coordinate needs its own validation. [Borsboom et al., IEEE TVT 2021](https://doi.org/10.1109/TVT.2021.3093164). Directed recurrent graph forecasting has also been studied for traffic networks, including scheduled sampling to address autoregressive rollout error; this is a transfer analogy, not a motorsport result. [Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926).

## Decisions and open questions before implementation

1. What observation-latency contract should define “available now,” and should aligned inputs be rebuilt with backward-only/as-of interpolation for a separate online evaluation?
2. Can the existing registered `Distance` be validated against a stable centerline/path coordinate, and is it available consistently for each car and target sector?
3. Which exact telemetry events identify pit entry, pit exit, tyre changes, and stint resets without using later session annotations?
4. Can complete `Duration` labels be joined to training windows with reliable sector identity, time coverage, and exit rules while preserving existing cohort filtering and splits?
5. Which future-sector query is actually known at the cutoff? If none, should the duration decoder jointly predict sector identity or output all candidate sectors?
6. What is the missing-position rule for exposure state: decay, uncertainty inflation, or explicit missingness gating? It must not interpret unavailable neighbors as zero exposure.
7. Should the primary operational target remain a retrospective benchmark only, while a distinct `B_asof` benchmark is introduced later?
8. Which state widths and auxiliary weights are affordable and useful on the target laptop? This requires measured, authentic-data experiments and cannot be inferred from this note.

The proposal becomes credible only if the as-of feature contract, target lineage, progress registration, and evaluation protocol are fixed first. Until then, the strongest safe next step is a controlled, source-auditable recurrent graph ablation with speed increments, leaving the established primary task and every acceptance gate intact.
