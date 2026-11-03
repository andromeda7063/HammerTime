# HammerTime project overview deck

`HammerTime_Project_Overview_2026_Registry.pptx` is the current eleven-slide editable project overview. It covers the residual target, graph-model flow, 2023 FastF1 race-held-out evaluation, matched baseline, per-race results, acceptance gates, separate OpenF1 model-family evidence, the dashboard and the experiment registry.

`HammerTime_Project_Overview_2026.pptx` is the earlier ten-slide version. The registry update is a new file so it can be reviewed alongside the prior deck.

## Evidence boundaries

- The FastF1 season test comparison includes only the ST-GNN and its matched zero predictor on the same held-out targets.
- The OpenF1 Hungary table is a separate validation-only suite with a different provider, split, and target. Its scores are not directly comparable with the FastF1 season test.
- The sector target uses full-session clean-air calibration, so the current evaluation is retrospective rather than strictly prospective.
- The DRS check is unmeasurable because the held-out set contains no complete qualifying lap.
- The dashboard shows archived metrics and a deterministic sample of saved predictions. It does not run checkpoint inference in the browser.

## Rebuild

The source is `build_deck.mjs`. Run it with the desktop bundled Node executable. It resolves `@oai/artifact-tool` from the bundled runtime directly, so the report directory needs no `node_modules` link. Both authentic frontend screenshots are tracked in `assets/`; rebuilds do not depend on ignored `.local` screenshots. Build previews and validation records remain under `.local/research-history/2026-11-02/presentation-build/`.

```bash
/home/andromeda/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node reports/HammerTime_Project_Overview_2026/build_deck.mjs
```

The deck's speaker notes identify source artifacts for each slide. It contains an editable MAE chart and evidence tables. The registry slide reports 23 records across 9 groups (21 with scores and 2 failed/no-score attempts) and explains that evidence groups are not globally ranked.
