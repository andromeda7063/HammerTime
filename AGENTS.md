# Repository working instructions

- Use Luna for commands and code generation, as requested by the user.
- Preserve both `hammertime_50/` and `hammertime_100/`, including existing reports, datasets, caches, checkpoints, and user-created untracked files.
- Use the locked environment through `uv` and `--frozen` for project commands.
- Use authentic source data for research claims. Do not substitute synthetic data or present fixture results as race results.
- Keep existing metrics, data coverage rules, splits, acceptance gates, and physics thresholds unchanged unless the user explicitly requests a reviewed change.
- Keep substantive implementation and research summaries in tracked files. Preserve scientific assumptions and unresolved questions in `ASSUMPTIONS.md` and `OPEN_QUESTIONS.md`.
- Store local execution logs and archived run records under `.local/research-history/`; that directory is ignored by Git. Do not put credentials, environment files, or raw personal data in those logs.
- Record durable research and cleanup changes in `RESEARCH_CHANGELOG.md`. Avoid rewriting Git history or removing substantive implementation commits as part of document cleanup.
