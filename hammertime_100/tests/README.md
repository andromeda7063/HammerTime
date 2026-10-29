# Focused regression tests

Run from the repository root with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m unittest discover -s hammertime_100/tests
```

Install the declared environment with `uv sync --frozen` first. The thread limits
keep tiny CPU fixture runs efficient; they do not change model or gate settings.

The tests use small in-memory fixtures, mocked FastF1 sessions and temporary local
artifacts. They do not download telemetry. Coverage includes measured sector
crossings, shared clocks, race isolation, train-only scaling, invalid training
states, checkpoint reproducibility, validation-only tuning and Streamlit artifact
handling. Fixture scores are not race-performance evidence.
