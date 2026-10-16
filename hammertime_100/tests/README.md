# Focused regression tests

Run from the repository root with:

```sh
python -m unittest discover -s hammertime_100/tests
```

The tests use small in-memory fixtures and do not download telemetry or require
FastF1. They still require the project dependencies used by the tested modules,
including PyTorch, PyTorch Geometric, pandas, NumPy, and scikit-learn.
