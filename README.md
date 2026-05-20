# T0 — MLOps Signal Pipeline

A minimal MLOps batch job that loads OHLCV price data, computes a rolling-mean trading signal, and writes structured metrics + logs. Demonstrates reproducibility, observability, and Docker-based deployment — the pattern used in MetaStackerBandit trading-signal pipelines.

---

## What it does

1. Loads `config.yaml` (seed, rolling-window size, version label)
2. Reads `data.csv` — 10,000 rows of 1-minute BTC/USD OHLCV data
3. Computes a 5-period rolling mean on the `close` column
4. Generates a binary signal per row: `1` if `close > rolling_mean`, else `0`
5. Writes `metrics.json` (machine-readable summary) and `run.log` (audit trail)
6. Prints the final metrics JSON to stdout and exits `0` on success, `1` on error

---

## Local run

### Prerequisites

Python 3.9+ and pip.

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

All four arguments are required. No hard-coded paths anywhere in the code.

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

Expected: **37 passed**.

---

## Docker build & run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The image bakes in `data.csv` and `config.yaml`. The container prints the metrics JSON to stdout and exits with code `0` on success, non-zero on failure.

---

## Configuration (`config.yaml`)

```yaml
seed: 42
window: 5
version: "v1"
```

| Key | Type | Description |
|-----|------|-------------|
| `seed` | int | NumPy random seed — set before all data operations for reproducibility |
| `window` | int (> 0) | Rolling mean lookback period in rows |
| `version` | string | Pipeline version label — copied verbatim into `metrics.json` |

---

## Output

### `metrics.json` — success

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 20,
  "seed": 42,
  "status": "success"
}
```

| Key | Description |
|-----|-------------|
| `version` | From `config.yaml` |
| `rows_processed` | Total rows loaded from the CSV |
| `metric` | Always `"signal_rate"` |
| `value` | Fraction of valid rows where `close > rolling_mean` (4 dp) |
| `latency_ms` | Total wall-clock runtime in milliseconds |
| `seed` | Seed used — confirms reproducibility |
| `status` | `"success"` or `"error"` |

### `metrics.json` — error

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input file not found: data.csv"
}
```

The metrics file is **always written**, even when the pipeline fails.

### `run.log` — sample

```
2026-05-20 15:23:28,990 [INFO] Job start
2026-05-20 15:23:28,991 [INFO] Config loaded: seed=42 window=5 version=v1
2026-05-20 15:23:29,007 [INFO] Dataset loaded: 10000 rows
2026-05-20 15:23:29,010 [INFO] Rolling mean computed (window=5), 4 warm-up rows excluded
2026-05-20 15:23:29,010 [INFO] Signal generated: 9996 rows, signal_rate=0.4991
2026-05-20 15:23:29,011 [INFO] Metrics: {"version": "v1", "rows_processed": 10000, "metric": "signal_rate", "value": 0.4991, "latency_ms": 20, "seed": 42, "status": "success"}
2026-05-20 15:23:29,011 [INFO] Job end: success
```

---

## Reproducibility

Running the pipeline twice on the same input always produces the same `value`. Verified by `tests/test_pipeline.py::TestEndToEnd::test_determinism`.

```bash
python run.py --input data.csv --config config.yaml --output m1.json --log-file l1.log
python run.py --input data.csv --config config.yaml --output m2.json --log-file l2.log
diff m1.json m2.json   # no output = identical
```

---

## Deliverables

| File | Purpose |
|------|---------|
| `run.py` | Pipeline — all logic in one file |
| `config.yaml` | Runtime parameters |
| `data.csv` | 10,000-row BTC/USD OHLCV dataset |
| `requirements.txt` | Pinned dependencies |
| `Dockerfile` | Container definition |
| `metrics.json` | Sample output from a successful run |
| `run.log` | Sample log from a successful run |
| `tests/test_pipeline.py` | 37-test validation suite |

---

## Evaluation rubric

| Criterion | Weight |
|-----------|--------|
| Correctness & determinism | 40% |
| Dockerization | 25% |
| Code quality | 20% |
| Observability | 15% |

Auto-fail conditions: Docker build/run fails · `metrics.json` not written · non-deterministic outputs · hard-coded paths.
