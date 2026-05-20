# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLOps technical assessment (T0): a minimal Python batch pipeline that loads OHLCV market data, computes a rolling mean on `close`, generates a binary trading signal, and writes structured metrics + logs. Mirrors MetaStackerBandit trading-signal pipelines.

## Required Commands

### Local run
```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

### Docker
```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

Docker is evaluated exactly as above — no flags, no volume mounts. `data.csv` and `config.yaml` must be baked into the image. Exit code 0 on success, non-zero on failure.

## Architecture

Single-file pipeline: `run.py` with CLI via `argparse`. No hard-coded paths — all inputs/outputs come from CLI args.

**Execution order:**
1. Parse CLI args
2. Start timer (`time.time()`)
3. Load + validate `config.yaml` (required keys: `seed`, `window`, `version`)
4. `numpy.random.seed(seed)` for determinism
5. Load + validate `data.csv` — must have `close` column, non-empty
6. Compute rolling mean on `close` with `window` (pandas `rolling().mean()`)
7. Generate signal: `1 if close > rolling_mean else 0` — skip first `window-1` NaN rows
8. Compute metrics: `rows_processed`, `signal_rate = mean(signal)`, `latency_ms`
9. Write `metrics.json` (see schema below) — write in **both** success and error cases
10. Print final metrics JSON to stdout

**metrics.json schemas:**
```json
// success
{"version": "v1", "rows_processed": 10000, "metric": "signal_rate", "value": 0.4990, "latency_ms": 127, "seed": 42, "status": "success"}

// error
{"version": "v1", "status": "error", "error_message": "..."}
```

**Logging** (`run.log`): use Python `logging` module, log job start, config values, rows loaded, each processing step, metrics summary, job end + status, and any exceptions.

## Required Files

| File | Notes |
|------|-------|
| `run.py` | Main pipeline — all logic here |
| `config.yaml` | `seed: 42`, `window: 5`, `version: "v1"` |
| `data.csv` | 10,000-row OHLCV dataset (must include `close`) |
| `requirements.txt` | Pin versions for reproducibility |
| `Dockerfile` | `python:3.9-slim` base; COPY all inputs; run pipeline as CMD |
| `metrics.json` | Sample output from a successful run |
| `run.log` | Sample log from a successful run |

## Evaluation Weights

- Correctness & determinism (40%) — signal logic, exact JSON keys, reproducible across runs
- Dockerization (25%) — `docker build` + `docker run --rm mlops-task` must work without modification
- Code quality (20%) — validation, clean error handling
- Observability (15%) — meaningful logs

**Auto-fail conditions:** Docker build/run fails · metrics.json not written · non-deterministic outputs · hard-coded paths
