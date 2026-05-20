import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-file", required=True)
    return parser.parse_args()


def setup_logging(log_file):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr),
        ],
    )


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    for key in ("seed", "window", "version"):
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if not isinstance(cfg["seed"], int):
        raise ValueError("Config 'seed' must be an integer")
    if not isinstance(cfg["window"], int) or cfg["window"] <= 0:
        raise ValueError("Config 'window' must be a positive integer")
    if not isinstance(cfg["version"], str):
        raise ValueError("Config 'version' must be a string")
    logging.info(f"Config loaded: seed={cfg['seed']} window={cfg['window']} version={cfg['version']}")
    return cfg


def load_dataset(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        df = pd.read_csv(input_path)
        # Some CSV exports (e.g. Google Sheets) wrap every row in outer quotes,
        # causing pandas to read the entire line as one column. Detect and retry.
        if len(df.columns) == 1 and "," in df.columns[0]:
            df = pd.read_csv(input_path, quoting=csv.QUOTE_NONE)
            df.columns = [col.strip('"') for col in df.columns]
    except pd.errors.EmptyDataError:
        raise ValueError("Empty dataset: input file has no rows")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")
    if len(df) == 0:
        raise ValueError("Empty dataset: input file has no rows")
    if "close" not in df.columns:
        raise ValueError("Missing required column: close")
    logging.info(f"Dataset loaded: {len(df)} rows")
    return df


def compute_signal(df, window):
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    nan_count = df["rolling_mean"].isna().sum()
    valid_df = df.dropna(subset=["rolling_mean"]).copy()
    valid_df["signal"] = (valid_df["close"] > valid_df["rolling_mean"]).astype(int)
    logging.info(f"Rolling mean computed (window={window}), {nan_count} warm-up rows excluded")
    logging.info(f"Signal generated: {len(valid_df)} rows, signal_rate={valid_df['signal'].mean():.4f}")
    return valid_df


def compute_metrics(valid_df, config, total_rows, latency_ms):
    return {
        "version": config["version"],
        "rows_processed": total_rows,
        "metric": "signal_rate",
        "value": round(float(valid_df["signal"].mean()), 4),
        "latency_ms": int(latency_ms),
        "seed": config["seed"],
        "status": "success",
    }


def write_metrics(output_path, payload):
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    t0 = time.time()
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("Job start")

    error_base = {"version": "unknown", "status": "error"}

    try:
        config = load_config(args.config)
        error_base["version"] = config.get("version", "unknown")
        np.random.seed(config["seed"])

        df = load_dataset(args.input)
        total_rows = len(df)

        valid_df = compute_signal(df, config["window"])

        latency_ms = (time.time() - t0) * 1000
        payload = compute_metrics(valid_df, config, total_rows, latency_ms)
        write_metrics(args.output, payload)

        logging.info(f"Metrics: {json.dumps(payload)}")
        logging.info("Job end: success")
        print(json.dumps(payload, indent=2))
        sys.exit(0)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        error_payload = {**error_base, "error_message": str(e)}
        try:
            write_metrics(args.output, error_payload)
        except Exception:
            pass
        logging.info("Job end: error")
        print(json.dumps(error_payload, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
