import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from run import (
    compute_metrics,
    compute_signal,
    load_config,
    load_dataset,
    write_metrics,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("seed: 42\nwindow: 5\nversion: v1\n")
        cfg = load_config(str(f))
        assert cfg == {"seed": 42, "window": 5, "version": "v1"}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_missing_seed(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("window: 5\nversion: v1\n")
        with pytest.raises(ValueError, match="seed"):
            load_config(str(f))

    def test_missing_window(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("seed: 42\nversion: v1\n")
        with pytest.raises(ValueError, match="window"):
            load_config(str(f))

    def test_missing_version(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("seed: 42\nwindow: 5\n")
        with pytest.raises(ValueError, match="version"):
            load_config(str(f))

    def test_invalid_yaml_structure(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("just a plain string\n")
        with pytest.raises((ValueError, Exception)):
            load_config(str(f))

    def test_zero_window_rejected(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("seed: 42\nwindow: 0\nversion: v1\n")
        with pytest.raises(ValueError, match="window"):
            load_config(str(f))

    def test_negative_window_rejected(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("seed: 42\nwindow: -3\nversion: v1\n")
        with pytest.raises(ValueError, match="window"):
            load_config(str(f))


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def _csv(self, tmp_path, content):
        f = tmp_path / "data.csv"
        f.write_text(content)
        return str(f)

    def test_valid_csv(self, tmp_path):
        path = self._csv(tmp_path, "open,high,low,close,volume\n1,2,3,4,100\n")
        df = load_dataset(path)
        assert "close" in df.columns
        assert len(df) == 1

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "missing.csv"))

    def test_empty_file(self, tmp_path):
        path = self._csv(tmp_path, "")
        with pytest.raises(ValueError, match="[Ee]mpty"):
            load_dataset(path)

    def test_header_only_no_rows(self, tmp_path):
        path = self._csv(tmp_path, "open,high,low,close,volume\n")
        with pytest.raises(ValueError, match="[Ee]mpty"):
            load_dataset(path)

    def test_missing_close_column(self, tmp_path):
        path = self._csv(tmp_path, "open,high,low,volume\n1,2,3,100\n")
        with pytest.raises(ValueError, match="close"):
            load_dataset(path)

    def test_multiple_rows_loaded(self, tmp_path):
        rows = "\n".join(f"1,2,3,{i},100" for i in range(10))
        path = self._csv(tmp_path, f"open,high,low,close,volume\n{rows}\n")
        df = load_dataset(path)
        assert len(df) == 10


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

class TestComputeSignal:
    def _df(self, closes):
        return pd.DataFrame({"close": closes})

    def test_signal_1_when_close_above_mean(self):
        closes = [1.0, 1.0, 1.0, 1.0, 10.0]
        result = compute_signal(self._df(closes), window=3)
        assert result["signal"].iloc[-1] == 1

    def test_signal_0_when_close_below_mean(self):
        closes = [10.0, 10.0, 10.0, 10.0, 1.0]
        result = compute_signal(self._df(closes), window=3)
        assert result["signal"].iloc[-1] == 0

    def test_nan_warmup_rows_excluded(self):
        closes = [1, 2, 3, 4, 5]
        result = compute_signal(self._df(closes), window=3)
        # window=3 → first 2 NaN rows dropped → 3 valid rows remain
        assert len(result) == 3
        assert result["rolling_mean"].isna().sum() == 0

    def test_signal_values_strictly_binary(self):
        closes = list(range(1, 21))
        result = compute_signal(self._df(closes), window=5)
        assert set(result["signal"].unique()).issubset({0, 1})

    def test_rolling_mean_value_correct(self):
        closes = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_signal(self._df(closes), window=3)
        # First valid row (index 2): mean([10,20,30]) = 20.0
        assert result["rolling_mean"].iloc[0] == pytest.approx(20.0)

    def test_window_1_keeps_all_rows(self):
        closes = [5.0, 3.0, 7.0]
        result = compute_signal(self._df(closes), window=1)
        assert len(result) == 3

    def test_determinism_same_seed(self):
        closes = list(range(1, 101))
        df = self._df(closes)

        np.random.seed(42)
        r1 = compute_signal(df.copy(), window=5)

        np.random.seed(42)
        r2 = compute_signal(df.copy(), window=5)

        assert r1["signal"].tolist() == r2["signal"].tolist()

    def test_equal_close_and_mean_gives_0(self):
        # close == rolling_mean → signal 0 (not strictly greater)
        closes = [5.0, 5.0, 5.0]
        result = compute_signal(self._df(closes), window=3)
        assert result["signal"].iloc[-1] == 0


# ---------------------------------------------------------------------------
# Metrics payload
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _valid_df(self):
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5], "rolling_mean": [np.nan, np.nan, 2, 3, 4]})
        df = df.dropna().copy()
        df["signal"] = [0, 1, 1]
        return df

    def test_success_payload_keys(self):
        cfg = {"version": "v1", "seed": 42}
        payload = compute_metrics(self._valid_df(), cfg, 5, 100.0)
        for key in ("version", "rows_processed", "metric", "value", "latency_ms", "seed", "status"):
            assert key in payload

    def test_status_is_success(self):
        cfg = {"version": "v1", "seed": 42}
        payload = compute_metrics(self._valid_df(), cfg, 5, 100.0)
        assert payload["status"] == "success"

    def test_signal_rate_rounded_to_4dp(self):
        cfg = {"version": "v1", "seed": 42}
        payload = compute_metrics(self._valid_df(), cfg, 5, 100.0)
        # 2/3 ≈ 0.6667
        assert payload["value"] == pytest.approx(0.6667, abs=1e-4)

    def test_rows_processed_is_total_not_valid(self):
        cfg = {"version": "v1", "seed": 42}
        payload = compute_metrics(self._valid_df(), cfg, 10000, 100.0)
        assert payload["rows_processed"] == 10000

    def test_latency_ms_is_int(self):
        cfg = {"version": "v1", "seed": 42}
        payload = compute_metrics(self._valid_df(), cfg, 5, 127.9)
        assert isinstance(payload["latency_ms"], int)


# ---------------------------------------------------------------------------
# write_metrics
# ---------------------------------------------------------------------------

class TestWriteMetrics:
    def test_writes_json_to_disk(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"status": "success", "value": 0.5}
        write_metrics(str(path), payload)
        data = json.loads(path.read_text())
        assert data == payload

    def test_error_payload_written(self, tmp_path):
        path = tmp_path / "out.json"
        payload = {"version": "v1", "status": "error", "error_message": "test"}
        write_metrics(str(path), payload)
        data = json.loads(path.read_text())
        assert data["status"] == "error"
        assert "error_message" in data

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text('{"old": true}')
        write_metrics(str(path), {"new": True})
        data = json.loads(path.read_text())
        assert "new" in data
        assert "old" not in data


# ---------------------------------------------------------------------------
# End-to-end via subprocess
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _run(self, tmp_path, extra_args=None):
        config = tmp_path / "config.yaml"
        config.write_text("seed: 42\nwindow: 5\nversion: v1\n")
        output = tmp_path / "metrics.json"
        log = tmp_path / "run.log"
        cmd = [
            sys.executable, "run.py",
            "--input", "data.csv",
            "--config", str(config),
            "--output", str(output),
            "--log-file", str(log),
        ]
        if extra_args:
            cmd = cmd[:4] + extra_args + cmd[4:]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        return result, output, log

    @pytest.mark.skipif(not (ROOT / "data.csv").exists(), reason="data.csv not present")
    def test_success_exit_code(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        assert result.returncode == 0

    @pytest.mark.skipif(not (ROOT / "data.csv").exists(), reason="data.csv not present")
    def test_success_payload_structure(self, tmp_path):
        _, output, _ = self._run(tmp_path)
        data = json.loads(output.read_text())
        assert data["status"] == "success"
        assert data["rows_processed"] == 10000
        assert data["seed"] == 42
        assert data["version"] == "v1"
        assert 0.0 <= data["value"] <= 1.0
        assert data["latency_ms"] >= 0
        assert data["metric"] == "signal_rate"

    @pytest.mark.skipif(not (ROOT / "data.csv").exists(), reason="data.csv not present")
    def test_determinism(self, tmp_path):
        _, out1, _ = self._run(tmp_path)
        out2 = tmp_path / "metrics2.json"
        log2 = tmp_path / "run2.log"
        config = tmp_path / "config.yaml"
        subprocess.run(
            [sys.executable, "run.py",
             "--input", "data.csv",
             "--config", str(config),
             "--output", str(out2),
             "--log-file", str(log2)],
            cwd=str(ROOT)
        )
        v1 = json.loads(out1.read_text())["value"]
        v2 = json.loads(out2.read_text())["value"]
        assert v1 == v2, "signal_rate must be identical across runs"

    def test_missing_input_exits_nonzero(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("seed: 42\nwindow: 5\nversion: v1\n")
        output = tmp_path / "metrics.json"
        result = subprocess.run(
            [sys.executable, "run.py",
             "--input", "no_such_file.csv",
             "--config", str(config),
             "--output", str(output),
             "--log-file", str(tmp_path / "run.log")],
            capture_output=True, cwd=str(ROOT)
        )
        assert result.returncode != 0

    def test_metrics_written_on_error(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("seed: 42\nwindow: 5\nversion: v1\n")
        output = tmp_path / "metrics.json"
        subprocess.run(
            [sys.executable, "run.py",
             "--input", "no_such_file.csv",
             "--config", str(config),
             "--output", str(output),
             "--log-file", str(tmp_path / "run.log")],
            capture_output=True, cwd=str(ROOT)
        )
        assert output.exists(), "metrics.json must be written even on error"
        data = json.loads(output.read_text())
        assert data["status"] == "error"
        assert "error_message" in data

    def test_missing_config_exits_nonzero(self, tmp_path):
        output = tmp_path / "metrics.json"
        result = subprocess.run(
            [sys.executable, "run.py",
             "--input", "data.csv",
             "--config", str(tmp_path / "no_config.yaml"),
             "--output", str(output),
             "--log-file", str(tmp_path / "run.log")],
            capture_output=True, cwd=str(ROOT)
        )
        assert result.returncode != 0

    def test_log_file_created(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("seed: 42\nwindow: 5\nversion: v1\n")
        log = tmp_path / "run.log"
        subprocess.run(
            [sys.executable, "run.py",
             "--input", "no_such_file.csv",
             "--config", str(config),
             "--output", str(tmp_path / "metrics.json"),
             "--log-file", str(log)],
            capture_output=True, cwd=str(ROOT)
        )
        assert log.exists()
        assert "Job start" in log.read_text()
