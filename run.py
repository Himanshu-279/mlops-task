"""
MLOps Batch Job - Rolling Mean Signal Generator
Primetrade.ai T0 Technical Assessment
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to both file and stdout."""
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def write_metrics(output_path: str, payload: dict) -> None:
    """Write metrics JSON to disk."""
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_error_metrics(output_path: str, version: str, error_msg: str) -> None:
    """Write error metrics JSON."""
    payload = {
        "version": version,
        "status": "error",
        "error_message": error_msg
    }
    write_metrics(output_path, payload)


def load_config(config_path: str, logger: logging.Logger) -> dict:
    """Load and validate YAML config."""
    logger.info(f"Loading config from: {config_path}")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is not a valid YAML mapping.")

    required_fields = ["seed", "window", "version"]
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Missing required config field: '{field}'")

    # Type validation
    if not isinstance(config["seed"], int):
        raise TypeError(f"Config 'seed' must be an integer, got: {type(config['seed']).__name__}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str):
        raise TypeError(f"Config 'version' must be a string, got: {type(config['version']).__name__}")

    logger.info(
        f"Config validated — seed={config['seed']}, "
        f"window={config['window']}, version={config['version']}"
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate input CSV."""
    logger.info(f"Loading dataset from: {input_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if Path(input_path).stat().st_size == 0:
        raise ValueError("Input file is empty.")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Dataset is empty after parsing.")

    if "close" not in df.columns:
        raise KeyError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")

    # Validate close column is numeric
    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise TypeError("Column 'close' must be numeric.")

    null_count = df["close"].isnull().sum()
    if null_count > 0:
        logger.warning(f"'close' column has {null_count} null values — they will be excluded from signal computation.")

    return df


def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """Compute rolling mean on close column."""
    logger.info(f"Computing rolling mean with window={window}")
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    valid_count = rolling_mean.notna().sum()
    logger.info(
        f"Rolling mean computed — "
        f"{window - 1} rows have NaN (first window-1 rows excluded from signal), "
        f"{valid_count} valid rows"
    )
    return rolling_mean


def compute_signal(df: pd.DataFrame, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    Generate binary signal:
      signal = 1 if close > rolling_mean else 0
    Rows where rolling_mean is NaN are excluded (signal = NaN → dropped from rate calc).
    """
    logger.info("Generating binary signal (close > rolling_mean → 1, else → 0)")
    signal = pd.Series(np.nan, index=df.index)
    valid_mask = rolling_mean.notna() & df["close"].notna()
    signal[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean[valid_mask]).astype(int)
    valid_signals = signal.dropna()
    logger.info(
        f"Signal generated — "
        f"{valid_signals.sum():.0f} buy signals (1), "
        f"{(valid_signals == 0).sum():.0f} hold signals (0) "
        f"out of {len(valid_signals)} valid rows"
    )
    return signal


def main():
    parser = argparse.ArgumentParser(description="MLOps Batch Job — Rolling Mean Signal")
    parser.add_argument("--input",   required=True, help="Path to input CSV file")
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--output",  required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path to log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    job_start = time.time()
    version = "v1"  # fallback before config is loaded

    logger.info("=" * 60)
    logger.info("MLOps Batch Job STARTED")
    logger.info(f"Input:   {args.input}")
    logger.info(f"Config:  {args.config}")
    logger.info(f"Output:  {args.output}")
    logger.info(f"Log:     {args.log_file}")
    logger.info("=" * 60)

    try:
        # Step 1 — Load & validate config
        config = load_config(args.config, logger)
        version = config["version"]
        seed    = config["seed"]
        window  = config["window"]

        # Step 2 — Set random seed for reproducibility
        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # Step 3 — Load & validate dataset
        df = load_dataset(args.input, logger)
        rows_loaded = len(df)

        # Step 4 — Rolling mean
        rolling_mean = compute_rolling_mean(df, window, logger)

        # Step 5 — Signal generation
        signal = compute_signal(df, rolling_mean, logger)

        # Step 6 — Compute metrics
        valid_signals = signal.dropna()
        rows_processed = rows_loaded
        signal_rate = float(round(valid_signals.mean(), 4))
        latency_ms = int(round((time.time() - job_start) * 1000))

        logger.info("-" * 60)
        logger.info("METRICS SUMMARY")
        logger.info(f"  version:         {version}")
        logger.info(f"  rows_processed:  {rows_processed}")
        logger.info(f"  signal_rate:     {signal_rate}")
        logger.info(f"  latency_ms:      {latency_ms}")
        logger.info(f"  seed:            {seed}")
        logger.info(f"  status:          success")
        logger.info("-" * 60)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success"
        }
        write_metrics(args.output, metrics)
        logger.info(f"Metrics written to: {args.output}")

        # Print final metrics JSON to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))

        logger.info("MLOps Batch Job COMPLETED — status: success")
        sys.exit(0)

    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        error_msg = str(e)
        logger.error(f"Job FAILED — {type(e).__name__}: {error_msg}")
        write_error_metrics(args.output, version, error_msg)
        logger.info(f"Error metrics written to: {args.output}")
        logger.info("MLOps Batch Job COMPLETED — status: error")
        sys.exit(1)

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception(error_msg)
        write_error_metrics(args.output, version, error_msg)
        logger.info(f"Error metrics written to: {args.output}")
        logger.info("MLOps Batch Job COMPLETED — status: error")
        sys.exit(1)


if __name__ == "__main__":
    main()
