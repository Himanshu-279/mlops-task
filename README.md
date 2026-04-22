# MLOps Batch Job — Rolling Mean Signal Generator

A minimal, reproducible MLOps-style batch pipeline that computes a rolling mean on OHLCV close prices and generates binary trading signals.

---

## Project Structure

```
mlops-task/
├── run.py           # Main batch script
├── config.yaml      # Configuration (seed, window, version)
├── data.csv         # 10,000-row OHLCV dataset
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker build file
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md        # This file
```

---

## Local Run Instructions

### Prerequisites
- Python 3.9+
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the batch job

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

### 3. View results

```bash
cat metrics.json
cat run.log
```

---

## Docker Build & Run

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container

```bash
docker run --rm mlops-task
```

The container will:
- Run the full pipeline using the bundled `data.csv` and `config.yaml`
- Print the final `metrics.json` to stdout
- Exit with code `0` on success, non-zero on failure

### Copy output files from container (optional)

```bash
# Run with a named container to extract files
docker run --name mlops-run mlops-task
docker cp mlops-run:/app/metrics.json ./metrics.json
docker cp mlops-run:/app/run.log ./run.log
docker rm mlops-run
```

---

## Configuration (`config.yaml`)

| Field     | Type   | Description                          |
|-----------|--------|--------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility |
| `window`  | int    | Rolling mean window size              |
| `version` | string | Pipeline version tag                  |

---

## Example `metrics.json` (Success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 34,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (Error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

---

## Design Decisions

- **NaN handling**: The first `window - 1` rows produce NaN rolling means and are excluded from signal computation. `signal_rate` is computed only over valid rows.
- **Reproducibility**: `numpy.random.seed(seed)` is set immediately after config load. Same config + data = identical output every run.
- **Error handling**: All known failure modes (missing file, bad CSV, missing column, invalid config) write an error `metrics.json` and exit with code 1.
- **No hardcoded paths**: All file paths come from CLI arguments (`--input`, `--config`, `--output`, `--log-file`).
