#!/usr/bin/env bash
set -euo pipefail

# CARC terminal helper for YAML batch evaluation.
# Usage:
#   bash scripts/run_batch_carc.sh
#   bash scripts/run_batch_carc.sh --batch-config configs/batch_eval.yaml
#   bash scripts/run_batch_carc.sh --batch-config configs/batch_eval.yaml --log-file outputs/carc_batch.log --background

BATCH_CONFIG="configs/batch_eval.yaml"
LOG_FILE=""
BACKGROUND=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-config)
      BATCH_CONFIG="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --background)
      BACKGROUND=true
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_batch_carc.sh [--batch-config <path>] [--log-file <path>] [--background] [--python-bin <python>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

if [[ ! -f "$BATCH_CONFIG" ]]; then
  echo "[ERROR] Batch config not found: $BATCH_CONFIG" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

# Recommended for vLLM + CUDA multiprocessing on Linux/HPC.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY is not set. OpenAI-based runs may fail."
fi

if [[ -z "$LOG_FILE" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="outputs/carc_batch_${TS}.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"

CMD=("$PYTHON_BIN" "scripts/run_batch_from_yaml.py" "--batch-config" "$BATCH_CONFIG")

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Batch config: $BATCH_CONFIG"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "[INFO] Command: ${CMD[*]}"

if [[ "$BACKGROUND" == true ]]; then
  nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] Started in background. PID=$PID"
  echo "[INFO] Follow logs: tail -f $LOG_FILE"
else
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi

