#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter tuning runner for TFT, TCN, xLSTM, and TiDE across tasks in config/public/task.yaml.
# Supports concurrent runs by default via isolated output directories; writes per-task/per-model
# outputs; skips combos that already have CV and evaluation outputs; logs a summary at the end.
# If a target output folder already exists, the script aborts unless ALLOW_OVERWRITE=1.
#
# Filters:
#   TASK=<task_name> or TASKS="task1,task2" to limit the tasks
# Models:
#   MODELS="TFT TCN" to override the default model list

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="${1:-config}"
if [[ "$CONFIG_DIR" = /* ]]; then
  BASE_CONFIG="$CONFIG_DIR"
else
  BASE_CONFIG="$REPO_ROOT/$CONFIG_DIR"
fi

BASE_PUBLIC="$BASE_CONFIG/public"
BASE_PRIVATE="$BASE_CONFIG/private"

TASK_FILE="$BASE_PUBLIC/task.yaml"
FILEPATHS_FILE="$BASE_PUBLIC/filepaths.yaml"
SYSTEM_FILE="$BASE_PUBLIC/system.yaml"
METRICS_FILE="$BASE_PUBLIC/metrics.yaml"

for cfg in "$TASK_FILE" "$FILEPATHS_FILE" "$SYSTEM_FILE" "$METRICS_FILE"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[!] Missing required config file: $cfg" >&2
    exit 1
  fi
done

if [[ -n "${MODELS:-}" ]]; then
  read -r -a MODELS <<<"$MODELS"
else
  MODELS=(TFT TCN XLSTM TIDE)
fi
# Normalize model names to uppercase to match ModelName enum keys
for i in "${!MODELS[@]}"; do
  MODELS[$i]="$(echo "${MODELS[$i]}" | tr '[:lower:]' '[:upper:]')"
done

RUN_ROOT="$BASE_CONFIG/tuning"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
ISOLATE_OUTPUTS="${ISOLATE_OUTPUTS:-1}"  # set to 0 to reuse base filepaths (not concurrency-safe)
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/runs/$RUN_TAG}"
RUN_DIR="$RUN_ROOT/hpo_$RUN_TAG"
RUN_PUBLIC="$RUN_DIR/public"
RUN_PRIVATE="$RUN_DIR/private"

dir_nonempty() {
  [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit | grep -q .
}

if [[ -d "$RUN_DIR" || ( "$ISOLATE_OUTPUTS" == "1" && -d "$OUTPUT_ROOT" ) ]]; then
  if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    block=()
    dir_nonempty "$RUN_DIR" && block+=("RUN_DIR=$RUN_DIR")
    if [[ "$ISOLATE_OUTPUTS" == "1" ]]; then
      dir_nonempty "$OUTPUT_ROOT" && block+=("OUTPUT_ROOT=$OUTPUT_ROOT")
    fi
    if ((${#block[@]} > 0)); then
      echo "[!] Target directory already exists and is not empty. Set ALLOW_OVERWRITE=1 to reuse:" >&2
      for b in "${block[@]}"; do echo "    $b" >&2; done
      exit 1
    else
      echo "[info] Existing but empty target directories detected; reusing them."
    fi
  fi
fi

mkdir -p "$RUN_PUBLIC" "$RUN_PRIVATE"

cp "$SYSTEM_FILE" "$RUN_PUBLIC/"
cp "$METRICS_FILE" "$RUN_PUBLIC/"

if [[ -d "$BASE_PRIVATE" ]]; then
  cp -R "$BASE_PRIVATE"/. "$RUN_PRIVATE"/ 2>/dev/null || true
fi

# Load file format from the base filepaths to reuse in isolated copies.
FILE_FORMAT=$(python - "$FILEPATHS_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("filepaths", {}).get("file_format", "FEATHER"))
PY
)
FILE_EXT="$(echo "$FILE_FORMAT" | tr '[:upper:]' '[:lower:]')"

if [[ "$ISOLATE_OUTPUTS" == "1" ]]; then
  OUT_PROCESSED="$OUTPUT_ROOT/processed"
  OUT_SKU="$OUTPUT_ROOT/sku_stats"
  OUT_CV="$OUTPUT_ROOT/cv_results"
  OUT_EVAL="$OUTPUT_ROOT/eval_results"
  OUT_PLOTS="$OUTPUT_ROOT/eval_plots"
  mkdir -p "$OUT_PROCESSED" "$OUT_SKU" "$OUT_CV" "$OUT_EVAL" "$OUT_PLOTS"
  cat >"$RUN_PUBLIC/filepaths.yaml" <<EOF
filepaths:
  processed_data_dir: $OUT_PROCESSED
  sku_stats_dir: $OUT_SKU
  cv_results_dir: $OUT_CV
  eval_results_dir: $OUT_EVAL
  eval_plots_dir: $OUT_PLOTS
  file_format: $FILE_FORMAT
EOF
else
  cp "$FILEPATHS_FILE" "$RUN_PUBLIC/"
fi

{
  echo "forecast:"
  echo "  models:"
  for model in "${MODELS[@]}"; do
    echo "    - $model"
  done
} >"$RUN_PUBLIC/forecast.yaml"

TASK_FILTER="${TASKS:-${TASK:-}}"

TASK_LINES=$(python - "$TASK_FILE" "$TASK_FILTER" <<'PY'
import sys
import yaml
from pathlib import Path
from src.constants.tasks import TASKS

task_path = Path(sys.argv[1])
filter_raw = sys.argv[2] if len(sys.argv) > 2 else ""
filters = [t.strip() for t in filter_raw.replace(",", " ").split() if t.strip()]

tasks_cfg = yaml.safe_load(task_path.read_text()) or {}
tasks = tasks_cfg.get("tasks", [])
if not tasks:
    sys.stderr.write(f"No tasks found in {task_path}\n")
    sys.exit(1)

if filters:
    missing = [t for t in filters if t not in tasks]
    if missing:
        sys.stderr.write("Filtered task(s) not present in task.yaml: %s\n" % ", ".join(missing))
        sys.exit(1)
    tasks = [t for t in tasks if t in filters]
    if not tasks:
        sys.stderr.write("No tasks left after applying filter.\n")
        sys.exit(1)

unknown = [t for t in tasks if t not in TASKS]
if unknown:
    sys.stderr.write("Unknown task names: %s\n" % ", ".join(unknown))
    sys.exit(1)

for name in tasks:
    dataset = TASKS[name].dataset_name.value
    print(f"{name}|{dataset}")
PY
) || { echo "[!] Failed to parse tasks or file paths" >&2; exit 1; }

declare -a completed skipped failed
total=0

while IFS='|' read -r TASK DATASET; do
  [[ -z "$TASK" ]] && continue
  for MODEL in "${MODELS[@]}"; do
    total=$((total + 1))
    TASK_SAFE="${TASK//[^A-Za-z0-9._-]/_}"
    MODEL_SAFE="${MODEL//[^A-Za-z0-9._-]/_}"

    # Per-task/per-model output directories
    TASK_MODEL_BASE="$OUTPUT_ROOT/$TASK_SAFE/$MODEL_SAFE"
    OUT_PROCESSED="$TASK_MODEL_BASE/processed"
    OUT_SKU="$TASK_MODEL_BASE/sku_stats"
    OUT_CV="$TASK_MODEL_BASE/cv_results"
    OUT_EVAL="$TASK_MODEL_BASE/eval_results"
    OUT_PLOTS="$TASK_MODEL_BASE/eval_plots"
    mkdir -p "$OUT_PROCESSED" "$OUT_SKU" "$OUT_CV" "$OUT_EVAL" "$OUT_PLOTS"

    CV_PATH="$OUT_CV/$DATASET.$FILE_EXT"
    EVAL_PATH="$OUT_EVAL/$DATASET.$FILE_EXT"

    if [[ -f "$CV_PATH" && -f "$EVAL_PATH" ]]; then
      echo "[skip] $TASK / $MODEL already has results at $CV_PATH and $EVAL_PATH"
      skipped+=("$TASK/$MODEL")
      continue
    fi

    # Write per-task forecast and filepaths
    {
      echo "forecast:"
      echo "  models:"
      echo "    - $MODEL"
    } >"$RUN_PUBLIC/forecast.yaml"

    cat >"$RUN_PUBLIC/filepaths.yaml" <<EOF
filepaths:
  processed_data_dir: $OUT_PROCESSED
  sku_stats_dir: $OUT_SKU
  cv_results_dir: $OUT_CV
  eval_results_dir: $OUT_EVAL
  eval_plots_dir: $OUT_PLOTS
  file_format: $FILE_FORMAT
EOF

    printf "tasks:\n  - %s\n" "$TASK" >"$RUN_PUBLIC/task.yaml"
    echo "[run] $TASK (dataset: $DATASET) with model: $MODEL"

    if python -m src.main --config-dir "$RUN_DIR"; then
      if [[ -f "$CV_PATH" && -f "$EVAL_PATH" ]]; then
        echo "[ok] $TASK / $MODEL completed"
        completed+=("$TASK/$MODEL")
      else
        echo "[warn] $TASK / $MODEL run finished but outputs not found" >&2
        failed+=("$TASK/$MODEL (missing outputs)")
      fi
    else
      echo "[error] Run failed for $TASK / $MODEL" >&2
      failed+=("$TASK/$MODEL (execution error)")
    fi
  done
done <<<"$TASK_LINES"

echo
echo "=== HPO summary ==="
echo "Config used: $RUN_DIR"
echo "Total tasks: $total"
echo "Completed: ${#completed[@]}"
echo "Skipped (existing results): ${#skipped[@]}"
echo "Failed: ${#failed[@]}"

if [[ ${#completed[@]} -gt 0 ]]; then
  printf "Completed tasks: %s\n" "${completed[*]}"
fi
if [[ ${#skipped[@]} -gt 0 ]]; then
  printf "Skipped tasks: %s\n" "${skipped[*]}"
fi
if [[ ${#failed[@]} -gt 0 ]]; then
  printf "Failed tasks: %s\n" "${failed[*]}"
  exit 1
fi
