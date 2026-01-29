#!/usr/bin/env bash
set -euo pipefail

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
FORECAST_FILE="$BASE_PUBLIC/forecast.yaml"

for cfg in "$TASK_FILE" "$FILEPATHS_FILE" "$SYSTEM_FILE" "$METRICS_FILE" "$FORECAST_FILE"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[!] Missing required config file: $cfg" >&2
    exit 1
  fi
done

RUN_ROOT="$BASE_CONFIG/tuning"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/runs}"
RUN_DIR="$RUN_ROOT/hpo_$RUN_TAG"
RUN_PUBLIC="$RUN_DIR/public"
RUN_PRIVATE="$RUN_DIR/private"

dir_nonempty() {
  [[ -d "$1" ]] && find "$1" -mindepth 1 -print -quit | grep -q .
}

if [[ -d "$RUN_DIR" ]]; then
  if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    if dir_nonempty "$RUN_DIR"; then
      echo "[!] Target directory already exists and is not empty: $RUN_DIR" >&2
      echo "    Set ALLOW_OVERWRITE=1 to reuse it." >&2
      exit 1
    else
      echo "[info] Existing but empty target directory detected; reusing it."
    fi
  fi
fi

mkdir -p "$RUN_PUBLIC" "$RUN_PRIVATE"
cp "$SYSTEM_FILE" "$RUN_PUBLIC/"
cp "$METRICS_FILE" "$RUN_PUBLIC/"
cp "$FORECAST_FILE" "$RUN_PUBLIC/"   # keep models as in config
cp "$FILEPATHS_FILE" "$RUN_PUBLIC/"  # will be overwritten per-task if ISOLATE_OUTPUTS=1

if [[ -d "$BASE_PRIVATE" ]]; then
  cp -R "$BASE_PRIVATE"/. "$RUN_PRIVATE"/ 2>/dev/null || true
fi

FILE_FORMAT=$(python - "$FILEPATHS_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("filepaths", {}).get("file_format", "FEATHER"))
PY
)
FILE_EXT="$(echo "$FILE_FORMAT" | tr '[:upper:]' '[:lower:]')"

ISOLATE_OUTPUTS="${ISOLATE_OUTPUTS:-1}"  # 1 = per-task output dirs (recommended)

if [[ "$ISOLATE_OUTPUTS" != "1" ]]; then
  BASE_CV_DIR=$(python - "$FILEPATHS_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("filepaths", {}).get("cv_results_dir", ""))
PY
)
  BASE_EVAL_DIR=$(python - "$FILEPATHS_FILE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("filepaths", {}).get("eval_results_dir", ""))
PY
)
fi

TASK_FILTER="${TASKS:-${TASK:-}}"

TASK_LINES=$(python - "$TASK_FILE" "$TASK_FILTER" <<'PY'
import sys, yaml
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
) || { echo "[!] Failed to parse tasks" >&2; exit 1; }

declare -a completed skipped failed
completed=()
skipped=()
failed=()
total=0

while IFS='|' read -r TASK DATASET; do
  [[ -z "$TASK" ]] && continue
  total=$((total + 1))
  TASK_SAFE="${TASK//[^A-Za-z0-9._-]/_}"

  if [[ "$ISOLATE_OUTPUTS" == "1" ]]; then
    TASK_BASE="$OUTPUT_ROOT/$TASK_SAFE"
    OUT_PROCESSED="$TASK_BASE/processed"
    OUT_SKU="$TASK_BASE/sku_stats"
    OUT_CV="$TASK_BASE/cv_results"
    OUT_EVAL="$TASK_BASE/eval_results"
    OUT_PLOTS="$TASK_BASE/eval_plots"
    mkdir -p "$OUT_PROCESSED" "$OUT_SKU" "$OUT_CV" "$OUT_EVAL" "$OUT_PLOTS"

    CV_PATH="$OUT_CV/$DATASET.$FILE_EXT"
    EVAL_PATH="$OUT_EVAL/$DATASET.$FILE_EXT"

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
    CV_PATH="$BASE_CV_DIR/$DATASET.$FILE_EXT"
    EVAL_PATH="$BASE_EVAL_DIR/$DATASET.$FILE_EXT"
  fi

  if [[ -f "$CV_PATH" && -f "$EVAL_PATH" ]]; then
    echo "[skip] $TASK already has results at $CV_PATH and $EVAL_PATH"
    skipped+=("$TASK")
    continue
  fi

  printf "tasks:\n  - %s\n" "$TASK" >"$RUN_PUBLIC/task.yaml"
  echo "[run] $TASK (dataset: $DATASET) using models from $FORECAST_FILE"

  if python -m src.main --config-dir "$RUN_DIR"; then
    if [[ -f "$CV_PATH" && -f "$EVAL_PATH" ]]; then
      echo "[ok] $TASK completed"
      completed+=("$TASK")
    else
      echo "[warn] $TASK run finished but outputs not found" >&2
      failed+=("$TASK (missing outputs)")
    fi
  else
    echo "[error] Run failed for $TASK" >&2
    failed+=("$TASK (execution error)")
  fi
done <<<"$TASK_LINES"

echo
echo "=== summary ==="
echo "Config used: $RUN_DIR"
echo "Total tasks: $total"
echo "Completed: ${#completed[@]:-0}"
echo "Skipped (existing results): ${#skipped[@]:-0}"
echo "Failed: ${#failed[@]:-0}"

if [[ ${#completed[@]:-0} -gt 0 ]]; then
  printf "Completed tasks: %s\n" "${completed[*]:-}"
fi
if [[ ${#skipped[@]:-0} -gt 0 ]]; then
  printf "Skipped tasks: %s\n" "${skipped[*]:-}"
fi
if [[ ${#failed[@]:-0} -gt 0 ]]; then
  printf "Failed tasks: %s\n" "${failed[*]:-}"
  exit 1
fi
