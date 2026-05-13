#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_DIR}"

RUNS=(
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r070_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_seed43_full100"
)

echo "timestamp: $(date '+%F %T %Z')"
echo
echo "GPU 5/6/7:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '$1>=5 && $1<=7 {printf "  GPU%s: mem %s/%s MiB, util %s%%\n", $1, $2, $3, $4}'
echo
echo "tmux:"
tmux ls 2>/dev/null | rg 'graphunet.*full|cortex_graphunet' || true
echo

for run in "${RUNS[@]}"; do
  echo "### ${run}"
  train_log="runs/${run}/train_log.csv"
  he_manifest="downstream/outputs/he_king4p5_anon/${run}/arena_manifest.json"
  if [[ -f "${train_log}" ]]; then
    rows=$(( $(wc -l < "${train_log}") - 1 ))
    echo "epochs_logged: ${rows}"
    tail -n 1 "${train_log}"
  else
    echo "epochs_logged: 0"
  fi
  if [[ -f "${he_manifest}" ]]; then
    python - "${he_manifest}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
print(f"h2_done: {payload.get('mean_h2_pca_weighted')}")
print(f"timestamp_utc: {payload.get('timestamp_utc')}")
PY
  else
    echo "h2_done: no"
  fi
  echo
done
