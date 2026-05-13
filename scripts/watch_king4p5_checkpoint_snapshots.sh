#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_DIR}"

INTERVAL_SEC=${INTERVAL_SEC:-60}
TARGET_EPOCHS=${TARGET_EPOCHS:-"10 25 50 75 100"}
SNAPSHOT_SUBDIR=${SNAPSHOT_SUBDIR:-snapshots}

DEFAULT_RUNS=(
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r070_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_seed43_full100"
  "ukb6067_gcn_mean_globalnorm_hybrid_r060"
  "ukb6067_gcn_mean_globalnorm_hybrid_r070"
  "ukb6067_meshhier_pca_globalnorm_hybrid_r070"
)

if [[ "$#" -gt 0 ]]; then
  RUNS=("$@")
else
  RUNS=("${DEFAULT_RUNS[@]}")
fi

read_current_epoch() {
  local train_log=$1
  awk -F, 'NF && $1 ~ /^[0-9]+$/ {epoch=$1} END {if (epoch == "") exit 1; print epoch}' "${train_log}"
}

ckpt_is_current_for_log() {
  local ckpt=$1
  local train_log=$2
  [[ -f "${ckpt}" ]] || return 1
  [[ "$(stat -c %Y "${ckpt}")" -ge "$(stat -c %Y "${train_log}")" ]]
}

snapshot_one_run() {
  local run=$1
  local run_dir="runs/${run}"
  local train_log="${run_dir}/train_log.csv"
  local ckpt_last="${run_dir}/ckpt_last.pt"
  local ckpt_best="${run_dir}/ckpt_best_recon.pt"
  local snap_dir="${run_dir}/${SNAPSHOT_SUBDIR}"
  local manifest="${snap_dir}/snapshot_manifest.tsv"

  [[ -f "${train_log}" ]] || return 0

  local current_epoch
  current_epoch=$(read_current_epoch "${train_log}" 2>/dev/null || true)
  [[ -n "${current_epoch}" ]] || return 0

  ckpt_is_current_for_log "${ckpt_last}" "${train_log}" || return 0

  mkdir -p "${snap_dir}"
  if [[ ! -f "${manifest}" ]]; then
    printf "timestamp\trun\ttarget_epoch\tepoch_seen\tlast_ckpt\tbest_ckpt\n" > "${manifest}"
  fi

  local target last_out best_out
  for target in ${TARGET_EPOCHS}; do
    [[ "${current_epoch}" -ge "${target}" ]] || continue
    last_out="${snap_dir}/ckpt_epoch$(printf "%03d" "${target}")_last.pt"
    best_out="${snap_dir}/ckpt_epoch$(printf "%03d" "${target}")_best_recon.pt"
    if [[ ! -f "${last_out}" ]]; then
      cp -p "${ckpt_last}" "${last_out}"
      if [[ -f "${ckpt_best}" ]]; then
        cp -p "${ckpt_best}" "${best_out}"
      else
        best_out=""
      fi
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T %Z')" "${run}" "${target}" "${current_epoch}" "${last_out}" "${best_out}" \
        >> "${manifest}"
      echo "snapshotted ${run}: target=${target}, epoch_seen=${current_epoch}"
    fi
  done
}

echo "watching checkpoint snapshots every ${INTERVAL_SEC}s"
echo "targets: ${TARGET_EPOCHS}"
printf "runs:\n"
printf "  %s\n" "${RUNS[@]}"

while true; do
  for run in "${RUNS[@]}"; do
    snapshot_one_run "${run}"
  done
  sleep "${INTERVAL_SEC}"
done
