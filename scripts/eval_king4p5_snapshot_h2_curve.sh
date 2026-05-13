#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

GPU=${GPU:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
TARGET_EPOCHS=${TARGET_EPOCHS:-"10 25 50 75 100"}
CKPT_KINDS=${CKPT_KINDS:-"last best_recon"}

ANON_MANIFEST=${ANON_MANIFEST:-cache/manifest_fsaverage4_king4p5_anon.csv}
ANON_EDGE_CACHE_DIR=${ANON_EDGE_CACHE_DIR:-cache/templates_ukb_fsaverage4}
ANON_BUNDLE_DIR=${ANON_BUNDLE_DIR:-downstream/new/king_anonymous_over4p5_discovery_bundle}
Z_ROOT=${Z_ROOT:-downstream/outputs/z_graph_king4p5_anon_snapshots}
HE_ROOT=${HE_ROOT:-downstream/outputs/he_king4p5_anon_snapshots}

RUNS=(
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r070_p080_full100"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_seed43_full100"
)

if [[ "$#" -gt 0 ]]; then
  RUNS=("$@")
fi

docker_py() {
  env GPU_SPEC="device=${GPU}" \
    PROJECT_MOUNT="/DATA" \
    CONTAINER_PROJECT_ROOT="/DATA" \
    CONTAINER_REPO_DIR="${REPO_DIR}" \
    "${REPO_DIR}/scripts/docker_run.sh" "$@"
}

eval_one() {
  local run=$1
  local target=$2
  local kind=$3
  local ckpt="runs/${run}/snapshots/ckpt_epoch$(printf "%03d" "${target}")_${kind}.pt"
  local eval_name="${run}__epoch$(printf "%03d" "${target}")_${kind}"
  local zdir="${Z_ROOT}/${eval_name}"
  local hdir="${HE_ROOT}/${eval_name}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "missing checkpoint: ${ckpt}" >&2
    return 0
  fi
  mkdir -p "${zdir}" "${hdir}"

  if [[ ! -f "${zdir}/embedding_stats.json" ]]; then
    echo "extract ${eval_name}"
    docker_py python src/train/extract_z_graph.py \
      --manifest "${ANON_MANIFEST}" \
      --ckpt "${ckpt}" \
      --res fsaverage4 \
      --edge_cache_dir "${ANON_EDGE_CACHE_DIR}" \
      --device cuda \
      --num_workers "${NUM_WORKERS}" \
      --skip_umap 1 \
      --out_dir "${zdir}" \
      > "${zdir}/extract.log" 2>&1
  else
    echo "reuse extract ${eval_name}"
  fi

  if [[ ! -f "${hdir}/arena_manifest.json" ]]; then
    echo "he ${eval_name}"
    docker_py python "${ANON_BUNDLE_DIR}/run_fast_h2_king_standard_anonymous.py" \
      --anon_bundle_dir "${ANON_BUNDLE_DIR}" \
      --phenotype_npy_dir "${zdir}" \
      --embedding_id_mode anon \
      --n_pca 128 \
      --output_root "${hdir}" \
      > "${hdir}/he.log" 2>&1
  else
    echo "reuse h2 ${eval_name}"
  fi
}

for run in "${RUNS[@]}"; do
  for target in ${TARGET_EPOCHS}; do
    for kind in ${CKPT_KINDS}; do
      eval_one "${run}" "${target}" "${kind}"
    done
  done
done

docker_py python scripts/summarize_king4p5_snapshot_curve.py \
  --runs "${RUNS[@]}" \
  --targets ${TARGET_EPOCHS} \
  --kinds ${CKPT_KINDS} \
  --he-root "${HE_ROOT}" \
  --out downstream/outputs/king4p5_snapshot_h2_curve.tsv \
  --fig-dir figures/king4p5_snapshot_h2_curve
