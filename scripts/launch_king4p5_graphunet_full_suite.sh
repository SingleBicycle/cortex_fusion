#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

SUITE_TAG=${1:-${SUITE_TAG:-20260429_graphunet_full}}
SUITE_DIR="${REPO_DIR}/experiments/king4p5_graphunet_full_${SUITE_TAG}"
MANIFEST_PATH="${SUITE_DIR}/launch_manifest.csv"

MANIFEST_CSV=${MANIFEST_CSV:-cache/manifest_fsaverage4_ukb6067.csv}
RESOLUTION=${RESOLUTION:-fsaverage4}
EDGE_CACHE_DIR=${EDGE_CACHE_DIR:-cache/templates_ukb_fsaverage4}
ANON_MANIFEST=${ANON_MANIFEST:-cache/manifest_fsaverage4_king4p5_anon.csv}
ANON_EDGE_CACHE_DIR=${ANON_EDGE_CACHE_DIR:-cache/templates_ukb_fsaverage4}
ANON_BUNDLE_DIR=${ANON_BUNDLE_DIR:-downstream/new/king_anonymous_over4p5_discovery_bundle}
Z_ROOT=${Z_ROOT:-downstream/outputs/z_graph_king4p5_anon}
HE_ROOT=${HE_ROOT:-downstream/outputs/he_king4p5_anon}
EPOCHS=${EPOCHS:-100}
NUM_WORKERS=${NUM_WORKERS:-4}

GPU_R060=${GPU_R060:-5}
GPU_R070=${GPU_R070:-6}
GPU_R060_SEED43=${GPU_R060_SEED43:-7}

mkdir -p "${SUITE_DIR}"

COMMON_TRAIN_ARGS=(
  python
  src/train/train_graph_branch.py
  --manifest "${MANIFEST_CSV}"
  --res "${RESOLUTION}"
  --input_mode main5
  --normalization_mode global_train
  --recon_loss wmse
  --recon_on all
  --lambda_recon 1.0
  --lambda_ce 0.2
  --use_ce 0
  --hidden_dim 32
  --dims 32 64 128 256 128 64 32
  --dropout 0.1
  --lr 0.001
  --weight_decay 0.0001
  --epochs "${EPOCHS}"
  --val_ratio 0.1
  --test_ratio 0.1
  --num_workers "${NUM_WORKERS}"
  --edge_cache_dir "${EDGE_CACHE_DIR}"
  --backbone_type graph_unet
  --graph_unet_depth 2
  --graph_unet_pool_ratios 0.8 0.8
  --pool_mode mean
  --mask_strategy hybrid
  --patch_hops 2
  --patch_num_seeds 16
  --hybrid_patch_fraction 0.7
)

# Format: run_name|gpu|extra train args
RUN_SPECS=(
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100|${GPU_R060}|--seed 42 --mask_ratio 0.60"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r070_p080_full100|${GPU_R070}|--seed 42 --mask_ratio 0.70"
  "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_seed43_full100|${GPU_R060_SEED43}|--seed 43 --mask_ratio 0.60"
)

printf "run_name,gpu,session_name,out_dir,z_graph_dir,he_dir,log_path,status,notes\n" > "${MANIFEST_PATH}"

for spec in "${RUN_SPECS[@]}"; do
  IFS="|" read -r run_name gpu extra <<< "${spec}"
  session_name="king4p5_${run_name}"
  out_dir="runs/${run_name}"
  z_graph_dir="${Z_ROOT}/${run_name}"
  he_dir="${HE_ROOT}/${run_name}"
  log_path="${SUITE_DIR}/${run_name}.log"

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${run_name}" "${gpu}" "${session_name}" "${out_dir}" "${z_graph_dir}" "${he_dir}" "${log_path}" "session_exists" "" \
      >> "${MANIFEST_PATH}"
    continue
  fi

  if [[ -f "${REPO_DIR}/${he_dir}/arena_manifest.json" ]]; then
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${run_name}" "${gpu}" "${session_name}" "${out_dir}" "${z_graph_dir}" "${he_dir}" "${log_path}" "already_done" "" \
      >> "${MANIFEST_PATH}"
    continue
  fi

  read -r -a extra_args <<< "${extra}"
  train_cmd=(
    env
    GPU_SPEC="device=${gpu}"
    PROJECT_MOUNT="/DATA"
    CONTAINER_PROJECT_ROOT="/DATA"
    CONTAINER_REPO_DIR="${REPO_DIR}"
    "${REPO_DIR}/scripts/docker_run.sh"
    "${COMMON_TRAIN_ARGS[@]}"
    "${extra_args[@]}"
    --out_dir "${out_dir}"
  )
  extract_cmd=(
    env
    GPU_SPEC="device=${gpu}"
    PROJECT_MOUNT="/DATA"
    CONTAINER_PROJECT_ROOT="/DATA"
    CONTAINER_REPO_DIR="${REPO_DIR}"
    "${REPO_DIR}/scripts/docker_run.sh"
    python src/train/extract_z_graph.py
    --manifest "${ANON_MANIFEST}"
    --ckpt "${out_dir}/ckpt_best_recon.pt"
    --res fsaverage4
    --edge_cache_dir "${ANON_EDGE_CACHE_DIR}"
    --device cuda
    --num_workers "${NUM_WORKERS}"
    --skip_umap 1
    --out_dir "${z_graph_dir}"
  )
  he_cmd=(
    env
    GPU_SPEC="device=${gpu}"
    PROJECT_MOUNT="/DATA"
    CONTAINER_PROJECT_ROOT="/DATA"
    CONTAINER_REPO_DIR="${REPO_DIR}"
    "${REPO_DIR}/scripts/docker_run.sh"
    python "${ANON_BUNDLE_DIR}/run_fast_h2_king_standard_anonymous.py"
    --anon_bundle_dir "${ANON_BUNDLE_DIR}"
    --phenotype_npy_dir "${z_graph_dir}"
    --embedding_id_mode anon
    --n_pca 128
    --output_root "${he_dir}"
  )
  summary_cmd=(
    python
    scripts/summarize_all_king4p5_outputs.py
    --out downstream/outputs/king4p5_all_downstream_summary.tsv
  )
  audit_cmd=(
    python
    scripts/audit_king4p5_checkpoint_coverage.py
    --out downstream/outputs/king4p5_checkpoint_coverage.tsv
  )

  printf -v train_shell "%q " "${train_cmd[@]}"
  printf -v extract_shell "%q " "${extract_cmd[@]}"
  printf -v he_shell "%q " "${he_cmd[@]}"
  printf -v summary_shell "%q " "${summary_cmd[@]}"
  printf -v audit_shell "%q " "${audit_cmd[@]}"

  tmux new-session -d -s "${session_name}" \
    "cd ${REPO_DIR@Q} && mkdir -p ${z_graph_dir@Q} ${he_dir@Q} && ${train_shell}> ${log_path@Q} 2>&1 && ${extract_shell}>> ${log_path@Q} 2>&1 && ${he_shell}>> ${log_path@Q} 2>&1 && ${summary_shell}>> ${log_path@Q} 2>&1 && ${audit_shell}>> ${log_path@Q} 2>&1"
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${run_name}" "${gpu}" "${session_name}" "${out_dir}" "${z_graph_dir}" "${he_dir}" "${log_path}" "launched" "${extra}" \
    >> "${MANIFEST_PATH}"
done

echo "Launched KING GraphUNet full suite manifest: ${MANIFEST_PATH}"
cat "${MANIFEST_PATH}"
