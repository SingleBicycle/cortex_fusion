#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${1:-${REPO_ROOT}/runs/graph_branch_ssl_main5}"
EMBED_DIR="${2:-${REPO_ROOT}/z_graph_cache}"
OUT_DIR="${3:-${REPO_ROOT}/handoff_to_tian}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

copy_or_placeholder() {
  local src="$1"
  local dst="$2"
  local placeholder="$3"
  if [[ -e "${src}" ]]; then
    cp -R "${src}" "${dst}"
  else
    printf '%s\n' "${placeholder}" > "${dst}"
  fi
}

copy_or_placeholder \
  "${REPO_ROOT}/README.md" \
  "${OUT_DIR}/README.md" \
  "README placeholder: ${REPO_ROOT}/README.md was not found."

copy_or_placeholder \
  "${REPO_ROOT}/docs/hparam_rationale.md" \
  "${OUT_DIR}/hparam_rationale.md" \
  "Hyperparameter rationale placeholder: docs/hparam_rationale.md was not found."

copy_or_placeholder \
  "${RUN_DIR}/recon_metrics.json" \
  "${OUT_DIR}/recon_metrics.json" \
  "Reconstruction metrics placeholder: expected ${RUN_DIR}/recon_metrics.json"

copy_or_placeholder \
  "${RUN_DIR}/per_dim_recon_mse.csv" \
  "${OUT_DIR}/per_dim_recon_mse.csv" \
  "Per-dimension reconstruction CSV placeholder: expected ${RUN_DIR}/per_dim_recon_mse.csv"

copy_or_placeholder \
  "${RUN_DIR}/masking_summary.json" \
  "${OUT_DIR}/masking_summary.json" \
  "Masking summary placeholder: expected ${RUN_DIR}/masking_summary.json"

if [[ -d "${RUN_DIR}/recon_examples" ]]; then
  cp -R "${RUN_DIR}/recon_examples" "${OUT_DIR}/recon_examples"
else
  mkdir -p "${OUT_DIR}/recon_examples"
  printf '%s\n' "Reconstruction examples placeholder: expected ${RUN_DIR}/recon_examples" \
    > "${OUT_DIR}/recon_examples/PLACEHOLDER.txt"
fi

if [[ -f "${EMBED_DIR}/pca_scatter.png" ]]; then
  cp "${EMBED_DIR}/pca_scatter.png" "${OUT_DIR}/pca_scatter.png"
else
  printf '%s\n' "PCA figure placeholder: expected ${EMBED_DIR}/pca_scatter.png" \
    > "${OUT_DIR}/PCA_FIGURE_PLACEHOLDER.txt"
fi

if [[ -f "${RUN_DIR}/ckpt_best_recon.pt" ]]; then
  printf '%s\n' "${RUN_DIR}/ckpt_best_recon.pt" > "${OUT_DIR}/CHECKPOINT_PATH.txt"
else
  printf '%s\n' "<replace-with-path-to-ckpt_best_recon.pt>" > "${OUT_DIR}/CHECKPOINT_PATH.txt"
fi

if [[ -f "${RUN_DIR}/run_config.json" ]]; then
  python - <<'PY' "${RUN_DIR}/run_config.json" "${OUT_DIR}/training_command.sh"
import json
import os
import sys

run_config_path = sys.argv[1]
out_path = sys.argv[2]

with open(run_config_path) as f:
    cfg = json.load(f)

cmd = f"""#!/usr/bin/env bash
python src/train/train_graph_branch.py \\
  --manifest {cfg.get('manifest', 'cache/manifest_fsaverage6.csv')} \\
  --res {cfg.get('res', 'fsaverage6')} \\
  --input_mode {cfg.get('input_mode', 'main5')} \\
  --mask_strategy {cfg.get('mask_strategy', 'hybrid')} \\
  --mask_ratio {cfg.get('mask_ratio', 0.35)} \\
  --patch_hops {cfg.get('patch_hops', 2)} \\
  --patch_num_seeds {cfg.get('patch_num_seeds', 16)} \\
  --recon_loss {cfg.get('recon_loss', 'wmse')} \\
  --recon_on {cfg.get('recon_on', 'all')} \\
  --lambda_recon {cfg.get('lambda_recon', 1.0)} \\
  --lambda_ce {cfg.get('lambda_ce', 0.2)} \\
  --use_ce {cfg.get('use_ce', 0)} \\
  --hidden_dim {cfg.get('hidden_dim', 32)} \\
  --dims {' '.join(str(x) for x in cfg.get('dims', [32, 64, 128, 256, 128, 64, 32]))} \\
  --dropout {cfg.get('dropout', 0.1)} \\
  --lr {cfg.get('lr', 1e-3)} \\
  --weight_decay {cfg.get('weight_decay', 1e-4)} \\
  --epochs {cfg.get('epochs', 100)} \\
  --edge_cache_dir {cfg.get('edge_cache_dir', 'cache/templates')} \\
  --out_dir <replace-with-output-run-dir>
"""

with open(out_path, "w") as f:
    f.write(cmd)
os.chmod(out_path, 0o755)
PY
else
  cat > "${OUT_DIR}/training_command.sh" <<'EOF'
#!/usr/bin/env bash
python src/train/train_graph_branch.py \
  --manifest cache/manifest_fsaverage6.csv \
  --res fsaverage6 \
  --input_mode main5 \
  --mask_strategy hybrid \
  --mask_ratio 0.35 \
  --patch_hops 2 \
  --patch_num_seeds 16 \
  --recon_loss wmse \
  --recon_on all \
  --lambda_recon 1.0 \
  --lambda_ce 0.2 \
  --use_ce 0 \
  --hidden_dim 32 \
  --dims 32 64 128 256 128 64 32 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --epochs 100 \
  --edge_cache_dir cache/templates \
  --out_dir <replace-with-output-run-dir>
EOF
  chmod +x "${OUT_DIR}/training_command.sh"
fi

cat > "${OUT_DIR}/HANDOFF_NOTES.txt" <<EOF
Run directory: ${RUN_DIR}
Embedding directory: ${EMBED_DIR}

Included:
- README
- training command
- checkpoint path placeholder
- PCA figure or placeholder
- reconstruction metrics or placeholder
- per-dimension reconstruction CSV or placeholder
- masking summary or placeholder
- reconstruction examples or placeholder
- hyperparameter rationale doc
EOF

printf 'Created handoff folder: %s\n' "${OUT_DIR}"
