# cortex_fusion

Graph-branch SSL pipeline for cortical template-aligned surface graphs. The surface graph construction, ADGCN encoder backbone, dataset manifest flow, and cached template edge index flow are preserved. The main objective is now masked feature reconstruction, and the exported subject embedding is a pooled 128D `z_graph`.

## Dependencies
- `torch`
- `torch_geometric`
- `nibabel`
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib` optional for plotting `pca_scatter.png` with Matplotlib; the extractor has a built-in PNG fallback
- `scikit-learn` optional for PCA; the extractor falls back to NumPy SVD

## Docker


1. Build the image:
```bash
cd /DATA/zihao/projects/medical/cortex_fusion
scripts/docker_build.sh
```

2. Start an interactive shell with GPU access and the host project tree mounted at the same absolute path inside the container:
```bash
cd /DATA/zihao/projects/medical/cortex_fusion
scripts/docker_run.sh
```

3. Run a training command directly:
```bash
cd /DATA/zihao/projects/medical/cortex_fusion
scripts/docker_run.sh python src/train/train_graph_branch.py \
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
  --out_dir runs/graph_branch_ssl_main5
```

Useful environment overrides:
- `IMAGE_NAME=my-tag scripts/docker_build.sh`
- `GPU_SPEC=device=0 scripts/docker_run.sh`
- `SHM_SIZE=32g scripts/docker_run.sh`
- `PROJECT_MOUNT=/some/other/medical-root scripts/docker_run.sh`

The image is based on `nvidia/cuda:12.8.1-runtime-ubuntu22.04`, installs PyTorch `2.7.1` with CUDA `12.8`, and keeps `PYTHONPATH` pointed at the repo root so the existing `src/...` commands continue to work unchanged.

## Input Modes
- `baseline8`: `[pial_x, pial_y, pial_z, white_x, white_y, white_z, thickness, curvature]`
- `main5`: `[mid_x, mid_y, mid_z, thickness, curvature]`, where `mid_xyz = 0.5 * (pial_norm + white_norm)`
- `ablation2`: `[thickness, curvature]`

Recommended default: `main5`

## Normalization
- XYZ channels are centered using the combined pial+white vertex cloud and scaled by subject-level RMS magnitude.
- Morphometry channels (`thickness`, `curvature`) are normalized independently with per-channel z-score.
- Reconstruction targets use the normalized, unmasked feature tensor for the selected input mode.

## Mask Strategies
- `random`: sample vertices uniformly at random
- `patch`: sample seed vertices and expand local graph neighborhoods with BFS / k-hop growth
- `hybrid`: patch masking plus random fill to the target mask ratio

For masked vertices, the selected reconstruction dimensions are zeroed in the input tensor and reconstructed from `X_target`.

## Commands
1. Build manifest:
```bash
python src/data/build_manifest.py \
  --root /DATA/zihao/projects/medical/dataset \
  --mode low \
  --res fsaverage6 \
  --out cache/manifest_fsaverage6.csv
```

2. Cache template edge index:
```bash
python src/data/cache_edge_index.py \
  --manifest cache/manifest_fsaverage6.csv \
  --res fsaverage6 \
  --out_dir cache/templates
```

3. Train reconstruction-first SSL graph branch:
```bash
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
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --edge_cache_dir cache/templates \
  --out_dir runs/graph_branch_ssl_main5
```

4. Extract pooled 128D subject embeddings and PCA outputs:
```bash
python src/train/extract_z_graph.py \
  --manifest cache/manifest_fsaverage6.csv \
  --ckpt runs/graph_branch_ssl_main5/ckpt_best_recon.pt \
  --res fsaverage6 \
  --edge_cache_dir cache/templates \
  --device cpu \
  --out_dir z_graph_cache
```

## Training Outputs
- `run_config.json`: run arguments, schema info, masking config, reconstruction weights, and model config
- `train_log.csv`: epoch-level total loss, reconstruction loss, CE loss, and per-dimension reconstruction loss
- `ckpt_last.pt`
- `ckpt_best_recon.pt`
- `recon_metrics.json`: validation/test reconstruction metrics for the best checkpoint
- `per_dim_recon_mse.csv`: masked-vertex reconstruction MSE by split and feature dimension
- `masking_summary.json`: requested vs realized masking statistics for validation/test
- `recon_examples/`: small qualitative artifact bundle with original input, masked input, reconstruction, boolean mask, and feature-profile summaries

## Extraction Outputs
- `<sid>.npy`: per-subject 128D embedding
- `embeddings.npy`: stacked `[N, 128]` embedding matrix
- `subject_ids.csv`
- `embeddings_with_sid.csv`
- `pca_2d.csv`
- `pca_scatter.png`

## Docs And Handoff
- `docs/hparam_rationale.md`: short literature-backed rationale for the default SSL configuration and initial ablation plan
- `scripts/package_for_tian.sh`: creates `handoff_to_tian/` with the current README, hparam note, training command, checkpoint pointer, PCA figure, and reconstruction artifacts

Example:
```bash
scripts/package_for_tian.sh \
  runs/graph_branch_ssl_main5 \
  z_graph_cache \
  handoff_to_tian
```
