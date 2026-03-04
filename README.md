# cortex_fusion

Graph-branch training pipeline (v1) for cortical template-aligned surfaces with subject-specific xyz/labels/morphometry.

## Dependencies
- `torch`
- `torch_geometric`
- `nibabel`
- `numpy`
- `pandas`
- `tqdm`

## Commands
1. Build manifest (default `fsaverage6`):
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

3. Train graph branch (MFM + optional CE):
```bash
python src/train/train_graph_branch.py \
  --manifest cache/manifest_fsaverage6.csv \
  --res fsaverage6 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --mask_ratio 0.3 \
  --alpha_ce 0.2 \
  --use_ce 1 \
  --edge_cache_dir cache/templates \
  --out_dir runs/graph_branch_v1
```

4. Extract subject embedding `z_graph`:
```bash
python src/train/extract_z_graph.py \
  --manifest cache/manifest_fsaverage6.csv \
  --ckpt runs/graph_branch_v1/ckpt_best_mfm.pt \
  --res fsaverage6 \
  --edge_cache_dir cache/templates \
  --out_dir z_graph_cache
```
