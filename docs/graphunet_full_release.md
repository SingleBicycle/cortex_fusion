# GraphUNet Surface Branch Release

This note records the GitHub-ready GraphUNet surface branch package in this repo.

## What Is Included

- `src/models/adgcn.py`: adds `GraphUNetEncoder`, backed by `torch_geometric.nn.models.GraphUNet`.
- `src/train/train_graph_branch.py`: accepts `--backbone_type graph_unet`, `--graph_unet_depth`, and `--graph_unet_pool_ratios`.
- `src/train/extract_z_graph.py`: restores GraphUNet model configuration from checkpoints before extracting 128D `z_graph`.
- `scripts/launch_king4p5_graphunet_full_suite.sh`: trains the full GraphUNet KING4.5 surface suite and then runs extraction plus HE evaluation.
- `scripts/check_graphunet_full_status.sh`: checks the three full GraphUNet runs.
- `scripts/watch_king4p5_checkpoint_snapshots.sh`: copies `ckpt_last.pt` and `ckpt_best_recon.pt` into fixed 10/25/50/75/100 epoch snapshots while long runs are active.
- `scripts/eval_king4p5_snapshot_h2_curve.sh`: evaluates saved 10/25/50/75/100 epoch snapshots.
- `scripts/summarize_king4p5_snapshot_curve.py`: summarizes snapshot HE results and plots curves.
- `checkpoints/graphunet_king4p5/ukb6067_graphunet_r060_p080_epoch100_best_recon.pt`: compact pretrained checkpoint with HE provenance.

## Full Training Command

The main launcher trains three comparable GraphUNet runs:

```bash
cd /DATA/zihao/projects/medical/cortex_fusion
SUITE_TAG=20260429_graphunet_full \
GPU_R060=5 GPU_R070=6 GPU_R060_SEED43=7 \
scripts/launch_king4p5_graphunet_full_suite.sh
```

For snapshot HE curves, run the watcher in a second shell while training is active:

```bash
cd /DATA/zihao/projects/medical/cortex_fusion
scripts/watch_king4p5_checkpoint_snapshots.sh
```

The strongest included checkpoint came from the `r060_p080` run:

```bash
python src/train/train_graph_branch.py \
  --manifest cache/manifest_fsaverage4_ukb6067.csv \
  --res fsaverage4 \
  --input_mode main5 \
  --normalization_mode global_train \
  --recon_loss wmse \
  --recon_on all \
  --lambda_recon 1.0 \
  --lambda_ce 0.2 \
  --use_ce 0 \
  --hidden_dim 32 \
  --dims 32 64 128 256 128 64 32 \
  --dropout 0.1 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --epochs 100 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --edge_cache_dir cache/templates_ukb_fsaverage4 \
  --backbone_type graph_unet \
  --graph_unet_depth 2 \
  --graph_unet_pool_ratios 0.8 0.8 \
  --pool_mode mean \
  --mask_strategy hybrid \
  --mask_ratio 0.60 \
  --patch_hops 2 \
  --patch_num_seeds 16 \
  --hybrid_patch_fraction 0.7 \
  --seed 42 \
  --out_dir runs/ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100
```

## Extract Embeddings From Included Checkpoint

```bash
python src/train/extract_z_graph.py \
  --manifest cache/manifest_fsaverage4_king4p5_anon.csv \
  --ckpt checkpoints/graphunet_king4p5/ukb6067_graphunet_r060_p080_epoch100_best_recon.pt \
  --res fsaverage4 \
  --edge_cache_dir cache/templates_ukb_fsaverage4 \
  --device cuda \
  --skip_umap 1 \
  --out_dir z_graph_cache_graphunet_r060_p080_epoch100_best_recon
```

## Checkpoint Evidence

The included checkpoint is small enough for GitHub and was selected from the snapshot HE curve:

- `mean_h2_pca_weighted=0.5037454617584528`
- `n_samples=2119`
- `n_phenotype_columns=128`
- source run: `ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100`
- source snapshot: `ckpt_epoch100_best_recon.pt`
- stored model epoch inside the best-reconstruction checkpoint: `98`

The downstream HE bundle and extracted subject embeddings are not committed because they are data/evaluation artifacts. Keep them local under `downstream/` and `z_graph_cache*/`.
