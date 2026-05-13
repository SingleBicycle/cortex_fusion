# GraphUNet KING4.5 Checkpoint

This folder contains one small, GitHub-trackable GraphUNet surface-branch checkpoint selected for downstream KING over4.5 HE performance.

## Included Checkpoint

- `ukb6067_graphunet_r060_p080_epoch100_best_recon.pt`
- Source run: `ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100`
- Source checkpoint: `runs/ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100/snapshots/ckpt_epoch100_best_recon.pt`
- Stored model epoch inside checkpoint: `98`
- SHA256: `d811f42190523cc7e9803608c4f0982e02b9433435d618e74db74867c9d6cb93`

## Model Configuration

- `input_mode=main5`
- `normalization_mode=global_train`
- `backbone_type=graph_unet`
- `graph_unet_depth=2`
- `graph_unet_pool_ratios=[0.8, 0.8]`
- `pool_mode=mean`
- `mask_strategy=hybrid`
- `mask_ratio=0.60`
- `patch_hops=2`
- `patch_num_seeds=16`
- `hidden_dim=32`
- `dims=[32, 64, 128, 256, 128, 64, 32]`

## Downstream HE Evidence

The checkpoint was evaluated after extracting 128D `z_graph` embeddings on the anonymous KING over4.5 discovery intersection.

- `n_samples=2119`
- `n_phenotype_columns=128`
- `mean_h2_pca_weighted=0.5037454617584528`
- HE run id: `ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100__epoch100_best_recon`
- HE timestamp: `2026-04-29T19:52:44Z`

`snapshot_manifest.tsv` is included to document the saved 10/25/50/75/100 epoch snapshot family from the source run.
