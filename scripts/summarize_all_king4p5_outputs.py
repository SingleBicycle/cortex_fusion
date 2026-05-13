#!/usr/bin/env python3
"""Build one table for all KING over4p5 downstream HE outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def norm_default(cfg: dict[str, Any], key: str, default: str) -> str:
    value = cfg.get(key)
    if value is None or value == "":
        nested = cfg.get("normalization", {})
        if isinstance(nested, dict):
            value = nested.get(key)
    return str(value) if value not in (None, "") else default


def read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "recon_metrics.json"
    if not metrics_path.is_file():
        return {}
    return read_json(metrics_path)


def metric_value(metrics: dict[str, Any], split: str, key: str) -> Any:
    return metrics.get("splits", {}).get(split, {}).get(key, "")


def load_ablation_map(summary_path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if not summary_path.is_file():
        return mapping
    with summary_path.open("r", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mapping[row["run"]] = row
    return mapping


def row_for_manifest(
    repo: Path,
    manifest_path: Path,
    source: str,
    ablation_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    run = manifest_path.parent.name
    manifest = read_json(manifest_path)

    if source == "ablation_fsaverage4":
        mapped = ablation_map.get(run, {})
        run_dir = repo / mapped.get("run_dir", "")
        ckpt = mapped.get("ckpt", "")
        group = mapped.get("group", "ablation")
    else:
        run_dir = repo / "runs" / run
        ckpt = str(Path("runs") / run / "ckpt_best_recon.pt")
        group = "main"

    cfg_path = run_dir / "run_config.json"
    cfg = read_json(cfg_path) if cfg_path.is_file() else {}
    metrics = read_metrics(run_dir)

    h2 = manifest.get("mean_h2_pca_weighted", "")
    best_epoch = metrics.get("best_epoch", cfg.get("best_epoch", ""))

    return {
        "run": run,
        "source": source,
        "group": group,
        "h2": h2,
        "n_samples": manifest.get("n_samples", ""),
        "n_features": manifest.get("n_phenotype_columns", ""),
        "res": cfg.get("res", ""),
        "num_train_subjects": cfg.get("num_subjects", ""),
        "input_mode": cfg.get("input_mode", ""),
        "normalization_mode": norm_default(cfg, "normalization_mode", "per_hemi(default)"),
        "xyz_norm_mode": norm_default(cfg, "xyz_norm_mode", "subject_rms(default)"),
        "morph_norm_mode": norm_default(cfg, "morph_norm_mode", "subject_zscore(default)"),
        "graph_global_mode": norm_default(cfg, "graph_global_mode", "none"),
        "graph_covariate_mode": norm_default(cfg, "graph_covariate_mode", "none"),
        "append_graph_covariates_to_z": cfg.get("append_graph_covariates_to_z", ""),
        "mask_strategy": cfg.get("mask_strategy", ""),
        "mask_ratio": cfg.get("mask_ratio", ""),
        "hybrid_patch_fraction": cfg.get("hybrid_patch_fraction", ""),
        "region_num_components": cfg.get("region_num_components", ""),
        "region_grow_mode": cfg.get("region_grow_mode", ""),
        "region_seed_min_hops": cfg.get("region_seed_min_hops", ""),
        "backbone_type": cfg.get("backbone_type", "gcn(default)"),
        "pool_mode": cfg.get("pool_mode", "mean(default)"),
        "hidden_dim": cfg.get("hidden_dim", ""),
        "dims": " ".join(map(str, cfg.get("dims", []))) if isinstance(cfg.get("dims"), list) else cfg.get("dims", ""),
        "mask_fill": cfg.get("mask_fill", "zero(default)"),
        "decoder_remask_ratio": cfg.get("decoder_remask_ratio", 0.0),
        "decoder_remask_strategy": cfg.get("decoder_remask_strategy", ""),
        "latent_consistency_weight": cfg.get("latent_consistency_weight", 0.0),
        "best_epoch": best_epoch,
        "best_selection_recon_loss": metrics.get("best_selection_recon_loss", ""),
        "val_weighted_mse": metric_value(metrics, "val", "weighted_mse"),
        "test_weighted_mse": metric_value(metrics, "test", "weighted_mse"),
        "val_realized_mask_ratio": metric_value(metrics, "val", "realized_mask_ratio"),
        "test_realized_mask_ratio": metric_value(metrics, "test", "realized_mask_ratio"),
        "timestamp_utc": manifest.get("timestamp_utc", ""),
        "ckpt": ckpt,
        "train_run_dir": str(run_dir.relative_to(repo)) if run_dir.is_absolute() and repo in run_dir.parents else str(run_dir),
        "he_dir": str(manifest_path.parent.relative_to(repo)),
        "phenotype_npy_dir": manifest.get("phenotype_npy_dir", ""),
        "arena_manifest": str(manifest_path.relative_to(repo)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("downstream/outputs/king4p5_all_downstream_summary.tsv"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    roots = [
        ("main", repo / "downstream/outputs/he_king4p5_anon"),
        ("ablation_fsaverage4", repo / "downstream/outputs/he_king4p5_anon_ablation_fsaverage4"),
    ]
    ablation_map = load_ablation_map(roots[1][1] / "summary.tsv")

    rows: list[dict[str, Any]] = []
    for source, root in roots:
        for manifest_path in sorted(root.glob("*/arena_manifest.json")):
            rows.append(row_for_manifest(repo, manifest_path, source, ablation_map))

    rows.sort(
        key=lambda row: float(row["h2"]) if row["h2"] not in ("", None) else float("-inf"),
        reverse=True,
    )

    fields = [
        "run",
        "source",
        "group",
        "h2",
        "n_samples",
        "n_features",
        "res",
        "num_train_subjects",
        "input_mode",
        "normalization_mode",
        "xyz_norm_mode",
        "morph_norm_mode",
        "graph_global_mode",
        "graph_covariate_mode",
        "append_graph_covariates_to_z",
        "mask_strategy",
        "mask_ratio",
        "hybrid_patch_fraction",
        "region_num_components",
        "region_grow_mode",
        "region_seed_min_hops",
        "backbone_type",
        "pool_mode",
        "hidden_dim",
        "dims",
        "mask_fill",
        "decoder_remask_ratio",
        "decoder_remask_strategy",
        "latent_consistency_weight",
        "best_epoch",
        "best_selection_recon_loss",
        "val_weighted_mse",
        "test_weighted_mse",
        "val_realized_mask_ratio",
        "test_realized_mask_ratio",
        "timestamp_utc",
        "ckpt",
        "train_run_dir",
        "he_dir",
        "phenotype_npy_dir",
        "arena_manifest",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
