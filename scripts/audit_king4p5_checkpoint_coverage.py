#!/usr/bin/env python3
"""Audit which trained checkpoints have KING over4p5 downstream outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def cfg_default(cfg: dict[str, Any], key: str, default: str) -> str:
    value = cfg.get(key)
    if value in (None, ""):
        nested = cfg.get("normalization", {})
        if isinstance(nested, dict):
            value = nested.get(key)
    return str(value) if value not in (None, "") else default


def h2_from_manifest(path: Path) -> str:
    if not path.is_file():
        return ""
    return str(read_json(path).get("mean_h2_pca_weighted", ""))


def load_ablation_eval_map(summary_path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if not summary_path.is_file():
        return mapping
    with summary_path.open("r", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mapping[row["ckpt"]] = row
    return mapping


def run_dirs(repo: Path) -> list[Path]:
    dirs = [p.parent for p in (repo / "runs").glob("*/run_config.json")]
    dirs += [p.parent for p in (repo / "ablation_result").glob("**/run_config.json")]
    return sorted(dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("downstream/outputs/king4p5_checkpoint_coverage.tsv"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    main_he = repo / "downstream/outputs/he_king4p5_anon"
    ablation_he = repo / "downstream/outputs/he_king4p5_anon_ablation_fsaverage4"
    ablation_map = load_ablation_eval_map(ablation_he / "summary.tsv")

    rows: list[dict[str, Any]] = []
    for rd in run_dirs(repo):
        cfg_path = rd / "run_config.json"
        cfg = read_json(cfg_path)
        ckpt_best = rd / "ckpt_best_recon.pt"
        if not ckpt_best.is_file():
            continue

        rel_dir = str(rd.relative_to(repo))
        run = rd.name
        res = str(cfg.get("res", ""))
        is_ablation = rel_dir.startswith("ablation_result/")
        eligible = res == "fsaverage4"

        if is_ablation:
            rel_ckpt = str(ckpt_best.relative_to(repo))
            eval_row = ablation_map.get(rel_ckpt, {})
            he_done = bool(eval_row)
            h2 = eval_row.get("h2", "")
            he_dir = str((ablation_he / eval_row["run"]).relative_to(repo)) if he_done else ""
            eval_run = eval_row.get("run", "")
        else:
            manifest = main_he / run / "arena_manifest.json"
            he_done = manifest.is_file()
            h2 = h2_from_manifest(manifest)
            he_dir = str(manifest.parent.relative_to(repo)) if he_done else ""
            eval_run = run if he_done else ""

        if not eligible:
            status = "not_current_king_fsaverage4"
        elif he_done:
            status = "evaluated"
        elif "smoke" in run or cfg.get("num_subjects") in (None, "", 100):
            status = "eligible_low_priority_smoke_or_small_train"
        else:
            status = "eligible_missing_downstream"

        rows.append(
            {
                "run": run,
                "run_dir": rel_dir,
                "ckpt_best": str(ckpt_best.relative_to(repo)),
                "has_ckpt_last": str((rd / "ckpt_last.pt").is_file()),
                "status": status,
                "eligible_current_king_fsaverage4": str(eligible),
                "he_done": str(he_done),
                "h2": h2,
                "eval_run": eval_run,
                "he_dir": he_dir,
                "res": res,
                "num_train_subjects": cfg.get("num_subjects", ""),
                "input_mode": cfg.get("input_mode", ""),
                "normalization_mode": cfg_default(cfg, "normalization_mode", "per_hemi(default)"),
                "xyz_norm_mode": cfg_default(cfg, "xyz_norm_mode", "subject_rms(default)"),
                "morph_norm_mode": cfg_default(cfg, "morph_norm_mode", "subject_zscore(default)"),
                "graph_global_mode": cfg_default(cfg, "graph_global_mode", "none"),
                "graph_covariate_mode": cfg_default(cfg, "graph_covariate_mode", "none"),
                "mask_strategy": cfg.get("mask_strategy", ""),
                "mask_ratio": cfg.get("mask_ratio", ""),
                "backbone_type": cfg.get("backbone_type", "gcn(default)"),
                "pool_mode": cfg.get("pool_mode", "mean(default)"),
                "mask_fill": cfg.get("mask_fill", "zero(default)"),
                "decoder_remask_ratio": cfg.get("decoder_remask_ratio", 0.0),
                "latent_consistency_weight": cfg.get("latent_consistency_weight", 0.0),
            }
        )

    rows.sort(
        key=lambda row: (
            row["status"] != "eligible_missing_downstream",
            row["status"],
            row["run_dir"],
        )
    )
    fields = [
        "run",
        "run_dir",
        "ckpt_best",
        "has_ckpt_last",
        "status",
        "eligible_current_king_fsaverage4",
        "he_done",
        "h2",
        "eval_run",
        "he_dir",
        "res",
        "num_train_subjects",
        "input_mode",
        "normalization_mode",
        "xyz_norm_mode",
        "morph_norm_mode",
        "graph_global_mode",
        "graph_covariate_mode",
        "mask_strategy",
        "mask_ratio",
        "backbone_type",
        "pool_mode",
        "mask_fill",
        "decoder_remask_ratio",
        "latent_consistency_weight",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
