"""Utilities for saving reconstruction evaluation artifacts."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_per_dim_recon_csv(
    path: str,
    feature_names: List[str],
    split_metrics: Dict[str, Dict[str, object]],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "feature", "is_active", "masked_mse", "masked_vertex_count"])
        for split_name, metrics in split_metrics.items():
            per_dim_mse = metrics.get("per_dim_mse", [])
            per_dim_count = metrics.get("per_dim_count", [])
            active_dims = set(int(idx) for idx in metrics.get("active_dims", []))
            for idx, feature_name in enumerate(feature_names):
                writer.writerow(
                    [
                        split_name,
                        feature_name,
                        int(idx in active_dims),
                        float(per_dim_mse[idx]) if idx < len(per_dim_mse) else "",
                        int(per_dim_count[idx]) if idx < len(per_dim_count) else "",
                    ]
                )


def save_recon_examples(
    out_dir: str,
    examples: List[Dict[str, object]],
    feature_names: List[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    profile_rows: List[List[object]] = []

    for example in examples:
        example_dir = os.path.join(out_dir, f"{example['split']}_{example['sid']}")
        os.makedirs(example_dir, exist_ok=True)

        example_manifest = {
            "split": example["split"],
            "sid": example["sid"],
            "res": example["res"],
            "input_mode": example["input_mode"],
            "feature_names": feature_names,
            "hemis": {},
        }

        for hemi_name, hemi_payload in example["hemis"].items():
            original = np.asarray(hemi_payload["original"], dtype=np.float32)
            masked = np.asarray(hemi_payload["masked"], dtype=np.float32)
            recon = np.asarray(hemi_payload["recon"], dtype=np.float32)
            mask = np.asarray(hemi_payload["mask"], dtype=bool)
            mask_idx = mask.astype(bool)

            np.save(os.path.join(example_dir, f"{hemi_name}_original.npy"), original)
            np.save(os.path.join(example_dir, f"{hemi_name}_masked.npy"), masked)
            np.save(os.path.join(example_dir, f"{hemi_name}_recon.npy"), recon)
            np.save(os.path.join(example_dir, f"{hemi_name}_mask.npy"), mask)

            example_manifest["hemis"][hemi_name] = {
                "original": f"{hemi_name}_original.npy",
                "masked": f"{hemi_name}_masked.npy",
                "recon": f"{hemi_name}_recon.npy",
                "mask": f"{hemi_name}_mask.npy",
                "masked_vertices": int(mask.sum()),
            }

            if int(mask.sum()) == 0:
                continue

            orig_masked = original[mask_idx]
            masked_masked = masked[mask_idx]
            recon_masked = recon[mask_idx]
            mse = ((recon_masked - orig_masked) ** 2).mean(axis=0)

            for feat_idx, feature_name in enumerate(feature_names):
                profile_rows.append(
                    [
                        example["split"],
                        example["sid"],
                        hemi_name,
                        feature_name,
                        float(orig_masked[:, feat_idx].mean()),
                        float(masked_masked[:, feat_idx].mean()),
                        float(recon_masked[:, feat_idx].mean()),
                        float(mse[feat_idx]),
                        int(mask.sum()),
                    ]
                )

        write_json(os.path.join(example_dir, "metadata.json"), example_manifest)
        manifest.append(example_manifest)

    write_json(os.path.join(out_dir, "example_manifest.json"), {"examples": manifest})

    with open(os.path.join(out_dir, "feature_profile_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "sid",
                "hemi",
                "feature",
                "original_masked_mean",
                "masked_input_mean",
                "recon_masked_mean",
                "masked_mse",
                "masked_vertex_count",
            ]
        )
        writer.writerows(profile_rows)


__all__ = [
    "save_recon_examples",
    "write_json",
    "write_per_dim_recon_csv",
]
