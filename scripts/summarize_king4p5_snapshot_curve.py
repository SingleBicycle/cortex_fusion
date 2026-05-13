#!/usr/bin/env python3
"""Summarize and plot snapshot HE results for KING over4p5 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _checkpoint_epoch(path: Path) -> int | str:
    if not path.is_file():
        return ""
    ckpt = torch.load(path, map_location="cpu")
    return int(ckpt.get("epoch", -1))


def _h2(path: Path) -> tuple[float | str, int | str, str]:
    manifest = path / "arena_manifest.json"
    if not manifest.is_file():
        return "", "", ""
    with manifest.open("r") as f:
        payload = json.load(f)
    return (
        float(payload["mean_h2_pca_weighted"]),
        int(payload.get("n_samples", 0)),
        str(payload.get("timestamp_utc", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_full100",
            "ukb6067_graphunet_mean_globalnorm_hybrid_r070_p080_full100",
            "ukb6067_graphunet_mean_globalnorm_hybrid_r060_p080_seed43_full100",
        ],
    )
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 25, 50, 75, 100])
    parser.add_argument("--kinds", nargs="+", default=["last", "best_recon"])
    parser.add_argument(
        "--he-root",
        type=Path,
        default=Path("downstream/outputs/he_king4p5_anon_snapshots"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("downstream/outputs/king4p5_snapshot_h2_curve.tsv"),
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/king4p5_snapshot_h2_curve"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    rows: list[dict[str, object]] = []
    for run in args.runs:
        for target in args.targets:
            for kind in args.kinds:
                ckpt_name = f"ckpt_epoch{target:03d}_{kind}.pt"
                ckpt = repo / "runs" / run / "snapshots" / ckpt_name
                eval_name = f"{run}__epoch{target:03d}_{kind}"
                h2, n_samples, timestamp = _h2(repo / args.he_root / eval_name)
                rows.append(
                    {
                        "run": run,
                        "target_epoch": target,
                        "ckpt_kind": kind,
                        "stored_epoch": _checkpoint_epoch(ckpt),
                        "ckpt": str(ckpt.relative_to(repo)) if ckpt.is_file() else "",
                        "h2": h2,
                        "n_samples": n_samples,
                        "timestamp_utc": timestamp,
                        "he_dir": str((repo / args.he_root / eval_name).relative_to(repo)),
                    }
                )

    args.out = repo / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "target_epoch", "ckpt_kind", "stored_epoch", "ckpt", "h2", "n_samples", "timestamp_utc", "he_dir"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    complete_rows = [r for r in rows if r["h2"] != ""]
    if not complete_rows:
        print("no completed h2 rows yet; skipping plot")
        return

    fig_dir = repo / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        plt.figure(figsize=(8.5, 5.0))
        plotted = False
        for run in args.runs:
            run_rows = [
                r for r in complete_rows
                if r["run"] == run and r["ckpt_kind"] == kind
            ]
            if not run_rows:
                continue
            run_rows.sort(key=lambda r: int(r["target_epoch"]))
            x = [int(r["target_epoch"]) for r in run_rows]
            y = [float(r["h2"]) for r in run_rows]
            label = run.replace("ukb6067_graphunet_mean_globalnorm_hybrid_", "")
            label = label.replace("_full100", "").replace("_p080", " p0.8")
            plt.plot(x, y, marker="o", linewidth=2.0, label=label)
            plotted = True
        if not plotted:
            continue
        plt.axhline(0.41521712716552805, linestyle="--", color="0.35", linewidth=1.5, label="previous GCN ref 0.415")
        plt.xlabel("Training snapshot target epoch")
        plt.ylabel("PCA-weighted HE h2")
        plt.title(f"GraphUNet snapshot h2 curve ({kind})")
        plt.xticks(args.targets)
        plt.ylim(bottom=0.0)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        png = fig_dir / f"graphunet_snapshot_h2_curve_{kind}.png"
        pdf = fig_dir / f"graphunet_snapshot_h2_curve_{kind}.pdf"
        plt.savefig(png, dpi=180)
        plt.savefig(pdf)
        plt.close()
        print(f"wrote {png}")
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
