#!/usr/bin/env python3
"""Plot train/validation reconstruction loss curves from train_log.csv."""

from __future__ import annotations

import argparse
import csv
import os
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reconstruction loss curves from train_log.csv")
    parser.add_argument("--train_log", type=str, required=True, help="Path to train_log.csv")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--title", type=str, default="Reconstruction Loss Curve")
    return parser.parse_args()


def _read_series(path: str) -> tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    train_recon: List[float] = []
    val_recon: List[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "train_recon_loss", "val_recon_loss"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_recon.append(float(row["train_recon_loss"]))
            val_recon.append(float(row["val_recon_loss"]))

    if not epochs:
        raise ValueError(f"No rows found in {path}")
    return epochs, train_recon, val_recon


def main() -> None:
    args = parse_args()
    train_log = os.path.abspath(args.train_log)
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(train_log), "recon_loss_curve.png")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs, train_recon, val_recon = _read_series(train_log)
    best_idx = min(range(len(val_recon)), key=val_recon.__getitem__)
    best_epoch = epochs[best_idx]
    best_val = val_recon[best_idx]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, train_recon, label="Train Recon Loss", color="#205cc4", linewidth=2.0)
    ax.plot(epochs, val_recon, label="Val Recon Loss", color="#c45520", linewidth=2.0)
    ax.scatter([best_epoch], [best_val], color="#c45520", s=40, zorder=3)
    ax.axvline(best_epoch, color="#c45520", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.annotate(
        f"best val\n(epoch={best_epoch}, loss={best_val:.4f})",
        xy=(best_epoch, best_val),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
        color="#7a2f10",
    )
    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to: {out_path}")
    print(f"Best val_recon_loss: {best_val:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
