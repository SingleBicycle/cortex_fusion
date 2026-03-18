"""Loss helpers for masked feature reconstruction."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def get_recon_dim_indices(in_dim: int, geo_dim: int, morph_dim: int, recon_on: str) -> List[int]:
    recon_on = str(recon_on).lower()
    if recon_on == "all":
        return list(range(in_dim))
    if recon_on == "morph_only":
        if morph_dim <= 0:
            raise ValueError("recon_on='morph_only' requires morph_dim > 0")
        start = in_dim - morph_dim
        return list(range(start, in_dim))
    raise ValueError(f"Unsupported recon_on mode: {recon_on}")


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vertex_mask: torch.Tensor,
    dim_weights: torch.Tensor,
    active_dims: List[int],
    loss_type: str = "wmse",
    huber_delta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must match, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.dim() != 2:
        raise ValueError(f"Expected [N, D] tensors, got {tuple(pred.shape)}")

    num_nodes, in_dim = pred.shape
    if target.shape[1] != int(dim_weights.shape[0]):
        raise ValueError(
            f"dim_weights length must equal feature dimension, got {dim_weights.shape[0]} vs {target.shape[1]}"
        )

    vertex_mask = vertex_mask.bool().view(num_nodes)
    if int(vertex_mask.sum().item()) == 0:
        zero = pred.new_zeros(())
        per_dim = pred.new_zeros(in_dim)
        return {"loss": zero, "per_dim_loss": per_dim}

    masked_pred = pred[vertex_mask]
    masked_target = target[vertex_mask]

    loss_type = str(loss_type).lower()
    if loss_type == "wmse":
        per_entry = (masked_pred - masked_target) ** 2
    elif loss_type == "huber":
        per_entry = F.huber_loss(masked_pred, masked_target, reduction="none", delta=huber_delta)
    else:
        raise ValueError(f"Unsupported reconstruction loss: {loss_type}")

    per_dim_loss = per_entry.mean(dim=0)

    active_weight_mask = torch.zeros(in_dim, dtype=pred.dtype, device=pred.device)
    active_weight_mask[active_dims] = 1.0
    weights = dim_weights.to(device=pred.device, dtype=pred.dtype) * active_weight_mask
    weight_sum = weights.sum().clamp_min(1e-8)
    loss = (per_dim_loss * weights).sum() / weight_sum

    return {
        "loss": loss,
        "per_dim_loss": per_dim_loss * active_weight_mask,
    }


__all__ = [
    "get_recon_dim_indices",
    "masked_reconstruction_loss",
]
