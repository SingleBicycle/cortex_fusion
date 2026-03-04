"""ADGCN-style graph branch model for cortical surface data."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked mean pooling over vertices."""
    if mask is None:
        return x.mean(dim=0)
    mask = mask.bool().view(-1)
    if int(mask.sum().item()) == 0:
        return x.mean(dim=0)
    return x[mask].mean(dim=0)


class GCNBlock(nn.Module):
    """GCN + BN + ReLU + Dropout + residual."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.res_proj = (
            nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        return h + self.res_proj(x)


class GraphSE(nn.Module):
    """Squeeze-Excitation channel attention for graph vertex features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = masked_mean(x, valid_mask)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled), inplace=True))).unsqueeze(0)
        return x * gate


class ADGCNEncoder(nn.Module):
    """Deep U-shaped channel-space GCN encoder with skip links."""

    def __init__(
        self,
        in_channels: int = 16,
        dims: Sequence[int] = (16, 32, 64, 128, 64, 32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dims) != 7:
            raise ValueError(f"dims must have length 7, got {len(dims)}")
        if dims[0] != in_channels:
            raise ValueError(f"dims[0]={dims[0]} must equal in_channels={in_channels}")

        self.dims = list(dims)
        self.b1 = GCNBlock(dims[0], dims[1], dropout=dropout)
        self.se = GraphSE(dims[1])
        self.b2 = GCNBlock(dims[1], dims[2], dropout=dropout)
        self.b3 = GCNBlock(dims[2], dims[3], dropout=dropout)
        self.b4 = GCNBlock(dims[3], dims[4], dropout=dropout)
        self.b5 = GCNBlock(dims[4], dims[5], dropout=dropout)
        self.b6 = GCNBlock(dims[5], dims[6], dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h0 = x
        h1 = self.b1(h0, edge_index)
        h1 = self.se(h1, valid_mask=valid_mask)

        h2 = self.b2(h1, edge_index)
        h3 = self.b3(h2, edge_index)

        u2 = self.b4(h3, edge_index) + h2
        u1 = self.b5(u2, edge_index) + h1
        u0 = self.b6(u1, edge_index) + h0
        return u0


class GraphBranchModel(nn.Module):
    """Graph branch with split input encoders + ADGCN backbone + task heads."""

    def __init__(
        self,
        hidden_dim: int = 16,
        dims: Sequence[int] = (16, 32, 64, 128, 64, 32, 16),
        num_classes: int = 0,
        dropout: float = 0.1,
        fuse_mode: str = "sum",
    ) -> None:
        super().__init__()
        if fuse_mode not in {"sum", "concat"}:
            raise ValueError(f"Unsupported fuse_mode: {fuse_mode}")

        self.hidden_dim = hidden_dim
        self.num_classes = int(num_classes)
        self.fuse_mode = fuse_mode

        self.pos_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.morph_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse_linear = nn.Linear(hidden_dim * 2, hidden_dim) if fuse_mode == "concat" else None

        self.encoder = ADGCNEncoder(in_channels=hidden_dim, dims=dims, dropout=dropout)

        self.morph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self.label_head = None
        if self.num_classes > 0:
            self.label_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_classes),
            )

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def _fuse_input(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[:, :6]
        morph = x[:, 6:8]

        h_pos = self.pos_encoder(pos)
        h_morph = self.morph_encoder(morph)

        if self.fuse_mode == "sum":
            return h_pos + h_morph
        return self.fuse_linear(torch.cat([h_pos, h_morph], dim=-1))

    def forward_hemi(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h0 = self._fuse_input(x)
        h_v = self.encoder(h0, edge_index=edge_index, valid_mask=valid_mask)

        morph_pred = self.morph_head(h_v)
        label_logits = self.label_head(h_v) if self.label_head is not None else None
        z_hemi = masked_mean(h_v, valid_mask)

        out = {
            "H_v": h_v,
            "morph_pred": morph_pred,
            "z_hemi": z_hemi,
        }
        if label_logits is not None:
            out["label_logits"] = label_logits
        return out

    def forward(self, lh: Dict[str, torch.Tensor], rh: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_lh = self.forward_hemi(
            x=lh["x"],
            edge_index=lh["edge_index"],
            valid_mask=lh.get("valid_mask", None),
        )
        out_rh = self.forward_hemi(
            x=rh["x"],
            edge_index=rh["edge_index"],
            valid_mask=rh.get("valid_mask", None),
        )

        z_graph = self.graph_head(torch.cat([out_lh["z_hemi"], out_rh["z_hemi"]], dim=-1))
        return {
            "lh": out_lh,
            "rh": out_rh,
            "z_graph": z_graph,
        }


__all__ = [
    "masked_mean",
    "GCNBlock",
    "GraphSE",
    "ADGCNEncoder",
    "GraphBranchModel",
]
