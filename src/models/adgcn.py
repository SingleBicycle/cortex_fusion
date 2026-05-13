"""ADGCN-style graph branch model for cortical surface data."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GraphUNet

from src.data.mesh_hierarchy import (
    load_hierarchy_definitions,
    pool_features_by_parent,
    unpool_features_by_parent,
)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked mean pooling over vertices."""
    if mask is None:
        return x.mean(dim=0)
    mask = mask.bool().view(-1)
    if int(mask.sum().item()) == 0:
        return x.mean(dim=0)
    return x[mask].mean(dim=0)


def masked_pca_pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Project vertex features to the first principal direction."""
    orig_dtype = x.dtype
    if mask is not None:
        mask = mask.bool().view(-1)
        if int(mask.sum().item()) > 0:
            x = x[mask]
    if x.shape[0] <= 1:
        return x.mean(dim=0)

    device_type = "cuda" if x.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        x32 = x.float()
        mean = x32.mean(dim=0)
        centered = x32 - mean.unsqueeze(0)
        cov = centered.transpose(0, 1).matmul(centered) / max(int(x.shape[0]) - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_val = eigvals[-1].clamp_min(0.0)
        top_vec = eigvecs[:, -1]

        # Make the principal direction sign deterministic across samples.
        if torch.dot(top_vec, mean) < 0:
            top_vec = -top_vec
        pooled = top_vec * torch.sqrt(top_val)
    return pooled.to(dtype=orig_dtype)


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


class MLPBlock(nn.Module):
    """Per-vertex MLP block used as a non-graph control baseline."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.res_proj = (
            nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        h = self.fc(x)
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


class VertexMLPEncoder(nn.Module):
    """U-shaped per-vertex MLP encoder used to test message passing necessity."""

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
        self.b1 = MLPBlock(dims[0], dims[1], dropout=dropout)
        self.se = GraphSE(dims[1])
        self.b2 = MLPBlock(dims[1], dims[2], dropout=dropout)
        self.b3 = MLPBlock(dims[2], dims[3], dropout=dropout)
        self.b4 = MLPBlock(dims[3], dims[4], dropout=dropout)
        self.b5 = MLPBlock(dims[4], dims[5], dropout=dropout)
        self.b6 = MLPBlock(dims[5], dims[6], dropout=dropout)

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


class HierarchicalMeshEncoder(nn.Module):
    """Fixed-template hierarchical GCN with deterministic mesh pooling/unpooling."""

    def __init__(self, in_channels: int = 16, dropout: float = 0.1, max_levels: int = 2) -> None:
        super().__init__()
        self.max_levels = max(0, int(max_levels))
        self.in_block = GCNBlock(in_channels, in_channels, dropout=dropout)
        self.se = GraphSE(in_channels)
        self.down_blocks = nn.ModuleList(
            [GCNBlock(in_channels, in_channels, dropout=dropout) for _ in range(max(self.max_levels, 1))]
        )
        self.bottleneck = GCNBlock(in_channels, in_channels, dropout=dropout)
        self.up_blocks = nn.ModuleList(
            [GCNBlock(in_channels, in_channels, dropout=dropout) for _ in range(max(self.max_levels, 1))]
        )
        self.out_block = GCNBlock(in_channels, in_channels, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        hierarchy: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        h0 = self.in_block(x, edge_index)
        h0 = self.se(h0, valid_mask=valid_mask)

        levels = list(hierarchy or [])[: self.max_levels]
        if not levels:
            return self.out_block(h0 + x, edge_index)

        skip_feats = [h0]
        skip_edges = [edge_index]
        current = h0
        current_mask = valid_mask

        for idx, bundle in enumerate(levels):
            current, current_mask = pool_features_by_parent(
                current,
                parent_index=bundle["parent_index"],
                num_parents=int(bundle["num_coarse_nodes"]),
                valid_mask=current_mask,
            )
            current = self.down_blocks[idx](current, bundle["coarse_edge_index"])
            skip_feats.append(current)
            skip_edges.append(bundle["coarse_edge_index"])

        current = self.bottleneck(current, skip_edges[-1])

        for idx in range(len(levels) - 1, -1, -1):
            current = unpool_features_by_parent(current, parent_index=levels[idx]["parent_index"])
            current = self.up_blocks[idx](current + skip_feats[idx], skip_edges[idx])

        return self.out_block(current + x, edge_index)


class GraphUNetEncoder(nn.Module):
    """Learned graph pooling/unpooling backbone for MAE reconstruction ablations."""

    def __init__(
        self,
        in_channels: int = 16,
        depth: int = 2,
        pool_ratios: Sequence[float] | float = (0.8, 0.8),
    ) -> None:
        super().__init__()
        depth = int(depth)
        if depth <= 0:
            raise ValueError(f"graph_unet_depth must be positive, got {depth}")
        self.depth = depth
        if isinstance(pool_ratios, (float, int)):
            ratios = [float(pool_ratios)] * depth
        else:
            ratios = [float(x) for x in pool_ratios]
            if len(ratios) == 1:
                ratios = ratios * depth
        if len(ratios) != depth:
            raise ValueError(
                f"graph_unet_pool_ratios must have length 1 or depth={depth}, got {len(ratios)}"
            )
        if any(r <= 0.0 or r > 1.0 for r in ratios):
            raise ValueError(f"graph_unet_pool_ratios must be in (0, 1], got {ratios}")
        self.pool_ratios = ratios
        self.net = GraphUNet(
            in_channels=in_channels,
            hidden_channels=in_channels,
            out_channels=in_channels,
            depth=depth,
            pool_ratios=ratios,
            sum_res=True,
            act="relu",
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del valid_mask
        return self.net(x, edge_index)


class GraphAttentionPool(nn.Module):
    """Learned attention pooling over vertices."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        if valid_mask is not None:
            mask = valid_mask.bool().view(-1)
            if int(mask.sum().item()) == 0:
                return x.mean(dim=0)
            logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=0).unsqueeze(-1)
        return (attn * x).sum(dim=0)


class GraphBranchModel(nn.Module):
    """Graph branch with schema-aware split encoders + ADGCN backbone."""

    def __init__(
        self,
        input_mode: str,
        in_dim: int,
        geo_dim: int,
        morph_dim: int,
        hidden_dim: int = 32,
        dims: Sequence[int] = (32, 64, 128, 256, 128, 64, 32),
        num_classes: int = 0,
        dropout: float = 0.1,
        fuse_mode: str = "sum",
        backbone_type: str = "gcn",
        pool_mode: str = "mean",
        hierarchy_dir: str | None = None,
        edge_cache_dir: str | None = None,
        max_hierarchy_levels: int = 2,
        graph_unet_depth: int = 2,
        graph_unet_pool_ratios: Sequence[float] | float = (0.8, 0.8),
        graph_global_dim: int = 0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if geo_dim < 0 or morph_dim < 0 or (geo_dim + morph_dim) != in_dim:
            raise ValueError(
                f"Expected geo_dim + morph_dim == in_dim, got geo_dim={geo_dim}, "
                f"morph_dim={morph_dim}, in_dim={in_dim}"
            )
        if fuse_mode not in {"sum", "concat"}:
            raise ValueError(f"Unsupported fuse_mode: {fuse_mode}")
        if backbone_type not in {"gcn", "mlp", "mesh_hier", "graph_unet"}:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        if pool_mode not in {"mean", "attn", "pca", "mean_pca"}:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")
        if dims[0] != hidden_dim:
            raise ValueError(f"dims[0]={dims[0]} must equal hidden_dim={hidden_dim}")

        self.input_mode = str(input_mode)
        self.in_dim = int(in_dim)
        self.geo_dim = int(geo_dim)
        self.morph_dim = int(morph_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.fuse_mode = fuse_mode
        self.backbone_type = backbone_type
        self.pool_mode = pool_mode
        self.pool_out_dim = self.hidden_dim * 2 if self.pool_mode == "mean_pca" else self.hidden_dim
        self.max_hierarchy_levels = max(0, int(max_hierarchy_levels))
        self.graph_unet_depth = int(graph_unet_depth)
        self.graph_unet_pool_ratios = (
            [float(graph_unet_pool_ratios)] if isinstance(graph_unet_pool_ratios, (float, int))
            else [float(x) for x in graph_unet_pool_ratios]
        )
        self.graph_global_dim = max(0, int(graph_global_dim))
        self.hierarchy_defs = (
            load_hierarchy_definitions(hierarchy_dir=hierarchy_dir, edge_cache_dir=edge_cache_dir)
            if self.backbone_type == "mesh_hier"
            else {}
        )

        self.geo_encoder = self._build_encoder(self.geo_dim) if self.geo_dim > 0 else None
        self.morph_encoder = self._build_encoder(self.morph_dim) if self.morph_dim > 0 else None
        self.fuse_linear = None
        if self.geo_encoder is not None and self.morph_encoder is not None and self.fuse_mode == "concat":
            self.fuse_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        if self.backbone_type == "gcn":
            self.encoder = ADGCNEncoder(in_channels=self.hidden_dim, dims=dims, dropout=dropout)
        elif self.backbone_type == "mesh_hier":
            self.encoder = HierarchicalMeshEncoder(
                in_channels=self.hidden_dim,
                dropout=dropout,
                max_levels=self.max_hierarchy_levels,
            )
        elif self.backbone_type == "graph_unet":
            self.encoder = GraphUNetEncoder(
                in_channels=self.hidden_dim,
                depth=self.graph_unet_depth,
                pool_ratios=self.graph_unet_pool_ratios,
            )
        else:
            self.encoder = VertexMLPEncoder(in_channels=self.hidden_dim, dims=dims, dropout=dropout)

        self.recon_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.in_dim),
        )

        self.label_head = None
        if self.num_classes > 0:
            self.label_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.num_classes),
            )

        self.global_encoder = None
        graph_head_in_dim = self.pool_out_dim * 2
        if self.graph_global_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.graph_global_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )
            graph_head_in_dim += self.hidden_dim

        self.graph_head = nn.Sequential(
            nn.Linear(graph_head_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.attn_pool = GraphAttentionPool(self.hidden_dim) if self.pool_mode == "attn" else None

    def _build_encoder(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _split_input(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        geo = x[:, : self.geo_dim] if self.geo_dim > 0 else None
        morph = x[:, self.geo_dim :] if self.morph_dim > 0 else None
        return {"geo": geo, "morph": morph}

    def _fuse_input(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._split_input(x)
        encoded = []

        if self.geo_encoder is not None:
            encoded.append(self.geo_encoder(parts["geo"]))
        if self.morph_encoder is not None:
            encoded.append(self.morph_encoder(parts["morph"]))

        if not encoded:
            raise RuntimeError("At least one encoder branch must be enabled.")
        if len(encoded) == 1:
            return encoded[0]
        if self.fuse_mode == "sum":
            return encoded[0] + encoded[1]
        return self.fuse_linear(torch.cat(encoded, dim=-1))

    def forward_hemi(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        res_name: Optional[str] = None,
        hemi_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        h0 = self._fuse_input(x)
        hierarchy = self._resolve_hierarchy(res_name=res_name, hemi_name=hemi_name, device=h0.device)
        if self.backbone_type == "mesh_hier":
            h_v = self.encoder(h0, edge_index=edge_index, valid_mask=valid_mask, hierarchy=hierarchy)
        else:
            h_v = self.encoder(h0, edge_index=edge_index, valid_mask=valid_mask)

        recon_pred = self.recon_head(h_v)
        label_logits = self.label_head(h_v) if self.label_head is not None else None
        z_hemi = self._pool_vertices(h_v, valid_mask=valid_mask)

        out = {
            "H_v": h_v,
            "recon_pred": recon_pred,
            "z_hemi": z_hemi,
        }
        if label_logits is not None:
            out["label_logits"] = label_logits
        return out

    def _pool_vertices(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pool_mode == "mean":
            return masked_mean(x, valid_mask)
        if self.pool_mode == "attn":
            if self.attn_pool is None:
                raise RuntimeError("attn_pool is not initialized")
            return self.attn_pool(x, valid_mask=valid_mask)
        if self.pool_mode == "pca":
            return masked_pca_pool(x, valid_mask)
        if self.pool_mode == "mean_pca":
            return torch.cat([masked_mean(x, valid_mask), masked_pca_pool(x, valid_mask)], dim=-1)
        raise RuntimeError(f"Unsupported pool_mode: {self.pool_mode}")

    def _resolve_hierarchy(
        self,
        res_name: Optional[str],
        hemi_name: Optional[str],
        device: torch.device,
    ) -> list[Dict[str, torch.Tensor]]:
        if self.backbone_type != "mesh_hier":
            return []
        if not res_name or not hemi_name:
            return []

        chain: list[Dict[str, torch.Tensor]] = []
        current_res = str(res_name)
        hemi = str(hemi_name)
        seen = set()
        while len(chain) < self.max_hierarchy_levels:
            key = (hemi, current_res)
            if key not in self.hierarchy_defs or key in seen:
                break
            payload = self.hierarchy_defs[key]
            chain.append(
                {
                    "parent_index": payload["parent_index"].to(device=device),
                    "coarse_edge_index": payload["coarse_edge_index"].to(device=device),
                    "num_coarse_nodes": int(payload["num_coarse_nodes"]),
                }
            )
            seen.add(key)
            current_res = str(payload["coarse_res"])
        return chain

    def forward(
        self,
        lh: Dict[str, torch.Tensor],
        rh: Dict[str, torch.Tensor],
        graph_global: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out_lh = self.forward_hemi(
            x=lh["x"],
            edge_index=lh["edge_index"],
            valid_mask=lh.get("valid_mask", None),
            res_name=lh.get("res_name", None),
            hemi_name="lh",
        )
        out_rh = self.forward_hemi(
            x=rh["x"],
            edge_index=rh["edge_index"],
            valid_mask=rh.get("valid_mask", None),
            res_name=rh.get("res_name", None),
            hemi_name="rh",
        )

        graph_parts = [out_lh["z_hemi"], out_rh["z_hemi"]]
        if self.graph_global_dim > 0:
            if graph_global is None:
                raise RuntimeError("graph_global_dim > 0 but graph_global was not provided")
            if self.global_encoder is None:
                raise RuntimeError("global_encoder is not initialized")
            graph_parts.append(self.global_encoder(graph_global.view(-1).to(dtype=out_lh["z_hemi"].dtype)))

        z_graph = self.graph_head(torch.cat(graph_parts, dim=-1))
        return {
            "lh": out_lh,
            "rh": out_rh,
            "z_graph": z_graph,
        }


__all__ = [
    "masked_mean",
    "masked_pca_pool",
    "GCNBlock",
    "MLPBlock",
    "GraphSE",
    "ADGCNEncoder",
    "VertexMLPEncoder",
    "HierarchicalMeshEncoder",
    "GraphUNetEncoder",
    "GraphAttentionPool",
    "GraphBranchModel",
]
