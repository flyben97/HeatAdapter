from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Native scatter-sum over dim 0, replacing torch_scatter for this prototype."""
    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(out_shape)
    if src.numel() == 0:
        return out
    view_shape = (index.numel(),) + (1,) * (src.dim() - 1)
    out.scatter_add_(0, index.view(view_shape).expand_as(src), src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(src[..., :1]), index, dim_size).clamp_min(1.0)
    return out / count


def graph_slices(batch: torch.Tensor) -> list[torch.Tensor]:
    """Return node indexes for each graph in a batch."""
    if batch.numel() == 0:
        return []
    num_graphs = int(batch.max().item()) + 1
    return [torch.nonzero(batch == g, as_tuple=False).flatten() for g in range(num_graphs)]


def radius_graph(pos: torch.Tensor, batch: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Small native radius graph. Directed edges, no self loops."""
    rows = []
    cols = []
    cutoff2 = cutoff * cutoff
    for idx in graph_slices(batch):
        if idx.numel() <= 1:
            continue
        p = pos[idx]
        dist2 = torch.cdist(p, p).pow(2)
        mask = (dist2 <= cutoff2) & (~torch.eye(idx.numel(), dtype=torch.bool, device=pos.device))
        src, dst = torch.nonzero(mask, as_tuple=True)
        rows.append(idx[src])
        cols.append(idx[dst])
    if not rows:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def dense_laplacian(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Combinatorial Laplacian L = D - A for one graph."""
    adj = torch.zeros(n, n, device=device, dtype=dtype)
    if edge_index.numel() > 0:
        row, col = edge_index
        weight = (
            edge_weight.to(device=device, dtype=dtype)
            if edge_weight is not None
            else torch.ones(row.numel(), device=device, dtype=dtype)
        )
        adj.index_put_((row, col), weight, accumulate=True)
    adj = 0.5 * (adj + adj.T)
    degree = adj.sum(dim=-1)
    return torch.diag(degree) - adj


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Numerically stable softmax over variable-size segments on dim 0."""
    out = torch.empty_like(scores)
    for i in range(dim_size):
        mask = index == i
        if mask.any():
            out[mask] = torch.softmax(scores[mask], dim=0)
    return out
