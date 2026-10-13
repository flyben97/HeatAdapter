"""Molecular heat operators with full-spectrum or degeneracy-safe preprocessing.

The active operator pads minibatches and applies batched spectral projections.
Historical v2 results used a fixed-rank float32 decomposition and must not be
interpreted as evaluations of this repaired implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .ops import scatter_sum


def unnormalized_laplacian(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unnormalized Laplacian: L = D - A."""
    from .ops import dense_laplacian

    return dense_laplacian(n, edge_index, edge_weight, device, dtype)


def random_walk_laplacian(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Random-walk normalized Laplacian: L_rw = I - D^{-1}A = D^{-1}L."""
    L = unnormalized_laplacian(n, edge_index, edge_weight, device, dtype)
    deg = L.diag().clamp_min(1e-6)
    inv_deg = 1.0 / deg
    return inv_deg.unsqueeze(0) * L


def normalized_laplacian(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}."""
    from .ops import dense_laplacian

    L = dense_laplacian(n, edge_index, edge_weight, device, dtype)
    deg = L.diag().clamp_min(1e-6)
    inv_sqrt_deg = deg.pow(-0.5)
    return inv_sqrt_deg.unsqueeze(0) * L * inv_sqrt_deg.unsqueeze(1)


def precompute_spectral(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    normalized: bool = True,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute eigendecomposition of graph Laplacian (called once during data prep)."""
    device = edge_index.device
    dtype = torch.float64
    if normalized:
        lap = normalized_laplacian(
            n, edge_index, device=device, dtype=dtype, edge_weight=edge_weight
        )
    else:
        from .ops import dense_laplacian

        lap = dense_laplacian(n, edge_index, edge_weight, device, dtype)
    eigvals, eigvecs = torch.linalg.eigh(lap)
    # A cutoff must never split a degenerate eigenspace: its basis is arbitrary.
    # Extend through the entire boundary cluster before casting for training.
    if topk is not None and 0 < topk < n:
        while topk < n and torch.isclose(eigvals[topk], eigvals[topk - 1], atol=1e-8, rtol=1e-7):
            topk += 1
        eigvals = eigvals[:topk]
        eigvecs = eigvecs[:, :topk]
    return eigvals.clamp_min(0).float(), eigvecs.float()


def precompute_spectral_unnorm(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute eigendecomposition of UNNORMALIZED Laplacian L = D - A."""
    return precompute_spectral(n, edge_index, edge_weight, normalized=False, topk=topk)


def compute_node_degree(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted node degree (including self-loops)."""
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=torch.float32)
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    deg.scatter_add_(0, edge_index[0], edge_weight)
    return deg.clamp_min(1e-6)


def precompute_spectral_rw(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute for random-walk Laplacian L_rw = D^{-1}L.

    L_rw is similar to L_sym (same eigenvalues).  For the heat kernel:
      H_rw = D^{-1/2} @ V_sym @ diag(exp(-τλ)) @ V_sym^T @ D^{1/2}

    Returns (eigvals, eigvecs_sym, sqrt_deg) where sqrt_deg is used for
    pre/post scaling in GraphHeatBlock_Lrw.
    """
    device = edge_index.device
    dtype = torch.float32
    deg = compute_node_degree(n, edge_index, edge_weight)
    sqrt_deg = deg.sqrt()
    # L_sym eigendecomposition
    lap_sym = normalized_laplacian(n, edge_index, edge_weight, device, dtype)
    eigvals, eigvecs = torch.linalg.eigh(lap_sym)
    eigvals = eigvals.to(dtype)
    eigvecs = eigvecs.to(dtype)
    if topk is not None and topk < n:
        eigvals = eigvals[:topk]
        eigvecs = eigvecs[:, :topk]
    return eigvals, eigvecs, sqrt_deg


# ---------------------------------------------------------------------------
# Fast segment softmax — scatter-based (same as v7)
# ---------------------------------------------------------------------------


def fast_segment_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter-based softmax: max-subtract → exp → scatter_sum → divide."""
    max_vals = scores.new_full((dim_size,), float("-inf"))
    max_vals.scatter_reduce_(0, index, scores, reduce="amax", include_self=False)
    scores_shifted = scores - max_vals[index]
    exp_scores = torch.exp(scores_shifted)
    exp_sum = scatter_sum(exp_scores, index, dim_size).clamp_min(1e-10)
    return exp_scores / exp_sum[index]


# ---------------------------------------------------------------------------
# GraphHeatOperator — padded batched spectral, fused tau
# ---------------------------------------------------------------------------


class GraphHeatOperator(nn.Module):
    """Heat diffusion using padded minibatch projections with variable ranks.

    ``min_group_size`` is accepted for checkpoint-era constructor compatibility;
    the repaired path always processes the complete padded minibatch.
    """

    def __init__(
        self,
        channels: int,
        num_kernels: int = 3,
        init_taus: tuple[float, ...] = (0.25, 1.0, 4.0),
        learnable_taus: bool = True,
        residual: bool = True,
        dropout: float = 0.0,
        min_group_size: int = 5,
    ) -> None:
        super().__init__()
        init = torch.tensor(init_taus[:num_kernels], dtype=torch.float32)
        self.log_taus = nn.Parameter(init.log(), requires_grad=learnable_taus)
        self.kernel_logits = nn.Parameter(torch.zeros(num_kernels))
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.min_group_size = min_group_size

    @property
    def taus(self) -> torch.Tensor:
        return self.log_taus.exp().clamp(1e-4, 50.0)

    def export_parameters(self) -> dict[str, list[float]]:
        """Return learned diffusion parameters in a JSON-serialisable form."""
        with torch.no_grad():
            return {
                "taus": [float(value) for value in self.taus.detach().cpu()],
                "kernel_weights": [
                    float(value) for value in F.softmax(self.kernel_logits, dim=0).detach().cpu()
                ],
            }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        eigvals_list: list[torch.Tensor] | None = None,
        eigvecs_list: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if eigvals_list is None or eigvecs_list is None:
            return self._forward_eager(x, edge_index, batch, edge_weight)

        # Pad a whole minibatch once; no per-molecule GPU .item() synchronization.
        # Variable k is essential when a cutoff expands across a degenerate space.
        sizes = [v.shape[0] for v in eigvecs_list]
        if not sizes:
            return torch.zeros_like(x)
        max_n = max(sizes)
        max_k = max(v.shape[1] for v in eigvecs_list)
        vectors = torch.stack(
            [F.pad(v.to(x), (0, max_k - v.shape[1], 0, max_n - v.shape[0])) for v in eigvecs_list]
        )
        values = torch.stack([F.pad(v.to(x), (0, max_k - len(v))) for v in eigvals_list])
        features = nn.utils.rnn.pad_sequence(x.split(sizes), batch_first=True)
        coeff = torch.bmm(vectors.transpose(1, 2), features)
        mode = getattr(self, "kernel_mode", "heat")
        if mode == "identity":
            output = x
        else:
            if mode == "zero":
                combined = (values < 1e-7).to(x.dtype)
            else:
                decays = torch.exp(-self.taus[:, None, None].to(x) * values[None])
                combined = (F.softmax(self.kernel_logits, dim=0)[:, None, None] * decays).sum(0)
            result = torch.bmm(vectors, combined.unsqueeze(-1) * coeff)
            output = torch.cat([result[i, :n] for i, n in enumerate(sizes)])

        output = self.dropout(output)
        output = self.proj(output)
        if self.residual:
            output = output + x
        output = self.norm(output)
        return output

    def _forward_eager(self, x, edge_index, batch, edge_weight):
        """Fallback: eigh on-the-fly (compatibility)."""
        from .ops import graph_slices

        weights = F.softmax(self.kernel_logits, dim=0)
        taus = self.taus.to(x.dtype)
        output = torch.zeros_like(x)
        for idx in graph_slices(batch):
            n = idx.numel()
            if n == 0:
                continue
            local_id = torch.empty(x.shape[0], device=x.device, dtype=torch.long)
            local_id[idx] = torch.arange(n, device=x.device)
            g0 = int(batch[idx[0]].item())
            mask = (
                batch[edge_index[0]] == g0
                if edge_index.numel()
                else torch.zeros(0, dtype=torch.bool, device=x.device)
            )
            sub_edges = edge_index[:, mask] if edge_index.numel() else edge_index
            sub_edges = local_id[sub_edges] if sub_edges.numel() else sub_edges
            sub_w = edge_weight[mask] if edge_weight is not None and edge_weight.numel() else None
            ev, eV = precompute_spectral(n, sub_edges, sub_w)
            ev, eV = ev.to(x), eV.to(x)
            coeff = eV.T @ x[idx]
            mode = getattr(self, "kernel_mode", "heat")
            if mode == "identity":
                output[idx] = x[idx]
            else:
                if mode == "zero":
                    combined = (ev < 1e-7).to(x.dtype)
                else:
                    decays = torch.exp(-taus[:, None] * ev[None, :])
                    combined = (weights[:, None] * decays).sum(0)
                output[idx] = eV @ (combined.unsqueeze(-1) * coeff)
        output = self.dropout(output)
        output = self.proj(output)
        if self.residual:
            output = output + x
        output = self.norm(output)
        return output


# ---------------------------------------------------------------------------
# LocalAttentiveLayer — same as v7 (scatter-based softmax)
# ---------------------------------------------------------------------------


class LocalAttentiveLayer(nn.Module):
    """AttentiveFP-style local message layer with fast scatter-based softmax."""

    def __init__(self, hidden: int, edge_dim: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.align = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.value = nn.Linear(hidden + edge_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        row, col = edge_index
        msg_in = torch.cat([h[row], h[col], edge_attr], dim=-1)
        score = self.align(msg_in).squeeze(-1)
        attn = fast_segment_softmax(score, row, h.shape[0]).unsqueeze(-1)
        value = self.value(torch.cat([h[col], edge_attr], dim=-1))
        context = scatter_sum(attn * value, row, h.shape[0])
        updated = self.gru(context, h)
        return self.norm(updated + h)


# ---------------------------------------------------------------------------
# GraphHeatBlock
# ---------------------------------------------------------------------------


class GraphHeatBlock(nn.Module):
    """Local MP (fast softmax) + grouped-batched spectral heat diffusion."""

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local = LocalAttentiveLayer(hidden)
        self.heat = GraphHeatOperator(
            hidden, num_kernels=3, init_taus=(0.2, 1.0, 5.0), dropout=dropout
        )
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())

    def forward(self, h: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
        edge_attr = data["edge_weight"].unsqueeze(-1)
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        local = self.local(h, data["edge_index"], edge_attr)
        heat = self.heat(
            local,
            data["edge_index"],
            data["batch"],
            data["edge_weight"],
            eigvals_list=eigvals,
            eigvecs_list=eigvecs,
        )
        gate = self.gate(torch.cat([local, heat], dim=-1))
        return gate * heat + (1.0 - gate) * local


class GraphHeatNet(nn.Module):
    """v10 GroupedBMM: size-grouped batched spectral, fused tau, identical accuracy to v2/v7."""

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        depth: int = 2,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GraphHeatBlock(hidden, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for block in self.blocks:
            h = block(h, data)
        return self.head(global_mean_pool(h, data["batch"]))


# ---------------------------------------------------------------------------
# Retained comparison variant
# ---------------------------------------------------------------------------


class GraphHeatBlock_NoGate(nn.Module):
    """Equal-weight local/heat fusion used as a parameter-free gate ablation."""

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local = LocalAttentiveLayer(hidden)
        self.heat = GraphHeatOperator(
            hidden, num_kernels=3, init_taus=(0.2, 1.0, 5.0), dropout=dropout
        )

    def forward(self, h, data):
        local = self.local(h, data["edge_index"], data["edge_weight"].unsqueeze(-1))
        heat = self.heat(
            local,
            data["edge_index"],
            data["batch"],
            data["edge_weight"],
            eigvals_list=data.get("eigvals_list"),
            eigvecs_list=data.get("eigvecs_list"),
        )
        # ``heat`` already contains an internal residual connection to ``local``.
        # Averaging therefore gives an explicit equal-fusion control instead of
        # silently double-counting the local representation.
        return 0.5 * (local + heat)


def _make_net(block_cls, name):
    class _Net(nn.Module):
        def __init__(self, in_dim=9, hidden=128, depth=3, out_dim=1, dropout=0.1):
            super().__init__()
            self.embed = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([block_cls(hidden, dropout=dropout) for _ in range(depth)])
            self.head = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
            )

        def forward(self, data):
            h = self.embed(data["x"])
            for block in self.blocks:
                h = block(h, data)
            return self.head(global_mean_pool(h, data["batch"]))

    _Net.__name__ = name
    _Net.__qualname__ = name
    return _Net


GraphHeatNet_NoGate = _make_net(GraphHeatBlock_NoGate, "GraphHeatNet_NoGate")
