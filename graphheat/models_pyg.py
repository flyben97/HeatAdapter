"""Local PyTorch Geometric backbones, heat adapters, and capacity controls."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    APPNP,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    GPSConv,
    MixHopConv,
    NNConv,
    PNAConv,
    SAGEConv,
    AttentiveFP,
    global_mean_pool,
)
from torch_geometric.utils import degree as pyg_degree

from .operators import GraphHeatOperator


def _make_mlp(hidden: int) -> nn.Sequential:
    """Standard 2-layer MLP used inside GINConv / GINEConv."""
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
    )


def _make_head(hidden: int, out_dim: int = 1) -> nn.Sequential:
    """Readout head: SiLU → Linear → SiLU → Linear."""
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


# ═══════════════════════════════════════════════════════════════════
#  GCN — Kipf & Welling 2017
# ═══════════════════════════════════════════════════════════════════


class GCNNet(nn.Module):
    """GCN using PyG ``GCNConv`` (degree-normalized convolution)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden, add_self_loops=True) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GIN — Xu et al. 2019
# ═══════════════════════════════════════════════════════════════════


class GINNet(nn.Module):
    """GIN using PyG ``GINConv`` (sum aggregation + MLP, learnable eps)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GINConv(_make_mlp(hidden), eps=0.0, train_eps=True) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GINE — Hu et al. 2020  (edge-aware GIN)
# ═══════════════════════════════════════════════════════════════════


class GINENet(nn.Module):
    """GINE using PyG ``GINEConv`` (edge-conditioned sum aggregation)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GINEConv(_make_mlp(hidden), eps=0.0, train_eps=True, edge_dim=1) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        edge_attr = data["edge_weight"].unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"], edge_attr=edge_attr)) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GAT — Veličković et al. 2018
# ═══════════════════════════════════════════════════════════════════


class GATNet(nn.Module):
    """Classic GAT using PyG ``GATConv`` (additive attention)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GATConv(hidden, hidden, heads=1, concat=True) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GATv2 — Brody et al. 2022  (dynamic attention)
# ═══════════════════════════════════════════════════════════════════


class GATv2Net(nn.Module):
    """GATv2 using PyG ``GATv2Conv`` (dynamic graph attention)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GATv2Conv(hidden, hidden, heads=1, concat=True) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GraphSAGE — Hamilton et al. 2017
# ═══════════════════════════════════════════════════════════════════


class GraphSAGENet(nn.Module):
    """GraphSAGE using PyG ``SAGEConv`` (mean aggregator)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden, aggr="mean") for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  MPNN — edge-conditioned message passing via NNConv
# ═══════════════════════════════════════════════════════════════════


class MPNNNet(nn.Module):
    """Edge-conditioned MPNN using PyG ``NNConv``.

    Notes
    -----
    PyG has no generic "MPNN" layer.  ``NNConv`` is the closest official
    edge-conditioned convolution: it uses a neural network to map edge
    features to a weight matrix that transforms neighbour messages.
    """

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        # edge_nn: maps (1,) bond weight → (hidden × hidden) weight matrix
        self.convs = nn.ModuleList(
            [
                NNConv(hidden, hidden, nn.Linear(1, hidden * hidden), aggr="mean")
                for _ in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        edge_attr = data["edge_weight"].unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"], edge_attr)) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  PNA — Corso et al. 2020  (Principal Neighbourhood Aggregation)
# ═══════════════════════════════════════════════════════════════════


def compute_pna_deg(
    train_graphs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Compute degree histogram for PNA from training-graph list.

    Returns a 1-D float tensor ``deg`` where ``deg[d]`` is the number of
    nodes with degree *d* across the training set.  Required by
    ``PNAConv`` for its degree-scaler normalisation.
    """
    all_degs: list[torch.Tensor] = []
    for g in train_graphs:
        n = int(g["n_atoms"].item()) if isinstance(g["n_atoms"], torch.Tensor) else g["n_atoms"]
        ei = g["edge_index"]
        if ei.numel() > 0:
            d = pyg_degree(ei[0], num_nodes=n, dtype=torch.long)
        else:
            d = torch.zeros(n, dtype=torch.long)
        all_degs.append(d)
    deg = torch.cat(all_degs).to(device=device)
    max_deg = int(deg.max().item())
    hist = torch.zeros(max_deg + 1, dtype=torch.float, device=device)
    for d_val in range(max_deg + 1):
        hist[d_val] = (deg == d_val).sum().float()
    return hist


class PNANet(nn.Module):
    """PNA using PyG ``PNAConv`` (multi-aggregator + degree-scaler)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        deg: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if deg is None:
            deg = torch.ones(1, dtype=torch.float)
        # deg is consumed at init to build internal scalers — register as buffer
        self.register_buffer("deg", deg)
        aggregators = ["mean", "max", "min", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [PNAConv(hidden, hidden, aggregators, scalers, deg) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.silu(conv(h, data["edge_index"])) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  GPS — Rampášek et al. 2022  (General, Powerful, Scalable)
# ═══════════════════════════════════════════════════════════════════


class GPSNet(nn.Module):
    """GPS using PyG ``GPSConv`` (local GINE + multi-head global attention).

    Notes
    -----
    ``GPSConv`` already contains internal batch-norm + FFN, so we do
    **not** add an extra LayerNorm wrapper — the conv output is passed
    through a residual connection with a single SiLU activation.
    """

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(depth):
            local_conv = GINEConv(
                _make_mlp(hidden),
                eps=0.0,
                train_eps=True,
                edge_dim=1,
            )
            gps = GPSConv(
                hidden,
                local_conv,
                heads=4,
                attn_type="multihead",
                dropout=0.0,
            )
            self.convs.append(gps)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        edge_attr = data["edge_weight"].unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data["edge_index"], batch=data["batch"], edge_attr=edge_attr)
            h = norm(F.silu(h_new) + h)
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  Graphormer — Ying et al. 2021  (replaced by GPSConv)
# ═══════════════════════════════════════════════════════════════════


class GraphormerNet(GPSNet):
    """Graphormer baseline — replaced by ``GPSConv`` (PyG's closest graph transformer).

    PyG 2.7.0 has no official ``GraphormerConv``.  ``GPSConv`` with
    multi-head attention provides the closest sparse/dense graph
    transformer, albeit without the distance-bias spatial encoding of
    the original Graphormer or its centrality encoding.
    """

    pass


# ═══════════════════════════════════════════════════════════════════
#  Heat-adapter candidates for the next benchmark
# ═══════════════════════════════════════════════════════════════════


class GINEHeatNet(nn.Module):
    """PyG GINE backbone with a GraphHeat adapter after each local block.

    This tests whether the heat operator adds value on top of a standard
    PyG message-passing backbone, rather than only inside the original
    GraphHeat architecture.
    """

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.local = nn.ModuleList(
            [GINEConv(_make_mlp(hidden), eps=0.0, train_eps=True, edge_dim=1) for _ in range(depth)]
        )
        self.heat = nn.ModuleList(
            [
                GraphHeatOperator(
                    hidden,
                    num_kernels=3,
                    init_taus=(0.2, 1.0, 5.0),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid()) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        edge_attr = data["edge_weight"].unsqueeze(-1)
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        for conv, heat, gate, norm in zip(self.local, self.heat, self.gates, self.norms):
            local = norm(F.silu(conv(h, data["edge_index"], edge_attr=edge_attr)) + h)
            h_heat = heat(
                local,
                data["edge_index"],
                data["batch"],
                data["edge_weight"],
                eigvals_list=eigvals,
                eigvecs_list=eigvecs,
            )
            alpha = gate(torch.cat([local, h_heat], dim=-1))
            h = alpha * h_heat + (1.0 - alpha) * local
        return self.head(global_mean_pool(h, data["batch"]))


class GCNHeatNet(nn.Module):
    """PyG GCN backbone with a GraphHeat adapter after each local block."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.local = nn.ModuleList(
            [GCNConv(hidden, hidden, add_self_loops=True) for _ in range(depth)]
        )
        self.heat = nn.ModuleList(
            [
                GraphHeatOperator(
                    hidden,
                    num_kernels=3,
                    init_taus=(0.2, 1.0, 5.0),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid()) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        for conv, heat, gate, norm in zip(self.local, self.heat, self.gates, self.norms):
            local = norm(F.silu(conv(h, data["edge_index"])) + h)
            h_heat = heat(
                local,
                data["edge_index"],
                data["batch"],
                data["edge_weight"],
                eigvals_list=eigvals,
                eigvecs_list=eigvecs,
            )
            alpha = gate(torch.cat([local, h_heat], dim=-1))
            h = alpha * h_heat + (1.0 - alpha) * local
        return self.head(global_mean_pool(h, data["batch"]))


class GINHeatNet(nn.Module):
    """PyG GIN backbone with a GraphHeat adapter after each local block."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.local = nn.ModuleList(
            [GINConv(_make_mlp(hidden), eps=0.0, train_eps=True) for _ in range(depth)]
        )
        self.heat = nn.ModuleList(
            [
                GraphHeatOperator(
                    hidden,
                    num_kernels=3,
                    init_taus=(0.2, 1.0, 5.0),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid()) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        for conv, heat, gate, norm in zip(self.local, self.heat, self.gates, self.norms):
            local = norm(F.silu(conv(h, data["edge_index"])) + h)
            h_heat = heat(
                local,
                data["edge_index"],
                data["batch"],
                data["edge_weight"],
                eigvals_list=eigvals,
                eigvecs_list=eigvecs,
            )
            alpha = gate(torch.cat([local, h_heat], dim=-1))
            h = alpha * h_heat + (1.0 - alpha) * local
        return self.head(global_mean_pool(h, data["batch"]))


class MPNNHeatNet(nn.Module):
    """PyG MPNN (NNConv) backbone with a GraphHeat adapter after each local block."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.local = nn.ModuleList(
            [
                NNConv(hidden, hidden, nn.Linear(1, hidden * hidden), aggr="mean")
                for _ in range(depth)
            ]
        )
        self.heat = nn.ModuleList(
            [
                GraphHeatOperator(
                    hidden,
                    num_kernels=3,
                    init_taus=(0.2, 1.0, 5.0),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid()) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        edge_attr = data["edge_weight"].unsqueeze(-1)
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        for conv, heat, gate, norm in zip(self.local, self.heat, self.gates, self.norms):
            local = norm(F.silu(conv(h, data["edge_index"], edge_attr)) + h)
            h_heat = heat(
                local,
                data["edge_index"],
                data["batch"],
                data["edge_weight"],
                eigvals_list=eigvals,
                eigvecs_list=eigvecs,
            )
            alpha = gate(torch.cat([local, h_heat], dim=-1))
            h = alpha * h_heat + (1.0 - alpha) * local
        return self.head(global_mean_pool(h, data["batch"]))


class GATv2HeatNet(nn.Module):
    """PyG GATv2 backbone with a GraphHeat adapter after each local block."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.local = nn.ModuleList(
            [GATv2Conv(hidden, hidden, heads=1, concat=True) for _ in range(depth)]
        )
        self.heat = nn.ModuleList(
            [
                GraphHeatOperator(
                    hidden,
                    num_kernels=3,
                    init_taus=(0.2, 1.0, 5.0),
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid()) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        eigvals = data.get("eigvals_list", None)
        eigvecs = data.get("eigvecs_list", None)
        for conv, heat, gate, norm in zip(self.local, self.heat, self.gates, self.norms):
            local = norm(F.elu(conv(h, data["edge_index"])) + h)
            h_heat = heat(
                local,
                data["edge_index"],
                data["batch"],
                data["edge_weight"],
                eigvals_list=eigvals,
                eigvecs_list=eigvecs,
            )
            alpha = gate(torch.cat([local, h_heat], dim=-1))
            h = alpha * h_heat + (1.0 - alpha) * local
        return self.head(global_mean_pool(h, data["batch"]))


class ParameterMatchedMLPAdapter(nn.Module):
    """A capacity control for a GraphHeat adapter.

    For hidden width ``h``, a heat block plus its gate contributes
    ``3h^2 + 4h + 6`` trainable parameters.  This three-layer MLP plus
    LayerNorm contributes ``3h^2 + 5h`` parameters, differing by only ``h-6``
    parameters per layer while containing no graph diffusion operation.
    """

    def __init__(self, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.net(h))


class _LocalMLPAdapterNet(nn.Module):
    """Shared implementation of parameter-matched local-backbone controls."""

    def __init__(
        self,
        *,
        backbone: str,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone not in {"gcn", "gin", "gine", "mpnn", "gatv2"}:
            raise ValueError(f"Unsupported local backbone: {backbone}")
        self.backbone = backbone
        self.embed = nn.Linear(in_dim, hidden)
        if backbone == "gcn":
            self.local = nn.ModuleList(
                [GCNConv(hidden, hidden, add_self_loops=True) for _ in range(depth)]
            )
        elif backbone == "gin":
            self.local = nn.ModuleList(
                [GINConv(_make_mlp(hidden), eps=0.0, train_eps=True) for _ in range(depth)]
            )
        elif backbone == "gine":
            self.local = nn.ModuleList(
                [
                    GINEConv(_make_mlp(hidden), eps=0.0, train_eps=True, edge_dim=1)
                    for _ in range(depth)
                ]
            )
        elif backbone == "mpnn":
            self.local = nn.ModuleList(
                [
                    NNConv(hidden, hidden, nn.Linear(1, hidden * hidden), aggr="mean")
                    for _ in range(depth)
                ]
            )
        else:  # gatv2
            self.local = nn.ModuleList(
                [GATv2Conv(hidden, hidden, heads=1, concat=True) for _ in range(depth)]
            )
        self.adapters = nn.ModuleList(
            [ParameterMatchedMLPAdapter(hidden, dropout=dropout) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def _local_update(
        self, conv: nn.Module, h: torch.Tensor, data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.backbone == "gine":
            return F.silu(conv(h, data["edge_index"], edge_attr=data["edge_weight"].unsqueeze(-1)))
        if self.backbone == "mpnn":
            return F.silu(conv(h, data["edge_index"], data["edge_weight"].unsqueeze(-1)))
        if self.backbone == "gatv2":
            return F.elu(conv(h, data["edge_index"]))
        return F.silu(conv(h, data["edge_index"]))

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, adapter, norm in zip(self.local, self.adapters, self.norms):
            local = norm(self._local_update(conv, h, data) + h)
            h = adapter(local)
        return self.head(global_mean_pool(h, data["batch"]))


class GCNMLPAdapterNet(_LocalMLPAdapterNet):
    """Parameter-matched non-diffusion control for :class:`GCNHeatNet`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(backbone="gcn", **kwargs)


class GINMLPAdapterNet(_LocalMLPAdapterNet):
    """Parameter-matched non-diffusion control for :class:`GINHeatNet`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(backbone="gin", **kwargs)


class GINEMLPAdapterNet(_LocalMLPAdapterNet):
    """Parameter-matched non-diffusion control for :class:`GINEHeatNet`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(backbone="gine", **kwargs)


class MPNNMLPAdapterNet(_LocalMLPAdapterNet):
    """Parameter-matched non-diffusion control for :class:`MPNNHeatNet`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(backbone="mpnn", **kwargs)


class GATv2MLPAdapterNet(_LocalMLPAdapterNet):
    """Parameter-matched non-diffusion control for :class:`GATv2HeatNet`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(backbone="gatv2", **kwargs)


class AttentiveFPHeatNet(nn.Module):
    """PyG AttentiveFP graph encoder fused with a heat-diffusion branch.

    PyG's built-in ``AttentiveFP`` exposes graph-level embeddings when
    ``out_channels=hidden``. The heat branch operates on atom embeddings and
    is mean-pooled to the graph level before gated fusion.
    """

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.afp = AttentiveFP(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=hidden,
            edge_dim=1,
            num_layers=depth,
            num_timesteps=2,
            dropout=dropout,
        )
        self.atom_embed = nn.Linear(in_dim, hidden)
        self.heat = GraphHeatOperator(
            hidden,
            num_kernels=3,
            init_taus=(0.2, 1.0, 5.0),
            dropout=dropout,
        )
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        edge_attr = data["edge_weight"].unsqueeze(-1)
        afp_graph = self.afp(
            data["x"],
            data["edge_index"],
            edge_attr,
            data["batch"],
        )
        h = self.atom_embed(data["x"])
        h_heat = self.heat(
            h,
            data["edge_index"],
            data["batch"],
            data["edge_weight"],
            eigvals_list=data.get("eigvals_list", None),
            eigvecs_list=data.get("eigvecs_list", None),
        )
        heat_graph = global_mean_pool(h_heat, data["batch"])
        gate = self.gate(torch.cat([afp_graph, heat_graph], dim=-1))
        fused = gate * heat_graph + (1.0 - gate) * afp_graph
        return self.head(fused)


# ═══════════════════════════════════════════════════════════════════
#  AttentiveFP — Xiong et al. 2020  (full PyG model)
# ═══════════════════════════════════════════════════════════════════


class AttentiveFPNet(nn.Module):
    """AttentiveFP using PyG's built-in ``AttentiveFP`` model.

    Important
    ---------
    ``AttentiveFP`` returns **graph-level** predictions ``(batch_size, out_dim)``,
    *not* node-level.  The training loop in ``mol_bench.py`` detects this
    and skips ``global_mean_pool``.
    """

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=out_dim,
            edge_dim=1,
            num_layers=depth,
            num_timesteps=2,
            dropout=0.0,
        )

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            data["x"],
            data["edge_index"],
            data["edge_weight"].unsqueeze(-1),
            data["batch"],
        )


# ═══════════════════════════════════════════════════════════════════
#  APPNP — Klicpera et al. 2019
# ═══════════════════════════════════════════════════════════════════


class APPNPNet(nn.Module):
    """APPNP using PyG ``APPNP`` (personalised PageRank propagation)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
        K: int = 10,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.prop = APPNP(K=K, alpha=alpha)
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(data["x"])
        h = self.prop(h, data["edge_index"])
        return self.head(global_mean_pool(h, data["batch"]))


# ═══════════════════════════════════════════════════════════════════
#  MixHop — Abu-El-Haija et al. 2019
# ═══════════════════════════════════════════════════════════════════


class MixHopNet(nn.Module):
    """MixHop using PyG ``MixHopConv`` (0/1/2-hop feature mixing)."""

    def __init__(
        self,
        in_dim: int = 46,
        hidden: int = 128,
        depth: int = 3,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [MixHopConv(hidden, hidden, powers=[0, 1, 2]) for _ in range(depth)]
        )
        # MixHop output is 3×hidden (one per power), compress back to hidden
        self.mix = nn.ModuleList([nn.Linear(3 * hidden, hidden) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.head = _make_head(hidden, out_dim)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.embed(data["x"])
        for conv, mix, norm in zip(self.convs, self.mix, self.norms):
            h_multi = conv(h, data["edge_index"])  # (N, 3×hidden)
            h = norm(F.silu(mix(h_multi)) + h)
        return self.head(global_mean_pool(h, data["batch"]))
