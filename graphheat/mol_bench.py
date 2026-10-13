#!/usr/bin/env python
"""PyG-corrected molecular property benchmark with scaffold split.

Runs PyG official GNN baselines, the GraphHeat method, and focused heat-adapter
candidates on molecular property datasets using RDKit graph conversion.

Usage (local dev):
  python -m graphheat.mol_bench --datasets esol --models afp,graphheat,afp_heat --epochs 20 --seeds 42 --device cpu

Usage (cluster):
  DATASETS=esol,freesolv MODELS=afp,gps,graphheat,afp_heat,gine_heat \
    SEEDS=42,123,456 EPOCHS=300 python -m graphheat.mol_bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch_geometric

# GraphHeat core variants.
from .operators import (
    GraphHeatNet,
    GraphHeatNet_NoGate,
    GraphHeatOperator,
)

# PyG-official baseline models and heat-adapter candidates.
from .models_pyg import (
    APPNPNet,
    AttentiveFPHeatNet,
    AttentiveFPNet,
    GATNet,
    GATv2MLPAdapterNet,
    GATv2HeatNet,
    GATv2Net,
    GCNMLPAdapterNet,
    GCNHeatNet,
    GCNNet,
    GINEMLPAdapterNet,
    GINEHeatNet,
    GINENet,
    GINMLPAdapterNet,
    GINHeatNet,
    GINNet,
    GPSNet,
    GraphSAGENet,
    MixHopNet,
    MPNNMLPAdapterNet,
    MPNNHeatNet,
    MPNNNet,
    PNANet,
    compute_pna_deg,
)
from .mol_data import (
    _ATOM_FEAT_DIM,
    CONDITIONAL_DATASETS,
    DATASETS,
    PRIMARY_DATASETS,
    collate_mols,
    load_dataset,
    mol_to_graph,
    random_split,
    scaffold_split,
    set_spectral_topk,
    split_diagnostics,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "graphheat": GraphHeatNet,
    "graphheat_equalfusion": GraphHeatNet_NoGate,
    "afp_heat": AttentiveFPHeatNet,
    "gine_heat": GINEHeatNet,
    "gcn_heat": GCNHeatNet,
    "gin_heat": GINHeatNet,
    "mpnn_heat": MPNNHeatNet,
    "gatv2_heat": GATv2HeatNet,
    "gcn_mlp_adapter": GCNMLPAdapterNet,
    "gin_mlp_adapter": GINMLPAdapterNet,
    "gine_mlp_adapter": GINEMLPAdapterNet,
    "mpnn_mlp_adapter": MPNNMLPAdapterNet,
    "gatv2_mlp_adapter": GATv2MLPAdapterNet,
    "gps": GPSNet,
    "pna": PNANet,
    "gine": GINENet,
    "gatv2": GATv2Net,
    "gat": GATNet,
    "gin": GINNet,
    "mpnn": MPNNNet,
    "graphsage": GraphSAGENet,
    "gcn": GCNNet,
    "appnp": APPNPNet,
    "mixhop": MixHopNet,
    "afp": AttentiveFPNet,
}

NON_NEURAL_MODELS = {"ecfp4_rf"}

MODEL_NOTES = {
    "graphheat": "GraphHeat: local attentive MP + size-grouped spectral heat diffusion",
    "graphheat_equalfusion": "GraphHeat equal-weight local/heat fusion control without a learned gate",
    "afp_heat": "PyG AttentiveFP graph encoder fused with a heat-diffusion graph branch",
    "gine_heat": "PyG GINE backbone with a heat adapter after each local block",
    "gps": "PyG GPSConv: GINEConv local + multi-head global attention",
    "pna": "PyG PNAConv: mean/max/min/std aggregators + degree scalers",
    "gine": "PyG GINEConv: edge-conditioned sum aggregation",
    "gatv2": "PyG GATv2Conv: dynamic graph attention",
    "gat": "PyG GATConv: classic additive graph attention",
    "gin": "PyG GINConv: sum aggregation + MLP, learnable eps",
    "mpnn": "PyG NNConv: edge-conditioned message passing",
    "graphsage": "PyG SAGEConv: mean GraphSAGE aggregation",
    "gcn": "PyG GCNConv: degree-normalized convolution",
    "appnp": "PyG APPNP: MLP + personalized PageRank propagation",
    "mixhop": "PyG MixHopConv: 0/1/2-hop mixed propagation",
    "afp": "PyG AttentiveFP: graph attention + iterative readout (graph-level output)",
    "gcn_heat": "PyG GCN backbone with a heat adapter after each local block",
    "gin_heat": "PyG GIN backbone with a heat adapter after each local block",
    "mpnn_heat": "PyG NNConv (MPNN) backbone with a heat adapter after each local block",
    "gatv2_heat": "PyG GATv2 backbone with a heat adapter after each local block",
    "gcn_mlp_adapter": "parameter-matched MLP adapter control for GCNHeat",
    "gin_mlp_adapter": "parameter-matched MLP adapter control for GINHeat",
    "gine_mlp_adapter": "parameter-matched MLP adapter control for GINEHeat",
    "mpnn_mlp_adapter": "parameter-matched MLP adapter control for MPNNHeat",
    "gatv2_mlp_adapter": "parameter-matched MLP adapter control for GATv2Heat",
    "ecfp4_rf": "ECFP4 fingerprint random-forest baseline with validation-selected tree settings",
}

FINAL_PYG_MODELS = [
    "gcn",
    "gin",
    "gine",
    "mpnn",
    "gatv2",
    "afp",
    "pna",
    "gps",
    "graphheat",
    "graphheat_equalfusion",
]

HEAT_ADAPTER_MODELS = ["gcn_heat", "gin_heat", "gine_heat", "mpnn_heat", "gatv2_heat", "afp_heat"]
MLP_CONTROL_MODELS = [
    "gcn_mlp_adapter",
    "gin_mlp_adapter",
    "gine_mlp_adapter",
    "mpnn_mlp_adapter",
    "gatv2_mlp_adapter",
]
PRIMARY_BENCHMARK_MODELS = (
    ["ecfp4_rf"] + FINAL_PYG_MODELS + HEAT_ADAPTER_MODELS + MLP_CONTROL_MODELS
)


# ---------------------------------------------------------------------------
# Graph-level pooling
# ---------------------------------------------------------------------------


def global_mean_pool(node_pred: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean-pool legacy node-level predictions to graph-level predictions."""
    num_graphs = int(batch.max().item()) + 1
    out = node_pred.new_zeros(num_graphs, node_pred.shape[-1])
    ones = node_pred.new_ones(node_pred.shape[0], 1)
    count = node_pred.new_zeros(num_graphs, 1)
    count.scatter_add_(0, batch.unsqueeze(-1).expand(-1, 1), ones)
    out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, node_pred.shape[-1]), node_pred)
    return out / count.clamp_min(1)


def graph_predictions(pred: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return graph-level predictions and reject ambiguous model output shapes.

    All active models now pool hidden node states before the prediction head.
    The node-level fallback remains only for reading legacy model implementations
    during transition, and prevents silently comparing incompatible outputs.
    """
    num_graphs = int(data["batch"].max().item()) + 1
    if pred.size(0) == num_graphs:
        return pred
    if pred.size(0) == data["batch"].size(0):
        return global_mean_pool(pred, data["batch"])
    raise RuntimeError(
        f"Model returned {tuple(pred.shape)} predictions for {num_graphs} graphs "
        f"and {data['batch'].size(0)} nodes."
    )


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch before an initialization replicate."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def model_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class GateStatistics:
    """Collect mean and standard deviation of named GraphHeat fusion gates."""

    def __init__(self, model: nn.Module) -> None:
        self._stats: dict[str, dict[str, float]] = {}
        self._hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Sigmoid) and "gate" in name:
                self._stats[name] = {"sum": 0.0, "sum_sq": 0.0, "n": 0.0}
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            values = output.detach()
            item = self._stats[name]
            item["sum"] += float(values.sum())
            item["sum_sq"] += float(values.square().sum())
            item["n"] += float(values.numel())

        return hook

    def close(self) -> dict[str, dict[str, float]]:
        for hook in self._hooks:
            hook.remove()
        output: dict[str, dict[str, float]] = {}
        for name, item in self._stats.items():
            n = item["n"]
            if n <= 0:
                continue
            mean = item["sum"] / n
            variance = max(item["sum_sq"] / n - mean * mean, 0.0)
            output[name] = {"mean": mean, "std": variance**0.5, "n": int(n)}
        return output


def extract_heat_parameters(model: nn.Module) -> list[dict[str, object]]:
    """Extract learned heat-kernel parameters from a restored best checkpoint."""
    parameters: list[dict[str, object]] = []
    for name, module in model.named_modules():
        if isinstance(module, GraphHeatOperator):
            parameters.append({"module": name, **module.export_parameters()})
    return parameters


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_model(
    name: str,
    in_dim: int,
    hidden: int,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs,
):
    """Build a model instance. Extra kwargs forwarded to model constructor (e.g. ``deg`` for PNA)."""
    model_cls = MODELS[name]
    model = model_cls(in_dim=in_dim, hidden=hidden, depth=depth, **kwargs)
    model = model.to(device=device, dtype=dtype)
    return model


def move_data_to_device(
    data: dict[str, object],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Move a collated batch to device while preserving index tensor dtypes."""

    def move(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and dtype is not None:
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            return [move(item) for item in value]
        return value

    return {key: move(value) for key, value in data.items()}


def train_epoch(model, train_graphs, train_targets, optimizer, device, batch_size):
    """Mini-batch training epoch. Returns (avg_loss, avg_mae)."""
    model.train()
    n = len(train_graphs)
    # Shuffle indices each epoch
    perm = torch.randperm(n).tolist()
    total_loss = 0.0
    total_mae = 0.0
    total_graphs = 0

    for start in range(0, n, batch_size):
        indices = perm[start : start + batch_size]
        subset_graphs = [train_graphs[i] for i in indices]
        subset_targets = [train_targets[i] for i in indices]

        data = collate_mols(subset_graphs, subset_targets)
        y_graph = torch.tensor(subset_targets, dtype=torch.float32).unsqueeze(-1)

        model_dtype = next(model.parameters()).dtype
        data = move_data_to_device(data, device, model_dtype)
        y_graph = y_graph.to(device=device, dtype=model_dtype)

        # Skip batches with too few nodes — GPSConv BatchNorm needs ≥2 nodes
        if data["x"].size(0) < 2:
            continue

        optimizer.zero_grad()
        pred = model(data)
        pred_graph = graph_predictions(pred, data)
        loss = F.mse_loss(pred_graph, y_graph)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = (pred_graph - y_graph).abs().mean()
            batch_n = len(subset_graphs)
            total_loss += float(loss) * batch_n
            total_mae += float(mae) * batch_n
            total_graphs += batch_n

    return total_loss / total_graphs, total_mae / total_graphs


@torch.no_grad()
def evaluate(model, data, device, collect_gate_statistics: bool = False):
    """Evaluate a model and optionally collect fusion-gate summary statistics."""
    del device  # retained for backwards-compatible call sites
    model.eval()
    collector = GateStatistics(model) if collect_gate_statistics else None
    try:
        pred_graph = graph_predictions(model(data), data)
    finally:
        gate_statistics = collector.close() if collector is not None else None
    err = pred_graph - data["y_graph"]
    result = {
        "mae": float(err.abs().mean()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "loss": float(F.mse_loss(pred_graph, data["y_graph"])),
    }
    if gate_statistics is not None:
        result["gate_statistics"] = gate_statistics
    return result


@torch.no_grad()
def evaluate_chunked(model, graphs, targets, device, chunk_size=512):
    """Evaluate in chunks to avoid OOM on large datasets.

    Collates *chunk_size* graphs at a time, concatenates predictions,
    then computes metrics on the full concatenated vector.
    """
    model.eval()
    all_preds = []
    n = len(graphs)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        subset_graphs = graphs[start:end]
        subset_targets = targets[start:end]
        data = collate_mols(subset_graphs, subset_targets)
        data = move_data_to_device(data, device, next(model.parameters()).dtype)
        pred = model(data)
        pred_graph = graph_predictions(pred, data)
        all_preds.append(pred_graph.cpu())

    preds = torch.cat(all_preds, dim=0)
    y_true = torch.tensor(targets, dtype=preds.dtype).unsqueeze(-1)
    err = preds - y_true
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "loss": float(F.mse_loss(preds, y_true)),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_graphs(df, graphs_cache=None):
    """Convert a dataframe of SMILES+target to per-molecule graph/target lists.

    Args:
        df: DataFrame with columns SMILES and target
        graphs_cache: dict to cache mol_to_graph results (SMILES→graph)
    Returns:
        (all_graphs, all_targets) — lists of per-molecule graph dicts and float targets.
        Returns (None, None) if all SMILES failed to parse.
    """
    all_graphs = []
    all_targets = []
    for _, row in df.iterrows():
        smi = row["SMILES"]
        if graphs_cache is not None and smi in graphs_cache:
            g = graphs_cache[smi]
        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            g = mol_to_graph(mol)
            if graphs_cache is not None:
                graphs_cache[smi] = g
        all_graphs.append(g)
        all_targets.append(row["target"])

    if not all_graphs:
        return None, None
    return all_graphs, all_targets


def prepare_split(df, graphs_cache=None):
    """Convert a dataframe of SMILES+target to a full batched graph dict.

    Returns (data, (all_graphs, all_targets)) where *data* is for eval
    (full-pass) and the raw lists are for mini-batch training.
    """
    all_graphs, all_targets = prepare_graphs(df, graphs_cache)
    if all_graphs is None:
        return None, (None, None)

    data = collate_mols(all_graphs, all_targets)
    y_graph = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(-1)
    data["y_graph"] = y_graph
    return data, (all_graphs, all_targets)


def fit_scaler(train_df, val_df, test_df):
    """Fit StandardScaler on training targets; apply to all splits."""
    scaler = StandardScaler()
    scaler.fit(train_df[["target"]].values)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["target"] = scaler.transform(train_df[["target"]].values).ravel()
    val_df["target"] = scaler.transform(val_df[["target"]].values).ravel()
    test_df["target"] = scaler.transform(test_df[["target"]].values).ravel()
    return train_df, val_df, test_df, scaler


def ecfp4_features(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    """Build deterministic ECFP4 bit vectors for the classical baseline."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    features = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES reached ECFP featurization after curation: {smi!r}")
        features[index] = generator.GetFingerprintAsNumPy(mol)
    return features


def fit_ecfp4_random_forest(
    train_x: np.ndarray,
    train_y: list[float],
    val_x: np.ndarray,
    val_y: list[float],
    test_x: np.ndarray,
    test_y: list[float],
    *,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[dict[str, float], dict[str, object]]:
    """Fit a validation-selected ECFP4 random forest without test-set tuning."""
    candidates = [
        {"max_features": 1.0, "min_samples_leaf": 1},
        {"max_features": 0.5, "min_samples_leaf": 1},
        {"max_features": 1.0, "min_samples_leaf": 2},
    ]
    best_model = None
    best_config = None
    best_val_mae = float("inf")
    y_train = np.asarray(train_y, dtype=float)
    y_val = np.asarray(val_y, dtype=float)
    y_test = np.asarray(test_y, dtype=float)
    for candidate_index, config in enumerate(candidates):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed + candidate_index * 10_000,
            n_jobs=n_jobs,
            max_features=config["max_features"],
            min_samples_leaf=config["min_samples_leaf"],
            criterion="squared_error",
        )
        model.fit(train_x, y_train)
        val_mae = float(np.abs(model.predict(val_x) - y_val).mean())
        if val_mae < best_val_mae:
            best_model = model
            best_config = config
            best_val_mae = val_mae
    assert best_model is not None and best_config is not None
    test_error = best_model.predict(test_x) - y_test
    return {
        "val_mae": best_val_mae,
        "test_mae": float(np.abs(test_error).mean()),
        "test_rmse": float(np.sqrt(np.mean(np.square(test_error)))),
    }, {"selected_hyperparameters": best_config, "n_estimators": n_estimators}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _checkpoint_path(args, dataset_key: str, model_name: str, init_seed: int) -> Path:
    safe_parts = [
        args.campaign_id,
        args.run_id,
        dataset_key,
        model_name,
        args.split,
        f"split{args.split_seed}",
        f"init{init_seed}",
    ]
    filename = "_".join(str(part).replace("/", "-") for part in safe_parts) + ".pt"
    return Path(args.output_dir) / "checkpoints" / filename


def _save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def run_dataset(
    dataset_key: str,
    model_names: list[str],
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict], dict[str, object]]:
    """Run one curation-aware, leakage-controlled dataset benchmark."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_key}")
    print(f"{'=' * 60}")

    df, curation_report = load_dataset(dataset_key, return_report=True)
    if args.split == "random":
        train_df, val_df, test_df = random_split(df, seed=args.split_seed)
        split_protocol = "molecule-disjoint random split"
    else:
        train_df, val_df, test_df = scaffold_split(df, seed=args.split_seed)
        split_protocol = "Murcko scaffold split with molecule grouping for acyclic compounds"
    split_report = split_diagnostics(train_df, val_df, test_df)
    print(
        "  Split: "
        f"train={split_report['n_train']} val={split_report['n_val']} "
        f"test={split_report['n_test']} | molecule overlap={split_report['molecule_overlap']}"
    )

    # Targets are standardised only after curation and split construction.
    train_df, val_df, test_df, scaler = fit_scaler(train_df, val_df, test_df)

    graphs_cache: dict[str, dict] = {}
    t0 = time.time()
    train_data, (train_graphs, train_targets) = prepare_split(train_df, graphs_cache)
    val_data, (val_graphs, val_targets) = prepare_split(val_df, graphs_cache)
    test_data, (test_graphs, test_targets) = prepare_split(test_df, graphs_cache)
    if train_data is None or val_data is None or test_data is None:
        raise RuntimeError(f"Empty graph split after curation for dataset {dataset_key!r}.")
    graph_conversion_seconds = time.time() - t0
    print(f"  Graph conversion: {graph_conversion_seconds:.1f}s (train={len(train_graphs)} graphs)")

    val_data_dev = move_data_to_device(val_data, device, dtype)
    test_data_dev = move_data_to_device(test_data, device, dtype)

    eff_batch_size = min(args.batch_size, len(train_graphs))
    if eff_batch_size != args.batch_size:
        print(f"  batch_size: {args.batch_size} → {eff_batch_size} (dataset size)")

    pna_deg = None
    if "pna" in model_names:
        print(
            f"  Computing PNA degree histogram from {len(train_graphs)} training graphs...",
            end="",
            flush=True,
        )
        pna_deg = compute_pna_deg(train_graphs, device)
        print(f" max_deg={pna_deg.size(0) - 1}")

    fingerprints = None
    if "ecfp4_rf" in model_names:
        print("  Building ECFP4 fingerprints for the classical baseline...", end="", flush=True)
        fingerprints = {
            "train": ecfp4_features(train_df["SMILES"].tolist()),
            "val": ecfp4_features(val_df["SMILES"].tolist()),
            "test": ecfp4_features(test_df["SMILES"].tolist()),
        }
        print(" done")

    dataset_metadata: dict[str, object] = {
        "dataset": dataset_key,
        "curation": curation_report,
        "split_protocol": split_protocol,
        "split_seed": args.split_seed,
        "split_diagnostics": split_report,
        "target_scaler": {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])},
        "graph_conversion_seconds": graph_conversion_seconds,
        "actual_graph_counts": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
    }

    results: list[dict] = []
    for model_name in model_names:
        for seed in args.seeds_list:
            print(f"  {model_name:22s} init_seed={seed}", end="", flush=True)
            t0 = time.time()
            set_global_seed(seed, deterministic=args.deterministic)

            if model_name == "ecfp4_rf":
                assert fingerprints is not None
                rf_metrics, rf_metadata = fit_ecfp4_random_forest(
                    fingerprints["train"],
                    train_targets,
                    fingerprints["val"],
                    val_targets,
                    fingerprints["test"],
                    test_targets,
                    seed=seed,
                    n_estimators=args.rf_n_estimators,
                    n_jobs=args.rf_n_jobs,
                )
                elapsed = time.time() - t0
                row = {
                    "campaign_id": args.campaign_id,
                    "run_id": args.run_id,
                    "dataset": dataset_key,
                    "model": model_name,
                    "init_seed": seed,
                    "seed": seed,
                    "split_seed": args.split_seed,
                    "split_protocol": args.split,
                    "batch_size": None,
                    "hidden": None,
                    "depth": None,
                    "spectral_topk": None,
                    "best_epoch": None,
                    "epochs_trained": None,
                    "val_mae": rf_metrics["val_mae"] * scaler.scale_[0],
                    "test_mae": rf_metrics["test_mae"] * scaler.scale_[0],
                    "test_rmse": rf_metrics["test_rmse"] * scaler.scale_[0],
                    "seconds": elapsed,
                    "model_parameters": None,
                    "n_train": len(train_graphs),
                    "n_val": len(val_graphs),
                    "n_test": len(test_graphs),
                    "heat_parameters": [],
                    "gate_statistics": {},
                    "classical_baseline": rf_metadata,
                }
                results.append(row)
                print(
                    f"  test_mae={float(row['test_mae']):.4f} "
                    f"test_rmse={float(row['test_rmse']):.4f} time={elapsed:.1f}s"
                )
                continue

            extra_kwargs = {"deg": pna_deg} if model_name == "pna" else {}
            model = build_model(
                model_name,
                in_dim=_ATOM_FEAT_DIM,
                hidden=args.hidden,
                depth=args.depth,
                device=device,
                dtype=dtype,
                **extra_kwargs,
            )
            parameter_count = model_parameter_count(model)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

            best_val_mae = float("inf")
            best_state = None
            best_epoch = 0
            patience_counter = 0
            epochs_trained = 0
            for epoch in range(1, args.epochs + 1):
                epochs_trained = epoch
                train_loss, train_mae = train_epoch(
                    model, train_graphs, train_targets, optimizer, device, eff_batch_size
                )
                val_metrics = evaluate(model, val_data_dev, device)
                if val_metrics["mae"] < best_val_mae:
                    best_val_mae = val_metrics["mae"]
                    best_state = {
                        name: value.cpu().clone() for name, value in model.state_dict().items()
                    }
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= args.patience:
                    break
                if epoch % args.log_every == 0:
                    print(
                        f"\n    epoch={epoch} train_loss={train_loss:.4f} "
                        f"train_mae={train_mae:.4f} val_mae={val_metrics['mae']:.4f}",
                        end="",
                    )

            if best_state is None:
                raise RuntimeError(
                    f"No model state was selected for {dataset_key}/{model_name}/seed={seed}."
                )
            model.load_state_dict(best_state)
            test_metrics = evaluate(model, test_data_dev, device, collect_gate_statistics=True)
            elapsed = time.time() - t0

            test_mae_unscaled = test_metrics["mae"] * scaler.scale_[0]
            test_rmse_unscaled = test_metrics["rmse"] * scaler.scale_[0]
            val_mae_unscaled = best_val_mae * scaler.scale_[0]
            row: dict[str, object] = {
                "campaign_id": args.campaign_id,
                "run_id": args.run_id,
                "dataset": dataset_key,
                "model": model_name,
                "init_seed": seed,
                "seed": seed,  # retained for compatibility with legacy analysis code
                "split_seed": args.split_seed,
                "split_protocol": args.split,
                "batch_size": eff_batch_size,
                "hidden": args.hidden,
                "depth": args.depth,
                "spectral_topk": args.spectral_topk,
                "best_epoch": best_epoch,
                "epochs_trained": epochs_trained,
                "val_mae": val_mae_unscaled,
                "test_mae": test_mae_unscaled,
                "test_rmse": test_rmse_unscaled,
                "seconds": elapsed,
                "model_parameters": parameter_count,
                "n_train": len(train_graphs),
                "n_val": len(val_graphs),
                "n_test": len(test_graphs),
                "heat_parameters": extract_heat_parameters(model),
                "gate_statistics": test_metrics.get("gate_statistics", {}),
            }
            if args.save_checkpoints:
                checkpoint = _checkpoint_path(args, dataset_key, model_name, seed)
                _save_checkpoint(
                    model,
                    checkpoint,
                    {
                        "campaign_id": args.campaign_id,
                        "run_id": args.run_id,
                        "dataset": dataset_key,
                        "model": model_name,
                        "init_seed": seed,
                        "split_seed": args.split_seed,
                        "best_epoch": best_epoch,
                        "test_mae": test_mae_unscaled,
                    },
                )
                row["checkpoint_path"] = str(checkpoint)
            results.append(row)
            print(
                f"  test_mae={test_mae_unscaled:.4f} test_rmse={test_rmse_unscaled:.4f} "
                f"best_ep={best_epoch} trained={epochs_trained} time={elapsed:.1f}s"
            )
    return results, dataset_metadata


def write_markdown(rows: list[dict], path: str, metadata: dict[str, object]) -> None:
    """Write a human-readable companion file for one uniquely identified run."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Per-dataset summary
    datasets = sorted(set(r["dataset"] for r in rows))
    model_order = []
    for r in rows:
        if r["model"] not in model_order:
            model_order.append(r["model"])

    lines = [
        "# GraphHeat Molecular Benchmark",
        "",
        f"Campaign: `{metadata['campaign_id']}`",
        f"Run: `{metadata['run_id']}`",
        f"Generated (UTC): {metadata['generated_at_utc']}",
        f"Schema version: {metadata['schema_version']}",
        "",
    ]

    for ds in datasets:
        lines.append(f"## {ds}\n")
        lines.append(
            "| model | MAE mean | MAE SD across init seeds | RMSE mean | best epoch | seconds mean | parameters |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for m in model_order:
            group = [r for r in rows if r["dataset"] == ds and r["model"] == m]
            if not group:
                continue
            mae_vals = torch.tensor([float(r["test_mae"]) for r in group])
            rmse_vals = torch.tensor([float(r["test_rmse"]) for r in group])
            sec_vals = torch.tensor([float(r["seconds"]) for r in group])
            epoch_values = [
                float(r["best_epoch"]) for r in group if r.get("best_epoch") is not None
            ]
            ep_vals = torch.tensor(epoch_values) if epoch_values else None
            mae_std = float(mae_vals.std(unbiased=False)) if mae_vals.numel() > 1 else 0.0
            epoch_display = f"{float(ep_vals.mean()):.1f}" if ep_vals is not None else "NA"
            parameter_count = group[0].get("model_parameters")
            parameter_display = str(int(parameter_count)) if parameter_count is not None else "NA"
            lines.append(
                f"| {m} | {float(mae_vals.mean()):.4f} | {mae_std:.4f} | "
                f"{float(rmse_vals.mean()):.4f} | {epoch_display} | "
                f"{float(sec_vals.mean()):.1f} | {parameter_display} |"
            )
        lines.append("")

    # Per-seed details
    lines.append("## Per-seed details\n")
    headers = [
        "dataset",
        "model",
        "init_seed",
        "split_seed",
        "test_mae",
        "test_rmse",
        "best_epoch",
        "epochs_trained",
        "seconds",
        "model_parameters",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        for i, h in enumerate(headers):
            v = row.get(h, "")
            if isinstance(v, float):
                vals[i] = f"{v:.6f}"
        lines.append("| " + " | ".join(vals) + " |")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results written to {path}")


def write_json(rows: list[dict], path: str, metadata: dict[str, object]) -> None:
    """Write versioned, self-describing result data without silent overwrites."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing benchmark result: {target}")
    with target.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)
    print(f"JSON written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Leakage-controlled GraphHeat molecular benchmark")
    parser.add_argument(
        "--datasets",
        default=",".join(PRIMARY_DATASETS),
        help="Comma-separated dataset keys. Defaults to the five primary structure-only endpoints.",
    )
    parser.add_argument(
        "--models", default=",".join(PRIMARY_BENCHMARK_MODELS), help="Comma-separated model keys"
    )
    parser.add_argument(
        "--campaign-id",
        required=True,
        help="Stable identifier shared by all jobs in one planned benchmark campaign",
    )
    parser.add_argument(
        "--run-id", default=None, help="Unique run identifier. Generated automatically if omitted."
    )
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs per run")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated random seeds")
    parser.add_argument(
        "--split_seed",
        "--split-seed",
        dest="split_seed",
        type=int,
        default=42,
        help="Seed for the molecule/scaffold group assignment",
    )
    parser.add_argument(
        "--split",
        default="scaffold",
        choices=["scaffold", "random"],
        help="scaffold: Murcko split with acyclic molecule grouping; random: molecule-disjoint random split",
    )
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=3, help="Number of layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--log_every",
        "--log-every",
        dest="log_every",
        type=int,
        default=20,
        help="Log every N epochs",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        default="results/benchmark_v2",
        help="Output directory for results",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--float64", action="store_true", help="Use float64 precision")
    parser.add_argument(
        "--spectral-topk",
        type=int,
        default=0,
        help="Keep the lowest-k Laplacian eigenpairs for heat diffusion. Use 0 for full spectrum.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=128,
        help="Mini-batch size for training (default 128). "
        "Set to a large value for full-batch fallback.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic cuDNN settings where supported.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save the selected state for each dataset/model/split/init replicate.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Trees for each validation candidate in the ECFP4 random-forest baseline.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
        help="CPU workers for ECFP4 random forests; -1 uses all available workers.",
    )
    parser.add_argument(
        "--allow-condition-agnostic-dielectric",
        action="store_true",
        help="Allow the unsupported structure-only dielectric benchmark. Not for the JCIM primary campaign.",
    )
    args = parser.parse_args()

    # Parse lists
    dataset_keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    args.seeds_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )

    # Validate
    unknown_ds = sorted(set(dataset_keys) - set(DATASETS))
    if unknown_ds:
        print(f"ERROR: Unknown datasets: {unknown_ds}. Available: {sorted(DATASETS)}")
        sys.exit(1)
    unknown_m = sorted(set(model_names) - (set(MODELS) | NON_NEURAL_MODELS))
    if unknown_m:
        print(
            f"ERROR: Unknown models: {unknown_m}. Available: {sorted(set(MODELS) | NON_NEURAL_MODELS)}"
        )
        sys.exit(1)
    unsupported_conditional = sorted(set(dataset_keys) & set(CONDITIONAL_DATASETS))
    if unsupported_conditional and not args.allow_condition_agnostic_dielectric:
        parser.error(
            "Condition-dependent datasets are excluded from the structure-only benchmark: "
            f"{unsupported_conditional}. Add --allow-condition-agnostic-dielectric only for a legacy diagnostic run."
        )
    if args.spectral_topk is not None and args.spectral_topk <= 0:
        args.spectral_topk = None

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Campaign: {args.campaign_id}, run: {args.run_id}")
    print(f"Datasets: {dataset_keys}")
    print(f"Models: {model_names}")
    print(f"Seeds: {args.seeds_list}")
    print(f"Epochs: {args.epochs}, patience: {args.patience}")
    print(f"Hidden: {args.hidden}, depth: {args.depth}, lr: {args.lr}")

    set_spectral_topk(args.spectral_topk)
    print(
        f"Low-frequency eigenpairs: {args.spectral_topk if args.spectral_topk is not None else 'all'}"
    )

    all_rows: list[dict] = []
    dataset_records: dict[str, object] = {}
    for ds_key in dataset_keys:
        rows, dataset_record = run_dataset(ds_key, model_names, args, device, dtype)
        all_rows.extend(rows)
        dataset_records[ds_key] = dataset_record

    if not all_rows:
        print("ERROR: No results produced.")
        sys.exit(1)

    metadata: dict[str, object] = {
        "schema_version": 2,
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "rdkit": rdBase.rdkitVersion,
            "device": str(device),
        },
        "config": {
            "datasets": dataset_keys,
            "models": model_names,
            "init_seeds": args.seeds_list,
            "split": args.split,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden": args.hidden,
            "depth": args.depth,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "spectral_topk": args.spectral_topk,
            "deterministic": args.deterministic,
            "rf_n_estimators": args.rf_n_estimators,
            "rf_n_jobs": args.rf_n_jobs,
        },
        "datasets": dataset_records,
    }
    safe_campaign = args.campaign_id.replace("/", "-").replace(" ", "_")
    safe_run = args.run_id.replace("/", "-").replace(" ", "_")
    base_name = f"benchmark_{safe_campaign}_{safe_run}"
    md_path = os.path.join(args.output_dir, base_name + ".md")
    json_path = os.path.join(args.output_dir, base_name + ".json")
    if Path(md_path).exists() or Path(json_path).exists():
        raise FileExistsError(f"Result paths already exist for campaign/run: {base_name}")
    write_markdown(all_rows, md_path, metadata)
    write_json(all_rows, json_path, metadata)

    # Quick summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    datasets_seen = sorted(set(r["dataset"] for r in all_rows))
    models_seen = sorted(set(r["model"] for r in all_rows))
    for ds in datasets_seen:
        print(f"\n{ds}:")
        for m in models_seen:
            group = [r for r in all_rows if r["dataset"] == ds and r["model"] == m]
            if group:
                maes = [r["test_mae"] for r in group]
                print(f"  {m:16s} MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}")


if __name__ == "__main__":
    main()
