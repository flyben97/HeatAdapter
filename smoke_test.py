#!/usr/bin/env python
"""Small runtime checks for the PyG-corrected GraphHeat benchmark."""

from __future__ import annotations

from pathlib import Path
import torch
from rdkit import Chem

from graphheat.mol_bench import MODELS, build_model, model_parameter_count
from graphheat.mol_data import (
    _ATOM_FEAT_DIM,
    _scaffold,
    collate_mols,
    load_dataset,
    mol_to_graph,
    random_split,
    scaffold_split,
    validate_scaffold_integrity,
    validate_split_integrity,
)


def main() -> None:
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    targets = [0.1, -0.2, 0.3, -0.1]
    graphs = [mol_to_graph(Chem.MolFromSmiles(s), spectral_topk=8) for s in smiles]
    data = collate_mols(graphs, targets)
    data["y_graph"] = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    data = {
        k: (
            v.to(device=device)
            if isinstance(v, torch.Tensor)
            else (
                [t.to(device=device) for t in v]
                if isinstance(v, list) and v and isinstance(v[0], torch.Tensor)
                else v
            )
        )
        for k, v in data.items()
    }

    required = [
        "gcn",
        "gine",
        "afp",
        "gps",
        "graphheat",
        "graphheat_equalfusion",
        "afp_heat",
        "gine_heat",
        "gcn_heat",
        "gcn_mlp_adapter",
        "gine_mlp_adapter",
    ]
    missing = sorted(set(required) - set(MODELS))
    if missing:
        raise RuntimeError(f"Missing model keys: {missing}")

    for name in required:
        model = build_model(
            name,
            in_dim=_ATOM_FEAT_DIM,
            hidden=32,
            depth=2,
            device=device,
            dtype=dtype,
        )
        model.eval()
        with torch.no_grad():
            pred = model(data)
        assert pred.shape == (len(smiles), 1), (name, tuple(pred.shape))
        assert torch.isfinite(pred).all(), name
        print(f"{name:18s} ok shape={tuple(pred.shape)}")

    # Regression guard for the previous silent all-empty-scaffold failure.
    assert _scaffold("c1ccccc1") == "c1ccccc1"
    esol = load_dataset(
        "toy", csv_path=Path(__file__).resolve().parent / "examples/toy_molecules.csv"
    )
    for splitter in (scaffold_split, random_split):
        train_df, val_df, test_df = splitter(esol, seed=42)
        assert validate_split_integrity(train_df, val_df, test_df) == {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }
    assert validate_scaffold_integrity(*scaffold_split(esol, seed=42)) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    heat = build_model("gine_heat", _ATOM_FEAT_DIM, 32, 2, device, dtype)
    control = build_model("gine_mlp_adapter", _ATOM_FEAT_DIM, 32, 2, device, dtype)
    # The control is deliberately within a tiny parameter tolerance of the
    # heat adapter, making a capacity-only explanation testable.
    assert abs(model_parameter_count(heat) - model_parameter_count(control)) <= 64

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
