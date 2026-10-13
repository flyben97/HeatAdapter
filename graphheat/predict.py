"""Predict from checkpoints saved by the disjoint-fold HeatAdapter runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from analysis.revision_benchmark import configure_model, predict
from graphheat.mol_data import canonical_smiles, mol_to_graph


def predict_csv(checkpoint_path, csv_path, output_path, *, smiles_column="smiles", device="cpu"):
    """Preserve input rows and columns, appending predictions on the source target scale."""
    frame = pd.read_csv(csv_path)
    if smiles_column not in frame:
        raise ValueError(f"Input CSV has no {smiles_column!r} column")
    if frame.empty:
        raise ValueError("Input CSV has no molecules")
    if "prediction" in frame:
        raise ValueError("Input already contains a prediction column; choose a clean input CSV")
    canonical = frame[smiles_column].map(canonical_smiles)
    if canonical.isna().any():
        rows = (np.flatnonzero(canonical.isna()) + 2).tolist()
        raise ValueError(f"Invalid or missing SMILES at CSV line(s): {rows}")
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    variant = checkpoint["variant"]
    model = configure_model(
        checkpoint["backbone"], variant, checkpoint["hidden"], checkpoint["depth"], device
    )
    model.load_state_dict(checkpoint["state_dict"])
    graphs = [
        mol_to_graph(
            Chem.MolFromSmiles(smi),
            spectral_topk=20 if variant == "topk20" else 0,
            unweighted_heat=variant == "unweighted",
        )
        for smi in canonical
    ]
    scaler = checkpoint["scaler"]
    frame["prediction"] = predict(model, graphs, device) * scaler["scale"] + scaler["mean"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    if Path(args.csv).resolve() == Path(args.output).resolve():
        parser.error("output must differ from the input CSV")
    torch.set_num_threads(args.threads)
    predict_csv(
        args.checkpoint,
        args.csv,
        args.output,
        smiles_column=args.smiles_column,
        device=args.device,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
