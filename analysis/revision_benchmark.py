"""Train HeatAdapter and controls with validation-only tuning and disjoint test folds."""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from importlib.resources import files

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
import torch_geometric

from graphheat.mol_data import (
    load_dataset,
    mol_to_graph,
    collate_mols,
    scaffold_kfold,
    split_diagnostics,
    _scaffold_group_key,
    _ATOM_FEAT_DIM,
    PRIMARY_DATASETS,
)
from graphheat.mol_bench import (
    build_model,
    train_epoch,
    move_data_to_device,
    set_global_seed,
    extract_heat_parameters,
)
from graphheat.operators import GraphHeatOperator

VARIANTS = ("base", "mlp", "heat", "identity", "zero", "fixed", "single", "unweighted", "topk20")
BACKBONES = ("gcn", "gin", "gine", "mpnn", "gatv2")


def configure_model(backbone, variant, hidden, depth, device):
    name = (
        backbone
        if variant == "base"
        else backbone + ("_mlp_adapter" if variant == "mlp" else "_heat")
    )
    model = build_model(name, _ATOM_FEAT_DIM, hidden, depth, device, torch.float32)
    for module in model.modules():
        if not isinstance(module, GraphHeatOperator):
            continue
        if variant in ("identity", "zero"):
            module.kernel_mode = variant
            module.log_taus.requires_grad_(False)
            module.kernel_logits.requires_grad_(False)
        elif variant == "fixed":
            module.log_taus.requires_grad_(False)
        elif variant == "single":
            module.log_taus = torch.nn.Parameter(torch.zeros(1, device=device))
            module.kernel_logits = torch.nn.Parameter(
                torch.zeros(1, device=device), requires_grad=False
            )
    return model


@torch.no_grad()
def predict(model, graphs, device, chunk=128):
    model.eval()
    predictions = []
    for start in range(0, len(graphs), chunk):
        subset = graphs[start : start + chunk]
        data = move_data_to_device(collate_mols(subset, [0.0] * len(subset)), device, torch.float32)
        predictions.append(model(data).flatten().cpu().numpy())
    return np.concatenate(predictions)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # GPU scatter reductions may remain nondeterministic; do not claim bitwise reproduction.
    device = torch.device(args.device)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text())
    data = {
        name: load_dataset(
            name,
            base_dir=args.data_root,
            return_report=True,
            csv_path=args.csv,
            smiles_column=args.smiles_column,
            target_column=args.target_column,
        )
        for name in args.datasets.split(",")
    }
    config = dict(vars(args))
    package_dir = Path(str(files("graphheat")))
    sources = [Path(__file__), *sorted(package_dir.glob("*.py"))]
    config["source_sha256"] = {f"{p.parent.name}/{p.name}": digest(p) for p in sources}
    config["protocol_sha256"] = digest(args.protocol)
    config["dataset_sha256"] = {name: report["source_sha256"] for name, (_, report) in data.items()}
    config["software"] = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        pyg=torch_geometric.__version__,
        rdkit=rdBase.rdkitVersion,
    )
    config["hardware"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )
    config["bitwise_deterministic"] = False
    if (root / "manifest.json").exists():
        old = json.loads((root / "manifest.json").read_text())
        if old != config:
            raise RuntimeError(
                "Configuration/source differs from existing manifest; use a new output directory."
            )
    else:
        atomic_json(root / "manifest.json", config)
    for dataset in args.datasets.split(","):
        df, curation = data[dataset]
        destination = root / dataset
        destination.mkdir(exist_ok=True)
        atomic_json(destination / "curation.json", curation)
        frames = [scaffold_kfold(df, fold, seed=protocol["split_seed"]) for fold in range(5)]
        assignments = []
        for fold, splits in enumerate(frames):
            for role, frame in zip(("train", "validation", "test"), splits):
                for smi in frame.SMILES:
                    assignments.append(
                        dict(fold=fold, role=role, smiles=smi, scaffold=_scaffold_group_key(smi))
                    )
        pd.DataFrame(assignments).to_csv(destination / "split_assignments.csv", index=False)
        tests = [set(parts[2].SMILES) for parts in frames]
        assert sum(map(len, tests)) == len(set.union(*tests)) == len(df)
        atomic_json(
            destination / "split_audit.json",
            {
                "unique_oof_molecules": len(df),
                "test_pairwise_overlap": 0,
                "folds": [split_diagnostics(*parts) for parts in frames],
            },
        )
        graph_cache = {}
        for variant in args.variants.split(","):
            cache_key = (
                "unweighted"
                if variant == "unweighted"
                else "topk20" if variant == "topk20" else "full"
            )
            if cache_key not in graph_cache:
                started = time.perf_counter()
                graph_cache[cache_key] = {
                    s: mol_to_graph(
                        Chem.MolFromSmiles(s),
                        spectral_topk=20 if variant == "topk20" else 0,
                        unweighted_heat=variant == "unweighted",
                    )
                    for s in df.SMILES
                }
                print(
                    f"{dataset} {cache_key} graph preparation {time.perf_counter()-started:.2f}s",
                    flush=True,
                )
            graphs_by_smiles = graph_cache[cache_key]
            for fold in map(int, args.folds.split(",")):
                train, val, test = frames[fold]
                mean, scale = float(train.target.mean()), float(train.target.std(ddof=0))
                scale = scale if scale > 0 else 1.0
                tr_g, va_g, te_g = (
                    [graphs_by_smiles[s] for s in frame.SMILES] for frame in (train, val, test)
                )
                tr_y = ((train.target.to_numpy() - mean) / scale).tolist()
                va_y = (val.target.to_numpy() - mean) / scale
                for backbone in args.backbones.split(","):
                    for seed in map(int, args.seeds.split(",")):
                        key = f"{backbone}_{variant}_fold{fold}_seed{seed}"
                        result_path = destination / (key + ".json")
                        if result_path.exists():
                            existing = json.loads(result_path.read_text())
                            for field in ("checkpoint", "predictions"):
                                if not (destination / existing[field]).is_file():
                                    raise RuntimeError(f"Missing artifact: {existing[field]}")
                            continue
                        started = time.perf_counter()
                        best_val = float("inf")
                        selected = None
                        tuning = []
                        for lr in map(float, args.learning_rates.split(",")):
                            set_global_seed(seed)
                            model = configure_model(
                                backbone, variant, args.hidden, args.depth, device
                            )
                            optimizer = torch.optim.Adam(
                                model.parameters(), lr=lr, weight_decay=protocol["weight_decay"]
                            )
                            candidate_best, wait, state = float("inf"), 0, None
                            epoch_times = []
                            history = []
                            for epoch in range(1, args.epochs + 1):
                                synchronize(device)
                                tick = time.perf_counter()
                                train_epoch(model, tr_g, tr_y, optimizer, device, args.batch_size)
                                synchronize(device)
                                epoch_times.append(time.perf_counter() - tick)
                                pred_val = predict(model, va_g, device)
                                value = float(np.abs(pred_val - va_y).mean())
                                if not np.isfinite(value):
                                    raise RuntimeError(f"Nonfinite validation error in {key}")
                                history.append(
                                    dict(
                                        epoch=epoch,
                                        validation_mae=value * scale,
                                        train_epoch_seconds=epoch_times[-1],
                                    )
                                )
                                if value < candidate_best:
                                    candidate_best, wait, best_epoch = value, 0, epoch
                                    state = {
                                        k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()
                                    }
                                else:
                                    wait += 1
                                if epoch == 1 or epoch % 10 == 0:
                                    print(
                                        f"{dataset} {key} lr={lr} epoch={epoch} "
                                        f"val={value*scale:.5f} best={candidate_best*scale:.5f} "
                                        f"train_s={epoch_times[-1]:.2f}",
                                        flush=True,
                                    )
                                if wait >= args.patience:
                                    break
                            pd.DataFrame(history).to_csv(
                                destination / (key + f"_lr{lr:g}_history.csv"), index=False
                            )
                            tuning.append(
                                dict(
                                    lr=lr,
                                    validation_mae=candidate_best * scale,
                                    best_epoch=best_epoch,
                                    epochs=epoch,
                                    mean_train_epoch_seconds=float(np.mean(epoch_times)),
                                )
                            )
                            if candidate_best < best_val:
                                best_val = candidate_best
                                selected = (state, len(tuning) - 1)
                            print(
                                f"{dataset} {key} lr={lr} val={candidate_best*scale:.5f} epochs={epoch}",
                                flush=True,
                            )
                        state, chosen = selected
                        model.load_state_dict(state)
                        prediction = predict(model, te_g, device) * scale + mean
                        # Latency uses repeated forwards only after model selection; labels are not consulted.
                        synchronize(device)
                        tick = time.perf_counter()
                        for _ in range(3):
                            predict(model, te_g, device)
                        synchronize(device)
                        latency = (time.perf_counter() - tick) / 3
                        error = prediction - test.target.to_numpy()
                        output = pd.DataFrame(
                            dict(
                                smiles=test.SMILES,
                                scaffold=[_scaffold_group_key(s) for s in test.SMILES],
                                target=test.target,
                                prediction=prediction,
                                absolute_error=np.abs(error),
                                squared_error=error**2,
                                n_atoms=[g["n_atoms"] for g in te_g],
                            )
                        )
                        prediction_path = destination / (key + ".csv.gz")
                        output.to_csv(prediction_path, index=False)
                        checkpoint = destination / (key + ".pt")
                        torch.save(
                            {
                                "state_dict": state,
                                "backbone": backbone,
                                "variant": variant,
                                "hidden": args.hidden,
                                "depth": args.depth,
                                "scaler": {"mean": mean, "scale": scale},
                                "fold": fold,
                                "seed": seed,
                            },
                            checkpoint,
                        )
                        row = dict(
                            dataset=dataset,
                            backbone=backbone,
                            variant=variant,
                            fold=fold,
                            seed=seed,
                            test_mae=float(np.abs(error).mean()),
                            test_rmse=float(np.sqrt(np.mean(error**2))),
                            scaffold_macro_mae=float(
                                output.groupby("scaffold").absolute_error.mean().mean()
                            ),
                            p90_absolute_error=float(np.quantile(np.abs(error), 0.9)),
                            n_test=len(test),
                            validation_mae=best_val * scale,
                            tuning=tuning,
                            selected_candidate=chosen,
                            fit_and_evaluate_seconds=time.perf_counter() - started,
                            inference_seconds=latency,
                            inference_milliseconds_per_molecule=1000 * latency / len(test),
                            trainable_parameters=sum(
                                p.numel() for p in model.parameters() if p.requires_grad
                            ),
                            heat_parameters=extract_heat_parameters(model),
                            checkpoint=checkpoint.name,
                            predictions=prediction_path.name,
                        )
                        atomic_json(result_path, row)
                        print(
                            f'COMPLETE {dataset} {key} test={row["test_mae"]:.5f} elapsed={row["fit_and_evaluate_seconds"]:.1f}s',
                            flush=True,
                        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--datasets", help="Comma-separated built-in dataset keys.")
    inputs.add_argument("--csv", help="Custom structure-only regression CSV.")
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--target-column", default="target")
    p.add_argument(
        "--data-root",
        help="Root containing datasets/csv; defaults to HEATADAPTER_DATA_ROOT or the working directory.",
    )
    p.add_argument("--protocol", default=str(files("graphheat").joinpath("configs/benchmark.json")))
    p.add_argument("--backbones", default="gcn,gine")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seeds", default="104729,130363,155921")
    p.add_argument("--learning-rates", default="0.001,0.0003")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", default="results/heatadapter")
    args = p.parse_args()
    args.datasets = args.datasets or ("custom" if args.csv else ",".join(PRIMARY_DATASETS))
    for field in ("datasets", "backbones", "variants", "folds", "seeds", "learning_rates"):
        values = [value.strip() for value in getattr(args, field).split(",") if value.strip()]
        if not values or len(values) != len(set(values)):
            p.error(f"{field} must contain distinct, nonempty values")
        setattr(args, field, ",".join(values))
    if set(args.variants.split(",")) - set(VARIANTS):
        p.error(f"Unknown variant. Choose from {VARIANTS}")
    if set(args.backbones.split(",")) - set(BACKBONES):
        p.error(f"Unknown backbone. Choose from {BACKBONES}")
    if not args.csv and set(args.datasets.split(",")) - set(PRIMARY_DATASETS):
        p.error(f"Unknown primary dataset. Choose from {PRIMARY_DATASETS}")
    try:
        folds = list(map(int, args.folds.split(",")))
        seeds = list(map(int, args.seeds.split(",")))
        rates = list(map(float, args.learning_rates.split(",")))
    except ValueError:
        p.error("folds and seeds must be integers; learning rates must be numbers")
    if any(fold not in range(5) for fold in folds):
        p.error("folds must be between 0 and 4")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        p.error("seeds must be in [0, 2**32)")
    if any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        p.error("learning rates must be finite and positive")
    for field in ("epochs", "patience", "hidden", "depth", "batch_size", "threads"):
        if getattr(args, field) <= 0:
            p.error(f"{field} must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is not available; use --device cpu")
    for field in ("csv", "data_root", "protocol", "output"):
        if getattr(args, field):
            setattr(args, field, str(Path(getattr(args, field)).expanduser().resolve()))
    run(args)


if __name__ == "__main__":
    main()
