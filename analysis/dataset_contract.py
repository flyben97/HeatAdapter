#!/usr/bin/env python
"""Create manuscript-ready data-curation and split-contract records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphheat.mol_data import (
    DATASET_METADATA,
    PRIMARY_DATASETS,
    load_dataset,
    random_split,
    scaffold_split,
    split_diagnostics,
)


def parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_contract(
    dataset: str, split_seeds: list[int], data_root: str | None = None
) -> dict[str, Any]:
    frame, curation = load_dataset(dataset, base_dir=data_root, return_report=True)
    target = frame["target"]
    splits = []
    for split_seed in split_seeds:
        scaffold = scaffold_split(frame, seed=split_seed)
        random = random_split(frame, seed=split_seed)
        splits.append(
            {
                "split_seed": split_seed,
                "scaffold": split_diagnostics(*scaffold),
                "random_molecule_disjoint": split_diagnostics(*random),
            }
        )
    return {
        "dataset": dataset,
        "metadata": DATASET_METADATA[dataset],
        "curation": curation,
        "target_summary": {
            "min": float(target.min()),
            "mean": float(target.mean()),
            "std": float(target.std(ddof=1)),
            "median": float(target.median()),
            "max": float(target.max()),
        },
        "split_contracts": splits,
    }


def write_markdown(contracts: list[dict[str, Any]], output: Path) -> None:
    lines = ["# GraphHeat Dataset Contracts", ""]
    lines.extend(
        [
            "Each primary endpoint is canonicalized and de-duplicated before splitting.",
            "Random splits are molecule-disjoint; scaffold splits use Bemis–Murcko scaffolds with canonical-molecule grouping for acyclic compounds.",
            "",
            "| dataset | endpoint | unit | raw rows | invalid SMILES removed | duplicates collapsed | final molecules |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for contract in contracts:
        curation = contract["curation"]
        meta = contract["metadata"]
        lines.append(
            f"| {contract['dataset']} | {meta['endpoint']} | {meta['unit']} | "
            f"{curation['raw_rows']} | {curation['invalid_smiles_removed']} | "
            f"{curation['canonical_duplicates_collapsed']} | {curation['final_rows']} |"
        )
    lines.extend(["", "## Split Diagnostics", ""])
    for contract in contracts:
        lines.extend(
            [
                f"### {contract['dataset']}",
                "",
                "| split seed | protocol | train | validation | test | molecule overlap | Murcko overlap |",
                "| ---: | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for split in contract["split_contracts"]:
            for protocol, label in [
                ("scaffold", "Murcko scaffold"),
                ("random_molecule_disjoint", "molecule-disjoint random"),
            ]:
                item = split[protocol]
                lines.append(
                    f"| {split['split_seed']} | {label} | {item['n_train']} | {item['n_val']} | "
                    f"{item['n_test']} | {item['molecule_overlap']} | {item['murcko_scaffold_overlap']} |"
                )
        lines.append("")
    lines.extend(["## Source Provenance", ""])
    lines.extend(
        [
            "The source files listed above are immutable local snapshots identified by SHA-256 in the JSON contract.",
            "",
            "| dataset | source identifier | endpoint / transformation note |",
            "| --- | --- | --- |",
        ]
    )
    for contract in contracts:
        meta = contract["metadata"]
        lines.append(
            f"| {contract['dataset']} | {meta.get('source_identifier', 'local source record pending')} | "
            f"{meta.get('source_note', '')} |"
        )
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(PRIMARY_DATASETS))
    parser.add_argument("--split-seeds", default="42,7,13,21,99")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="results/dataset_contracts.md")
    parser.add_argument("--json-output", default="results/dataset_contracts.json")
    args = parser.parse_args()
    contracts = [
        build_contract(dataset, parse_csv(args.split_seeds, int), args.data_root)
        for dataset in parse_csv(args.datasets)
    ]
    write_markdown(contracts, Path(args.output))
    Path(args.json_output).write_text(json.dumps(contracts, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
