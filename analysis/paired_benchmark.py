#!/usr/bin/env python
"""Analyse paired GraphHeat-adapter effects using split seeds as the unit.

For every dataset, initialization seeds are averaged first.  The script then
compares heat adapters with their matching base backbone (and, optionally, a
parameter-matched MLP control) on exactly the same scaffold/random split seeds.
It intentionally does not pool raw MAE across endpoints with different units.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .pyg_summary import load_rows


def parse_pairs(value: str) -> list[tuple[str, str, str | None]]:
    """Parse ``base:heat[:mlp_control]`` entries separated by commas."""
    pairs = []
    for raw in (item.strip() for item in value.split(",")):
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) not in {2, 3} or any(not part for part in parts):
            raise ValueError(f"Invalid pair {raw!r}; use base:heat[:mlp_control].")
        pairs.append((parts[0], parts[1], parts[2] if len(parts) == 3 else None))
    if not pairs:
        raise ValueError("At least one comparison pair is required.")
    return pairs


def split_means(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["model"]), int(row["split_seed"]))].append(
            float(row["test_mae"])
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_mean_ci(
    values: list[float], seed: int, n_resamples: int = 20_000
) -> tuple[float, float]:
    """Percentile 95% bootstrap CI for a split-level mean difference."""
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(n_resamples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def comparison_table(
    means: dict[tuple[str, str, int], float],
    base: str,
    heat: str,
    control: str | None,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    datasets = sorted({dataset for dataset, _, _ in means})
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        seed_sets = {
            model: {seed for (ds, name, seed) in means if ds == dataset and name == model}
            for model in [base, heat] + ([control] if control else [])
        }
        common = set.intersection(*seed_sets.values()) if seed_sets else set()
        if not common:
            continue
        missing = {
            model: sorted(seed_sets[model] - common)
            for model in seed_sets
            if seed_sets[model] - common
        }
        base_values = [means[(dataset, base, seed)] for seed in sorted(common)]
        heat_values = [means[(dataset, heat, seed)] for seed in sorted(common)]
        deltas = [
            heat_value - base_value for heat_value, base_value in zip(heat_values, base_values)
        ]
        relative = [
            delta / max(abs(base_value), 1e-12) for delta, base_value in zip(deltas, base_values)
        ]
        low, high = bootstrap_mean_ci(deltas, bootstrap_seed + len(output))
        entry: dict[str, Any] = {
            "dataset": dataset,
            "base": base,
            "heat": heat,
            "control": control,
            "n_splits": len(common),
            "split_seeds": sorted(common),
            "base_mae_mean": float(np.mean(base_values)),
            "heat_mae_mean": float(np.mean(heat_values)),
            "heat_minus_base_mae": float(np.mean(deltas)),
            "heat_minus_base_bootstrap_95ci": [low, high],
            "relative_change_percent": float(100.0 * np.mean(relative)),
            "heat_wins": int(sum(delta < 0 for delta in deltas)),
            "base_wins": int(sum(delta > 0 for delta in deltas)),
            "ties": int(sum(delta == 0 for delta in deltas)),
            "unpaired_split_seeds": missing,
        }
        if control is not None:
            control_values = [means[(dataset, control, seed)] for seed in sorted(common)]
            control_deltas = [
                heat_value - control_value
                for heat_value, control_value in zip(heat_values, control_values)
            ]
            control_low, control_high = bootstrap_mean_ci(
                control_deltas, bootstrap_seed + 10_000 + len(output)
            )
            entry.update(
                {
                    "control_mae_mean": float(np.mean(control_values)),
                    "heat_minus_control_mae": float(np.mean(control_deltas)),
                    "heat_minus_control_bootstrap_95ci": [control_low, control_high],
                    "heat_vs_control_wins": int(sum(delta < 0 for delta in control_deltas)),
                }
            )
        output.append(entry)
    return output


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Paired GraphHeat Adapter Analysis", ""]
    lines.append(f"Campaign: `{report['campaign_id']}`")
    lines.append("Negative MAE differences favor the heat adapter.")
    lines.append("")
    for comparison in report["comparisons"]:
        base, heat, control = comparison["pair"]
        heading = f"{heat} vs {base}" + (f" vs {control}" if control else "")
        lines.extend([f"## {heading}", ""])
        columns = (
            "| dataset | n splits | base MAE | heat MAE | heat − base | 95% bootstrap CI | "
            "relative change | heat wins | unpaired seeds |"
        )
        lines.extend([columns, "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"])
        for item in comparison["datasets"]:
            ci = item["heat_minus_base_bootstrap_95ci"]
            lines.append(
                f"| {item['dataset']} | {item['n_splits']} | {item['base_mae_mean']:.4f} | "
                f"{item['heat_mae_mean']:.4f} | {item['heat_minus_base_mae']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {item['relative_change_percent']:.1f}% | "
                f"{item['heat_wins']}/{item['n_splits']} | {item['unpaired_split_seeds']} |"
            )
        lines.append("")
        if control:
            lines.extend(
                [
                    "| dataset | MLP-control MAE | heat − control | 95% bootstrap CI | heat wins vs control |",
                    "| --- | ---: | ---: | --- | ---: |",
                ]
            )
            for item in comparison["datasets"]:
                ci = item["heat_minus_control_bootstrap_95ci"]
                lines.append(
                    f"| {item['dataset']} | {item['control_mae_mean']:.4f} | "
                    f"{item['heat_minus_control_mae']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                    f"{item['heat_vs_control_wins']}/{item['n_splits']} |"
                )
            lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--pairs",
        default=(
            "gcn:gcn_heat:gcn_mlp_adapter,gin:gin_heat:gin_mlp_adapter,"
            "gine:gine_heat:gine_mlp_adapter,mpnn:mpnn_heat:mpnn_mlp_adapter,"
            "gatv2:gatv2_heat:gatv2_mlp_adapter"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()

    rows, audit = load_rows(Path(args.result_dir), campaign_id=args.campaign_id)
    means = split_means(rows)
    comparisons = []
    for index, (base, heat, control) in enumerate(parse_pairs(args.pairs)):
        comparisons.append(
            {
                "pair": [base, heat, control],
                "datasets": comparison_table(
                    means, base, heat, control, args.bootstrap_seed + index * 100_000
                ),
            }
        )
    report = {"campaign_id": audit["campaign_id"], "audit": audit, "comparisons": comparisons}
    write_markdown(report, Path(args.output))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
