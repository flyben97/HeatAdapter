#!/usr/bin/env python
"""Validate and summarize versioned GraphHeat benchmark result files.

The statistical unit is the split seed: initialization replicates are averaged
within each ``(dataset, model, split_seed)`` before dataset-level means and
standard deviations are calculated.  Schema-v2 result files carry a campaign
identifier and run metadata.  By default this script rejects legacy files,
duplicate keys, and incomplete expected result grids rather than silently
mixing retries into a manuscript table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_ROW_KEYS = {"dataset", "model", "split_seed", "test_mae"}


def _parse_csv(value: str | None, cast=str) -> list[Any] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return [cast(item) for item in values]


def _load_one(path: Path, allow_legacy: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        metadata = dict(payload.get("metadata", {}))
        schema_version = metadata.get("schema_version")
        if schema_version != 2:
            raise ValueError(f"{path.name}: expected schema_version=2, found {schema_version!r}")
        return metadata, [dict(row) for row in payload["rows"]]
    if isinstance(payload, list):
        if not allow_legacy:
            raise ValueError(
                f"{path.name}: legacy schema-v1 result file. Use --allow-legacy only for forensic inspection, "
                "never for a manuscript summary."
            )
        return {"schema_version": 1, "campaign_id": "legacy", "run_id": path.stem}, [
            dict(row) for row in payload
        ]
    raise ValueError(f"{path.name}: unsupported benchmark JSON structure")


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    """Return the unique statistical-key for one initialization replicate."""
    init_seed = row.get("init_seed", row.get("seed"))
    if init_seed is None:
        raise ValueError(f"Result row lacks init_seed/seed: {row}")
    return (
        str(row.get("campaign_id", "")),
        str(row["dataset"]),
        str(row["model"]),
        int(row["split_seed"]),
        int(init_seed),
    )


def load_rows(
    result_dir: Path,
    *,
    campaign_id: str | None = None,
    allow_legacy: bool = False,
    allow_duplicates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one campaign and fail closed on ambiguous benchmark evidence."""
    paths = sorted(result_dir.glob("benchmark_*.json"))
    if not paths:
        raise FileNotFoundError(f"No benchmark_*.json files found in {result_dir}")

    rows: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    for path in paths:
        metadata, file_rows = _load_one(path, allow_legacy)
        file_campaign = str(metadata.get("campaign_id", ""))
        if campaign_id is not None and file_campaign != campaign_id:
            continue
        for raw_row in file_rows:
            if not REQUIRED_ROW_KEYS <= set(raw_row):
                continue
            row = dict(raw_row)
            row_campaign = str(row.get("campaign_id", file_campaign))
            if row_campaign != file_campaign:
                raise ValueError(
                    f"{path.name}: row campaign_id={row_campaign!r} disagrees with metadata campaign_id={file_campaign!r}"
                )
            row["campaign_id"] = row_campaign
            row["init_seed"] = int(row.get("init_seed", row.get("seed")))
            row["_source"] = path.name
            rows.append(row)
        source_metadata.append({"source": path.name, **metadata})

    if not rows:
        available = sorted({str(item.get("campaign_id", "")) for item in source_metadata})
        raise ValueError(
            f"No usable rows for campaign {campaign_id!r}. Available campaigns: {available}"
        )
    campaigns = sorted({str(row["campaign_id"]) for row in rows})
    if campaign_id is None and len(campaigns) != 1:
        raise ValueError(
            f"Multiple campaigns found: {campaigns}. Pass --campaign-id to prevent mixing independent runs."
        )

    by_key: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[row_key(row)].append(row)
    duplicates = {key: values for key, values in by_key.items() if len(values) > 1}
    if duplicates and not allow_duplicates:
        sample = [
            {"key": key, "sources": [item["_source"] for item in values]}
            for key, values in list(duplicates.items())[:10]
        ]
        raise ValueError(
            f"Detected {len(duplicates)} duplicate replicate keys. Refusing to average retries. Sample: {sample}"
        )
    if duplicates:
        # Explicit forensic mode: retain the final lexical source for each key.
        rows = [sorted(values, key=lambda item: item["_source"])[-1] for values in by_key.values()]

    audit = {
        "campaign_id": campaigns[0],
        "n_files_considered": len(paths),
        "n_source_files": len(source_metadata),
        "n_rows": len(rows),
        "n_unique_replicate_keys": len({row_key(row) for row in rows}),
        "source_metadata": source_metadata,
    }
    return rows, audit


def validate_completeness(
    rows: list[dict[str, Any]],
    *,
    datasets: list[str] | None,
    models: list[str] | None,
    split_seeds: list[int] | None,
    init_seeds: list[int] | None,
) -> dict[str, Any]:
    """Compare rows against an explicitly declared planned benchmark grid."""
    declared = [datasets, models, split_seeds, init_seeds]
    if all(value is None for value in declared):
        return {"checked": False, "missing": [], "unexpected": []}
    if any(value is None for value in declared):
        raise ValueError(
            "Completeness checking requires --expected-datasets, --expected-models, "
            "--expected-split-seeds, and --expected-init-seeds together."
        )
    campaign = str(rows[0]["campaign_id"])
    expected = {
        (campaign, dataset, model, int(split_seed), int(init_seed))
        for dataset, model, split_seed, init_seed in itertools.product(
            datasets, models, split_seeds, init_seeds
        )
    }
    observed = {row_key(row) for row in rows}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {
        "checked": True,
        "expected_replicates": len(expected),
        "observed_replicates": len(observed),
        "missing": missing,
        "unexpected": unexpected,
    }


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return math.nan
    mu = mean(vals)
    return (sum((value - mu) ** 2 for value in vals) / (len(vals) - 1)) ** 0.5


def summarize(rows: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    """Aggregate initialization replicates, then independent split replicates."""
    init_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        init_groups[(str(row["dataset"]), str(row["model"]), int(row["split_seed"]))].append(
            float(row["test_mae"])
        )

    split_means: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, model, split_seed), values in init_groups.items():
        split_means[(dataset, model)].append(
            {
                "split_seed": split_seed,
                "n_init": len(values),
                "mean_test_mae": mean(values),
            }
        )

    datasets = sorted({dataset for dataset, _ in split_means})
    models = sorted({model for _, model in split_means})
    tables: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        entries = []
        for model in models:
            splits = sorted(
                split_means.get((dataset, model), []), key=lambda item: item["split_seed"]
            )
            if not splits:
                continue
            values = [float(item["mean_test_mae"]) for item in splits]
            entries.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_splits": len(splits),
                    "split_seeds": [int(item["split_seed"]) for item in splits],
                    "test_mae_mean": mean(values),
                    "test_mae_sd": stdev(values),
                    "split_means": splits,
                }
            )
        entries.sort(key=lambda item: float(item["test_mae_mean"]))
        for rank, item in enumerate(entries, start=1):
            item["rank"] = rank
        tables[dataset] = entries

    by_model_rank: dict[str, list[int]] = defaultdict(list)
    for entries in tables.values():
        for item in entries:
            by_model_rank[str(item["model"])].append(int(item["rank"]))
    avg_ranks = [
        {
            "model": model,
            "n_datasets": len(ranks),
            "avg_rank": mean([float(rank) for rank in ranks]),
            "ranks": ranks,
        }
        for model, ranks in by_model_rank.items()
    ]
    avg_ranks.sort(key=lambda item: float(item["avg_rank"]))
    return {
        **audit,
        "datasets": datasets,
        "models": models,
        "tables": tables,
        "avg_ranks": avg_ranks,
    }


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = ["# GraphHeat Benchmark Summary", ""]
    lines.extend(
        [
            f"Campaign: `{summary['campaign_id']}`",
            f"Replicate rows: {summary['n_rows']}",
            f"Unique replicate keys: {summary['n_unique_replicate_keys']}",
            "",
            "## Dataset Tables",
            "",
        ]
    )
    for dataset, entries in summary["tables"].items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| rank | model | test MAE mean | test MAE SD across splits | n splits | split seeds |",
                "| ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in entries:
            sd = (
                "NA"
                if math.isnan(float(item["test_mae_sd"]))
                else f"{float(item['test_mae_sd']):.4f}"
            )
            lines.append(
                f"| {item['rank']} | {item['model']} | {float(item['test_mae_mean']):.4f} | "
                f"{sd} | {item['n_splits']} | {item['split_seeds']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Average Ranks",
            "",
            "| model | average rank | n datasets | ranks |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for item in summary["avg_ranks"]:
        lines.append(
            f"| {item['model']} | {float(item['avg_rank']):.2f} | "
            f"{item['n_datasets']} | {item['ranks']} |"
        )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default="results/benchmark_v2")
    parser.add_argument("--output", default="results/benchmark_v2/SUMMARY.md")
    parser.add_argument("--json-output", default=None)
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Required when a directory contains more than one campaign.",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Inspect schema-v1 files. Not suitable for a manuscript table.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Forensic mode only: retain the final lexical source for duplicate keys.",
    )
    parser.add_argument("--expected-datasets", default=None)
    parser.add_argument("--expected-models", default=None)
    parser.add_argument("--expected-split-seeds", default=None)
    parser.add_argument("--expected-init-seeds", default=None)
    args = parser.parse_args()

    rows, audit = load_rows(
        Path(args.result_dir),
        campaign_id=args.campaign_id,
        allow_legacy=args.allow_legacy,
        allow_duplicates=args.allow_duplicates,
    )
    completeness = validate_completeness(
        rows,
        datasets=_parse_csv(args.expected_datasets),
        models=_parse_csv(args.expected_models),
        split_seeds=_parse_csv(args.expected_split_seeds, int),
        init_seeds=_parse_csv(args.expected_init_seeds, int),
    )
    if completeness["checked"] and (completeness["missing"] or completeness["unexpected"]):
        raise ValueError(
            "Benchmark completeness check failed: "
            f"missing={len(completeness['missing'])}, unexpected={len(completeness['unexpected'])}."
        )
    audit["completeness"] = completeness
    summary = summarize(rows, audit)
    write_markdown(summary, Path(args.output))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
