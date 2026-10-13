# HeatAdapter

Learnable multiscale heat diffusion for molecular property prediction. HeatAdapter
adds a spectral propagation branch to local molecular graph neural networks and
combines the two paths through learned gates.

![HeatAdapter architecture](docs/assets/heatadapter_architecture.svg)

*Multiscale heat diffusion and gated fusion with local molecular message passing.
The schematic retains the manuscript's original spectral cutoff; this release
uses full spectra by default.*

[中文说明](README.zh-CN.md) · [Data preparation](datasets/README.md) ·
[Evaluation protocol](docs/protocol.md) · [Release checks](docs/validation.md)

This repository contains the corrected full-spectrum implementation, training
and prediction tools, and numerical regression tests. The manuscript's historical
benchmark used an earlier truncated implementation. Those results do not measure
this corrected implementation; a full replacement comparative campaign remains
to be completed. The included toy example demonstrates software execution only.

## Installation

Use Python 3.11 or newer. The local release checks use Python 3.13 on CPU;
`requirements.txt` records the tested direct dependency versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

On Windows, activate with `.venv\Scripts\activate`. For GPU training, install
the appropriate PyTorch CUDA build instead of the CPU command above. The import
name remains `graphheat` for compatibility with the existing model classes.
No separately compiled PyG extension is needed by the included CPU checks.

## Quick start

The repository includes 55 small molecules with molecular-mass targets computed
by RDKit. These synthetic examples are independent of the research datasets.

```bash
python -m unittest discover -s tests -v
python smoke_test.py

heatadapter-train \
  --csv examples/toy_molecules.csv \
  --backbones gcn --variants base,heat,mlp \
  --folds 0 --seeds 42 --learning-rates 0.001 \
  --epochs 2 --patience 2 --hidden 32 --depth 2 \
  --device cpu --output results/toy

heatadapter-predict \
  --checkpoint results/toy/custom/gcn_heat_fold0_seed42.pt \
  --csv examples/toy_molecules.csv \
  --output results/toy/predictions.csv
```

`python -m analysis.revision_benchmark` and `python -m graphheat.predict` are
equivalent to the two console commands. Training writes targets and predictions
on the original target scale. Prediction preserves the input rows and columns
and appends `prediction`; a target column is not required for prediction.

## Train on your data

Supply a CSV with a SMILES column and a numeric regression target:

```bash
heatadapter-train \
  --csv my_molecules.csv --smiles-column smiles --target-column target \
  --backbones gcn,gine --variants base,heat,mlp \
  --folds 0,1,2,3,4 --seeds 104729,130363,155921 \
  --device auto --output results/my_experiment
```

The loader canonicalizes SMILES, removes invalid structures and nonfinite targets,
and averages duplicate canonical structures before splitting. This assumes one
structure-only endpoint under comparable measurement conditions. The splitter
keeps whole nonempty Bemis–Murcko scaffolds together; acyclic molecules are grouped
by canonical identity. At least six scaffold/molecule groups are required, and
inputs producing an empty partition are rejected.

Five outer folds test every curated molecule exactly once for each model and
initialization. `--folds 0` runs a partial diagnostic only. Epoch and learning-rate
selection use validation labels; the selected model is then evaluated on its test
fold. Target standardization uses training statistics only.

The named research datasets are `lipophilicity`, `thermosol`, `bradley_dpg`,
`esol`, and `freesolv`. Raw datasets are supplied separately:

```bash
heatadapter-train --datasets esol --data-root /path/to/data-root \
  --backbones gcn,gine --variants base,heat,mlp \
  --device auto --output results/esol
```

See [datasets/README.md](datasets/README.md) for exact file paths, columns,
source records, and historical snapshot hashes. `HEATADAPTER_DATA_ROOT` can
replace `--data-root`.

## Models and controls

The disjoint-fold runner supports GCN, GIN, GINE, NNConv/MPNN, and GATv2
(`gcn`, `gin`, `gine`, `mpnn`, `gatv2`).

| Variant | Meaning |
|---|---|
| `base` | Local backbone |
| `mlp` | Approximately parameter-matched MLP adapter |
| `heat` | Learnable multiscale heat adapter, full spectrum |
| `identity` | Adapter with identity propagation |
| `zero` | Complete zero-eigenspace projection |
| `fixed` | Fixed diffusion times, learned mixture weights |
| `single` | One learned diffusion time |
| `unweighted` | Full-spectrum heat on an unweighted molecular graph |
| `topk20` | At least 20 eigenmodes, extending through degenerate eigenspaces |

The historical-style runner, `python -m graphheat.mol_bench --help`, additionally
provides ECFP4–RF, AttentiveFP, PNA, GPS, and standalone variants. Its repeated
splits can overlap across runs. `analysis.pyg_summary` and
`analysis.paired_benchmark` read that runner's schema-v2 output; they do not read
the disjoint-fold runner's per-fit output. See [docs/protocol.md](docs/protocol.md).

## Outputs and resuming

Each experiment contains:

```text
manifest.json                         # effective settings, versions, source/data hashes
custom/                               # or a named dataset
  curation.json
  split_assignments.csv
  split_audit.json
  gcn_heat_fold0_seed42.json           # metrics and validation candidate selection
  gcn_heat_fold0_seed42.pt             # selected weights and training target scaler
  gcn_heat_fold0_seed42.csv.gz         # per-molecule held-out predictions
  gcn_heat_fold0_seed42_lr0.001_history.csv
```

Repeat the same command to resume: completed fits with existing artifacts are
skipped. Changed code, inputs, or settings require a new output directory. Resume
is at the completed-fit level; an interrupted fit restarts. Artifact links in
per-fit JSON files are relative to that JSON's directory. A copied output folder
can be used for prediction, but its original manifest still records the original
run locations. Use separate output directories for concurrent runs.

GPU reductions may be nondeterministic. Fold sizes can differ substantially when
scaffold groups are large. Benchmark interpretation and the remaining validation
limits are documented in [docs/protocol.md](docs/protocol.md).

## Repository layout

```text
graphheat/              # models, operators, molecular data, prediction, default protocol
analysis/               # disjoint-fold training and historical-schema analysis tools
tests/                  # numerical, split, input, checkpoint, and resume regression tests
examples/               # self-contained synthetic CSV and its generator
datasets/               # source documentation and snapshot identifiers
docs/                   # protocol and release verification
.github/workflows/      # CPU continuous integration
```

## License

The code is distributed under the [MIT License](LICENSE). Third-party packages
and separately obtained datasets retain their own licenses and attribution.
