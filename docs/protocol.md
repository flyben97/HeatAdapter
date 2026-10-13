# Evaluation protocol and implementation scope

## Corrected implementation

The default operator uses the full spectrum of a symmetric normalized molecular
graph Laplacian. Eigendecompositions are calculated in float64 and stored in the
training precision. An explicit positive cutoff is expanded through the complete
numerically degenerate boundary eigenspace. Variable-rank graphs are padded for
batched spectral projection. Conformer generation is disabled by default.

The local heat adapters initialize three diffusion times at 0.2, 1, and 5, with
equal mixture weights. Feature projection, residual connection, layer
normalization, and learned local/heat gates follow the original model definitions.
Bond categories are assigned scalar weights; these are numerical model inputs,
not physical bond orders or conductivities. Current atom features do not explicitly
encode stereochemistry.

Identity, zero-mode, fixed-time, single-kernel, unweighted-Laplacian, and safe
20-mode controls are available in the training runner. Their presence is not a
claim that a completed scientific benchmark established their relative performance.

## Recommended disjoint-fold evaluation

`analysis.revision_benchmark` uses five balanced scaffold-group test folds, with
each curated molecule tested exactly once per model/initialization when all folds
are run. Within each outer fold, validation groups are drawn from the remaining
groups, targeting roughly 70/10/20 training/validation/test proportions. Large
scaffold groups can make the realized fractions uneven. Group allocation never
uses target values.

The default protocol is [benchmark.json](../graphheat/configs/benchmark.json).
The CLI controls the effective model, fold, initialization, architecture, epoch,
patience, and learning-rate grid settings. The protocol supplies the split seed
and weight decay; use `--protocol` to supply a modified JSON. The manifest stores
both effective CLI settings and the protocol hash. The default learning-rate
budget is identical across model variants. Test labels are used only after
validation selection. Per-epoch histories report validation MAE and training time.

For a complete evaluation, combine held-out molecule errors across all five folds
for each initialization, then average each molecule's errors over initializations.
Report molecule-weighted means. Averaging predictions first evaluates an ensemble
and answers a different question. Fold-level standard deviations and overlapping
historical split intervals do not establish independent chemical validation.

## Historical benchmark interface

The `graphheat.mol_bench` runner retains the broader baseline registry and schema-v2
output. Example, with separately supplied ESOL data:

```bash
HEATADAPTER_DATA_ROOT=/path/to/data-root python -m graphheat.mol_bench \
  --campaign-id example_baselines --run-id esol_seed42 \
  --datasets esol --models ecfp4_rf,gcn,gcn_heat,gcn_mlp_adapter \
  --seeds 42 --split scaffold --split-seed 42 \
  --epochs 300 --spectral-topk 0 --device cpu \
  --output-dir results/baselines

python -m analysis.pyg_summary \
  --result-dir results/baselines --campaign-id example_baselines \
  --output results/baselines/SUMMARY.md
```

`analysis.paired_benchmark` performs exploratory paired summaries for this schema.
Repeated historical-style splits can share test molecules. These tools do not
consume the disjoint-fold runner's per-fit files. The current operator differs
from the fixed-rank float32 operator used to generate the manuscript's historical
`jcim_v2_full` results, so running this interface does not reproduce those exact
historical metrics.

## Evidence boundaries

The original evaluation found overlapping test populations, shared molecules
between lipophilicity and ThermoSol, and atom-order sensitivity at truncated
degenerate eigenspaces. Numerical tests verify the repaired implementation on
their tested cases. The full corrected comparative training campaign has not
been completed; local toy fits are software checks. New folds on previously
inspected datasets do not create an untouched external validation cohort.

GPU reductions can be nondeterministic. Runtime comparisons require matched
hardware and workloads. The publication's result tables, checkpoints and figures
are separate historical artifacts and are not bundled as outputs of this release.
