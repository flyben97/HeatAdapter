# Release verification

Checked on 2026-10-13 using Linux, Python 3.13.5 and CPU execution.

| Dependency | Version |
|---|---|
| PyTorch | 2.10.0+cpu |
| PyTorch Geometric | 2.7.0 |
| RDKit | 2025.9.4 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |

## Completed checks

- All **8 unittest cases passed**. The original four scientific regression tests
  cover atom-order invariance, agreement with a dense matrix exponential, disjoint
  scaffold test folds, and mixed-rank batching/propagation controls.
- Input tests check custom CSV column names, canonical duplicate averaging,
  invalid structures, nonfinite targets, empty curated inputs, and a named dataset
  loaded from an external root.
- The training artifact test runs **all nine variants** on synthetic data using
  two validation-selected learning-rate candidates. Saved checkpoints reproduce
  the persisted test predictions. Resume skips completed fits, and changing the
  input data causes the manifest guard to reject reuse of the existing output.
- The standalone smoke test passed for **11 model entries**, including local,
  standalone, heat-adapted and MLP-control models; it also checks split integrity
  and the parameter-count tolerance of the GINE heat/MLP pair.
- A wheel was built and installed in a separate virtual environment, reusing the
  already installed dependency packages. From a directory outside the checkout,
  the installed `heatadapter-train` command completed **15 synthetic fits**
  (GCN base/heat/MLP, five folds, one initialization, one epoch), and
  `heatadapter-predict` produced a CSV. The installed package contained its default
  protocol resource and imported from the environment's site-packages directory.
- AST comparisons confirmed that model/operator logic, molecular featurization,
  spectral preprocessing and split algorithms match the source project. Release
  changes concern packaging, documentation, external input handling, dataset
  provenance, argument validation, saved-artifact paths and resume checks.
- Black formatting checks passed. The release source was checked for local
  machine paths, cluster mounts and credential assignments. Raw research data,
  checkpoints, manuscript files, logs, build products and Python caches are
  excluded from the source archive.

These checks establish software behavior on the tested CPU environment. They do
not constitute a completed predictive-performance benchmark. CUDA training and
the configured GitHub Actions matrix have not been executed as part of this local
release preparation. Python 3.13 emits an upstream PyG typing deprecation warning;
it does not fail the tested operations.
