# Research data

Raw research datasets are external inputs. The package includes a synthetic
example in `examples/toy_molecules.csv`; its targets are computed molecular
masses, not experimental measurements.

For a named dataset, place its CSV at the path below relative to `--data-root`.
The default root is `HEATADAPTER_DATA_ROOT`, or the working directory if unset.
For other filenames or schemas, use `heatadapter-train --csv ...` with
`--smiles-column` and `--target-column`.

| Key | Path relative to data root | SMILES column | Target column |
|---|---|---|---|
| `lipophilicity` | `datasets/csv/default/logP_lipophilicity.csv` | `smiles` | `exp` |
| `bradley_dpg` | `datasets/csv/default/melting_point_bradley_double_plus_good.csv` | `smiles` | `mpC` |
| `thermosol` | `datasets/csv/default/solubility_thermosol.csv` | `smile` | `target` |
| `esol` | `datasets/csv/backup/logS_esol_delaney.csv` | `smiles` | `measured log solubility in mols per litre` |
| `freesolv` | `datasets/csv/backup/hydration_free_energy_freesolv_sampl.csv` | `smiles` | `expt` |

## Provenance

| Dataset | Source | Original snapshot rows | Target scale |
|---|---|---:|---|
| Lipophilicity | [Hop et al.](https://doi.org/10.1021/acs.molpharmaceut.7b01144) / AstraZeneca–ChEMBL | 4,200 | logD at pH 7.4 |
| Bradley DPG | [Double Plus Good Figshare record](https://doi.org/10.6084/m9.figshare.1031638), Bradley, Lang, Williams (2014) | 3,041 | degrees Celsius |
| ThermoSol | [ChEMBL assay CHEMBL3301364](https://www.ebi.ac.uk/chembl/assay_report_card/CHEMBL3301364/) and [Hop et al.](https://doi.org/10.1021/acs.molpharmaceut.7b01144) | 1,763 | log10(solubility / nM) |
| ESOL | [Delaney (2004)](https://doi.org/10.1021/ci034243x) / [MoleculeNet](https://doi.org/10.1039/C7SC02664A) | 1,128 | log10(mol/L) |
| FreeSolv | [Mobley and Guthrie (2014)](https://doi.org/10.1007/s10822-014-9747-x) | 642 | kcal/mol |

`source_snapshots.json` identifies the exact historical local CSV snapshots by
SHA-256. Downloads or exports from an updated upstream source can differ; the
loader records the hash of every actual input. The named ThermoSol input expects
the already prepared log10(nM) target; it does not transform a raw ChEMBL export.
Subtract nine to express the same solubility on a molar log scale. No implicit
unit conversion is applied by the loader.

The DPG record contains 3,041 measurements before local curation; the historical
snapshot produced 3,022 unique valid molecules. Exact curated counts depend on
the input snapshot and RDKit version. Data attribution and terms remain those of
the original providers.

To generate curation and historical split diagnostics from supplied files:

```bash
python -m analysis.dataset_contract --datasets esol \
  --data-root /path/to/data-root \
  --output results/dataset_contracts.md \
  --json-output results/dataset_contracts.json
```

This diagnostic command describes the historical repeated-split interface. The
recommended disjoint-fold runner writes its own `split_audit.json` and
`split_assignments.csv` automatically.
