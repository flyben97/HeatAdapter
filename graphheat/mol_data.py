"""Molecular data pipeline: SMILES to graph conversion, scaffold split, batching.

Designed for the PyG-corrected GraphHeat benchmark. Models receive:
  x: (N, in_dim)        - node features
  edge_index: (2, E)    - graph connectivity
  edge_weight: (E,)     - bond-type scalar weight
  batch: (N,)           - node→graph assignment
  pos: (N, 3)           - optional 3D coordinates for model extensions
  y: (G, 1)             - graph-level target, one value per molecular graph.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Spectral truncation control (module-level, set before data preparation)
# ---------------------------------------------------------------------------

_DEFAULT_SPECTRAL_TOPK: int | None = None


def set_spectral_topk(k: int | None) -> None:
    """Set the default number of eigenpairs to keep for GraphHeat spectral ops.

    Set before calling prepare_split().  ``None`` means keep all eigenvectors.
    Full spectrum is recommended. Positive cutoffs expand through degeneracies.
    """
    global _DEFAULT_SPECTRAL_TOPK
    _DEFAULT_SPECTRAL_TOPK = k


# ---------------------------------------------------------------------------
# Atom / bond featurization
# ---------------------------------------------------------------------------

# Elements commonly seen in organic / drug-like molecules (atomic numbers)
_ATOMIC_NUM_LIST = [
    0,
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    26,
    27,
    29,
    30,
    33,
    34,
    35,
    46,
    47,
    50,
    53,
    78,
    79,
    80,
    82,
    83,
]
_ATOMIC_NUM_TO_IDX = {z: i for i, z in enumerate(_ATOMIC_NUM_LIST)}

# Hybridization types
_HYB_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_HYB_TO_IDX = {h: i for i, h in enumerate(_HYB_TYPES)}

# Bond types
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_TO_IDX = {b: i for i, b in enumerate(_BOND_TYPES)}


def atom_features(mol: Chem.Mol) -> np.ndarray:
    """Fixed-dimension atom feature vector for every atom in *mol*.

    Returns (num_atoms, ATOM_FEAT_DIM) float32 array.
    """
    feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        z_idx = _ATOMIC_NUM_TO_IDX.get(z, 0)  # unknown → 0
        z_onehot = np.zeros(len(_ATOMIC_NUM_LIST), dtype=np.float32)
        z_onehot[z_idx] = 1.0

        degree = min(atom.GetDegree(), 6)
        charge = max(-3, min(3, atom.GetFormalCharge()))
        num_h = min(atom.GetTotalNumHs(), 4)
        hyb = atom.GetHybridization()
        hyb_idx = _HYB_TO_IDX.get(hyb, 0)
        hyb_onehot = np.zeros(len(_HYB_TYPES), dtype=np.float32)
        hyb_onehot[hyb_idx] = 1.0

        scalar = np.array(
            [
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(degree) / 6.0,
                float(charge + 3) / 6.0,  # shift to [0,1]
                float(num_h) / 4.0,
                float(atom.GetAtomicNum()) / 100.0,  # continuous atomic-number signal
            ],
            dtype=np.float32,
        )

        feats.append(np.concatenate([z_onehot, hyb_onehot, scalar]))

    if not feats:
        return np.zeros((0, _ATOM_FEAT_DIM), dtype=np.float32)
    return np.stack(feats, axis=0)


_ATOM_FEAT_DIM = len(_ATOMIC_NUM_LIST) + len(_HYB_TYPES) + 6  # 34 + 6 + 6 = 46


def bond_weight(bond: Chem.Bond) -> float:
    """Scalar weight for a bond (0–1)."""
    btype = bond.GetBondType()
    idx = _BOND_TO_IDX.get(btype, 0)
    return (idx + 1.0) / len(_BOND_TYPES)  # map to (0, 1]


# ---------------------------------------------------------------------------
# SMILES → graph conversion
# ---------------------------------------------------------------------------


def mol_to_graph(
    mol: Chem.Mol,
    spectral_topk: int | None = None,
    *,
    generate_conformer: bool = False,
    unweighted_heat: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a single RDKit molecule to graph tensors.

    Args:
        mol: RDKit molecule.
        spectral_topk: if set, retain at least k eigenvalues/vectors for truncated
            spectral decomposition (used by GraphHeat).  None = keep all.
            Falls back to module-level `_DEFAULT_SPECTRAL_TOPK` if not provided.
            Zero explicitly requests all modes; a positive boundary expands
            through its numerically degenerate eigenspace.
        generate_conformer: explicitly enable unused-by-default 3D coordinates.
        unweighted_heat: use unit bond coupling for spectra, retaining original
            local edge features for an edge-information ablation.

    Returns dict with keys: x, edge_index, edge_weight, pos, n_atoms, eigvals, eigvecs.
    """
    # Node features
    x = torch.from_numpy(atom_features(mol))  # (N, F)

    # Edges (undirected with self-loops)
    rows, cols, weights = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond_weight(bond)
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([w, w])
    # Self loops
    n = mol.GetNumAtoms()
    for i in range(n):
        rows.append(i)
        cols.append(i)
        weights.append(1.0)

    edge_index = (
        torch.tensor([rows, cols], dtype=torch.long)
        if rows
        else torch.empty(2, 0, dtype=torch.long)
    )
    edge_weight = (
        torch.tensor(weights, dtype=torch.float32)
        if weights
        else torch.empty(0, dtype=torch.float32)
    )

    # 3D conformer is retained for optional position-aware model extensions.
    pos = (
        _generate_conformer(mol) if generate_conformer else torch.zeros((n, 3), dtype=torch.float32)
    )

    # Precompute spectral decomposition for GraphHeat (done once, not every epoch).
    topk = spectral_topk if spectral_topk is not None else _DEFAULT_SPECTRAL_TOPK
    eigvals, eigvecs = _precompute_spectral_for_mol(
        n, edge_index, torch.ones_like(edge_weight) if unweighted_heat else edge_weight, topk=topk
    )

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "pos": pos,
        "n_atoms": n,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


def _precompute_spectral_for_mol(
    n: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute eigendecomposition of symmetric normalized graph Laplacian.

    Called ONCE during data preparation — the graph topology is static.
    This eliminates the O(n³) bottleneck from the training loop.
    """
    from .operators import precompute_spectral

    return precompute_spectral(n, edge_index, edge_weight, normalized=True, topk=topk)


def _generate_conformer(mol: Chem.Mol) -> torch.Tensor:
    """Generate a 3D conformer with ETKDG (heavy atoms only); fall back to zeros."""
    n = mol.GetNumAtoms()
    if n == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    m = Chem.Mol(mol)
    try:
        m = Chem.AddHs(m)
        status = AllChem.EmbedMolecule(m, randomSeed=42, maxAttempts=50)
        if status == 0:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
            conf = m.GetConformer()
            # Return only heavy-atom positions (first n atoms)
            pos = torch.tensor(conf.GetPositions()[:n], dtype=torch.float32)
        else:
            pos = torch.zeros((n, 3), dtype=torch.float32)
    except Exception:
        pos = torch.zeros((n, 3), dtype=torch.float32)
    return pos


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Dataset registry: maps dataset key → (csv_path, smiles_col, target_col).
# ``dielectric`` is retained for a future condition-aware task, but deliberately
# excluded from the default structure-only benchmark because its target varies
# with temperature and repeated molecules would otherwise leak across splits.
DATASETS = {
    "lipophilicity": (
        "datasets/csv/default/logP_lipophilicity.csv",
        "smiles",
        "exp",
    ),
    "bradley_dpg": (
        "datasets/csv/default/melting_point_bradley_double_plus_good.csv",
        "smiles",
        "mpC",
    ),
    "dielectric": (
        "datasets/csv/default/dielectric_liquidbenchmark.csv",
        "smiles",
        "Relative permittivity at zero frequency",
    ),
    "thermosol": (
        "datasets/csv/default/solubility_thermosol.csv",
        "smile",
        "target",
    ),
    "esol": (
        "datasets/csv/backup/logS_esol_delaney.csv",
        "smiles",
        "measured log solubility in mols per litre",
    ),
    "freesolv": (
        "datasets/csv/backup/hydration_free_energy_freesolv_sampl.csv",
        "smiles",
        "expt",
    ),
}

DATASET_METADATA: dict[str, dict[str, Any]] = {
    "lipophilicity": {
        "endpoint": "AstraZeneca lipophilicity (logD at pH 7.4)",
        "unit": "log10 octanol/buffer distribution coefficient (unitless)",
        "primary": True,
        "collapse_replicates": True,
        "source_identifier": "Hop et al., Mol. Pharm. 2018, DOI: 10.1021/acs.molpharmaceut.7b01144",
        "source_url": "https://doi.org/10.1021/acs.molpharmaceut.7b01144",
        "source_note": "AstraZeneca dataset released through ChEMBL; 4,200 examples in the published study.",
    },
    "bradley_dpg": {
        "endpoint": "melting point",
        "unit": "degrees Celsius",
        "primary": True,
        "collapse_replicates": True,
        "source_identifier": "Bradley, Lang, and Williams, Double Plus Good dataset, Figshare 2014, DOI: 10.6084/m9.figshare.1031638",
        "source_url": "https://doi.org/10.6084/m9.figshare.1031638",
        "source_note": "Double Plus Good subset: at least two measurements with a 0.01–5 °C range.",
    },
    "dielectric": {
        "endpoint": "relative permittivity at zero frequency",
        "unit": "unitless",
        "primary": False,
        "collapse_replicates": False,
        "conditional_columns": ["Temperature, K", "Pressure, kPa", "Mass density, kg/m3"],
    },
    "thermosol": {
        "endpoint": "AstraZeneca thermodynamic solubility of solid starting material in pH 7.4 buffer",
        "unit": "log10(solubility / nM); equivalent log10(mol L-1) = target − 9",
        "primary": True,
        "collapse_replicates": True,
        "source_identifier": "ChEMBL assay CHEMBL3301364; Hop et al., Mol. Pharm. 2018, DOI: 10.1021/acs.molpharmaceut.7b01144",
        "source_url": "https://www.ebi.ac.uk/chembl/assay_report_card/CHEMBL3301364/",
        "source_note": "The supplied target is retained as log10(nM); no additional target transformation is applied before training standardization.",
    },
    "esol": {
        "endpoint": "measured aqueous solubility",
        "unit": "log10(mol L-1)",
        "primary": True,
        "collapse_replicates": True,
        "source_identifier": "Delaney, J. Chem. Inf. Comput. Sci. 2004, DOI: 10.1021/ci034243x",
        "source_url": "https://doi.org/10.1021/ci034243x",
        "source_note": "ESOL data set, 1,128 original records.",
    },
    "freesolv": {
        "endpoint": "experimental hydration free energy",
        "unit": "kcal mol-1",
        "primary": True,
        "collapse_replicates": True,
        "source_identifier": "Mobley and Guthrie, J. Comput.-Aided Mol. Des. 2014, DOI: 10.1007/s10822-014-9747-x",
        "source_url": "https://doi.org/10.1007/s10822-014-9747-x",
        "source_note": "Experimental values from the FreeSolv hydration free-energy database.",
    },
}

PRIMARY_DATASETS = tuple(key for key, meta in DATASET_METADATA.items() if bool(meta["primary"]))
CONDITIONAL_DATASETS = tuple(
    key for key, meta in DATASET_METADATA.items() if not bool(meta["primary"])
)


@lru_cache(maxsize=100_000)
def canonical_smiles(smiles: str) -> str | None:
    """Return a canonical isomeric SMILES, or ``None`` for invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    key: str,
    base_dir: str | None = None,
    *,
    return_report: bool = False,
    collapse_replicates: bool | None = None,
    csv_path: str | Path | None = None,
    smiles_column: str = "smiles",
    target_column: str = "target",
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate, canonicalize, and de-duplicate a benchmark dataset.

    Invalid SMILES are removed *before* any split is made.  For ordinary
    structure-only endpoints, repeated canonical molecules are collapsed by
    arithmetic mean of their targets.  The condition-dependent dielectric data
    are intentionally not collapsed by default and are excluded from
    :data:`PRIMARY_DATASETS`.
    """
    if csv_path is None:
        if key not in DATASETS:
            raise KeyError(f"Unknown dataset {key!r}. Available: {sorted(DATASETS)}")
        base_dir = base_dir or os.environ.get("HEATADAPTER_DATA_ROOT", str(Path.cwd()))
        csv_rel_path, smiles_col, target_col = DATASETS[key]
        csv_path = Path(base_dir).expanduser() / csv_rel_path
        metadata = DATASET_METADATA[key]
    else:
        if key in DATASETS:
            raise ValueError("Use a distinct dataset name for custom CSV input.")
        csv_path = Path(csv_path).expanduser().resolve()
        csv_rel_path = str(csv_path)
        smiles_col, target_col = smiles_column, target_column
        metadata = {
            "collapse_replicates": True,
            "endpoint": target_column,
            "unit": "user-supplied target scale",
            "primary": False,
        }
    if not Path(csv_path).is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. See datasets/README.md or pass --csv to the training runner."
        )
    raw = pd.read_csv(csv_path)
    raw_rows = len(raw)
    if smiles_col not in raw or target_col not in raw:
        raise ValueError(
            f"Dataset {key!r} is missing required columns {smiles_col!r} and/or {target_col!r}."
        )

    frame = raw.rename(columns={smiles_col: "SMILES", target_col: "target"})
    frame = frame[["SMILES", "target"]].dropna().copy()
    rows_after_missing = len(frame)
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame = frame.dropna(subset=["target"])
    frame = frame.loc[np.isfinite(frame["target"])].copy()
    rows_after_numeric = len(frame)

    canonical = frame["SMILES"].map(canonical_smiles)
    invalid_smiles_removed = int(canonical.isna().sum())
    frame = frame.loc[canonical.notna()].copy()
    frame["SMILES"] = canonical.loc[canonical.notna()].astype(str).to_numpy()
    rows_after_validation = len(frame)

    if collapse_replicates is None:
        collapse_replicates = bool(metadata["collapse_replicates"])
    rows_before_deduplication = len(frame)
    if collapse_replicates:
        frame = (
            frame.groupby("SMILES", as_index=False, sort=True)["target"]
            .agg(["mean", "size"])
            .reset_index()
            .rename(columns={"mean": "target", "size": "n_measurements"})
        )
    else:
        frame["n_measurements"] = 1
    frame["target"] = frame["target"].astype(float)
    frame["n_measurements"] = frame["n_measurements"].astype(int)
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"Dataset {key!r} contains no valid finite-target molecules after curation."
        )

    report: dict[str, Any] = {
        "dataset": key,
        "source_path": csv_rel_path,
        "source_sha256": _sha256(csv_path),
        "raw_rows": raw_rows,
        "rows_after_missing_value_removal": rows_after_missing,
        "rows_after_numeric_target_validation": rows_after_numeric,
        "invalid_smiles_removed": invalid_smiles_removed,
        "rows_after_smiles_validation": rows_after_validation,
        "collapse_replicates": bool(collapse_replicates),
        "canonical_duplicates_collapsed": rows_before_deduplication - len(frame),
        "final_rows": len(frame),
        "unique_canonical_molecules": int(frame["SMILES"].nunique()),
        "endpoint": metadata["endpoint"],
        "unit": metadata["unit"],
        "primary_benchmark": bool(metadata["primary"]),
    }
    if "conditional_columns" in metadata:
        report["conditional_columns"] = metadata["conditional_columns"]
    frame.attrs["curation_report"] = report
    return (frame, report) if return_report else frame


# ---------------------------------------------------------------------------
# Leakage-controlled molecule and scaffold splits
# ---------------------------------------------------------------------------


@lru_cache(maxsize=100_000)
def _scaffold(smiles: str) -> str:
    """Return the canonical Bemis–Murcko scaffold, or ``""`` for acyclic molecules.

    This function deliberately uses :mod:`rdkit.Chem.Scaffolds.MurckoScaffold`.
    The previous implementation accessed a nonexistent ``Chem.Scaffolds``
    attribute and silently returned an empty scaffold for every molecule.
    """
    canonical = canonical_smiles(smiles)
    if canonical is None:
        return ""
    mol = Chem.MolFromSmiles(canonical)
    assert mol is not None  # guarded by canonical_smiles above
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        return ""
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    return Chem.MolToSmiles(scaffold_mol, canonical=True) if scaffold_mol is not None else scaffold


def _scaffold_group_key(smiles: str) -> str:
    """Group cyclic molecules by Murcko scaffold and acyclic ones by molecule.

    Treating every acyclic molecule as the same empty Murcko scaffold can put
    half a benchmark into one split.  Grouping acyclic entries by canonical
    molecule preserves the non-leakage guarantee while retaining usable split
    sizes.  Manuscripts must report this convention explicitly.
    """
    canonical = canonical_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Cannot build a split key for invalid SMILES: {smiles!r}")
    scaffold = _scaffold(canonical)
    return f"scaffold::{scaffold}" if scaffold else f"acyclic::{canonical}"


def _validate_fractions(frac_train: float, frac_val: float) -> tuple[float, float, float]:
    frac_test = 1.0 - frac_train - frac_val
    if frac_train <= 0 or frac_val <= 0 or frac_test <= 0:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    return frac_train, frac_val, frac_test


def _split_by_groups(
    df: pd.DataFrame,
    group_keys: list[str],
    *,
    frac_train: float,
    frac_val: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign whole groups to 80/10/10-like partitions with deterministic balancing."""
    if len(df) != len(group_keys):
        raise ValueError("group_keys must contain one key per dataframe row.")
    if len(df) < 3:
        raise ValueError("At least three rows are required for train/validation/test splitting.")

    fractions = np.asarray(_validate_fractions(frac_train, frac_val), dtype=float)
    row_positions: dict[str, list[int]] = defaultdict(list)
    for position, group_key in enumerate(group_keys):
        row_positions[group_key].append(position)
    groups = list(row_positions.items())
    if len(groups) < 3:
        raise ValueError(
            "At least three molecule/scaffold groups are required; refusing to split a group across sets."
        )

    rng = np.random.RandomState(seed)
    # Randomise ties only, then preserve descending group size for a stable
    # size-aware greedy assignment.
    groups = [groups[i] for i in rng.permutation(len(groups))]
    groups.sort(key=lambda item: -len(item[1]))

    target = fractions * len(df)
    counts = np.zeros(3, dtype=int)
    assignments: list[list[int]] = [[], [], []]
    for group_index, (_, positions) in enumerate(groups):
        group_size = len(positions)
        remaining_groups = len(groups) - group_index
        empty_splits = [split for split, count in enumerate(counts) if count == 0]
        # Reserve one group for each still-empty split when needed.
        candidates = empty_splits if remaining_groups == len(empty_splits) else [0, 1, 2]

        scores: list[tuple[float, float, int]] = []
        for split in candidates:
            trial = counts.copy()
            trial[split] += group_size
            relative_error = (trial - target) / np.maximum(target, 1.0)
            # A secondary term favors the currently most underfilled split
            # when two assignments have the same global balance cost.
            score = float(np.square(relative_error).sum())
            underfill = float((counts[split] - target[split]) / max(target[split], 1.0))
            scores.append((score, underfill, split))
        _, _, selected = min(scores)
        assignments[selected].extend(positions)
        counts[selected] += group_size

    splits = [df.iloc[sorted(positions)].reset_index(drop=True).copy() for positions in assignments]
    if any(split.empty for split in splits):
        raise RuntimeError("Balanced group splitter created an empty partition.")
    validate_split_integrity(*splits)
    return splits[0], splits[1], splits[2]


def validate_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Assert that canonical molecules cannot occur in more than one split."""
    molecule_sets = [
        set(frame["SMILES"].map(canonical_smiles)) for frame in (train_df, val_df, test_df)
    ]
    overlaps = {
        "train_val": len(molecule_sets[0] & molecule_sets[1]),
        "train_test": len(molecule_sets[0] & molecule_sets[2]),
        "val_test": len(molecule_sets[1] & molecule_sets[2]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Molecule leakage detected across splits: {overlaps}")
    return overlaps


def validate_scaffold_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Assert that nonempty Murcko scaffolds cannot occur in more than one split."""
    overlaps = scaffold_overlap_counts(train_df, val_df, test_df)
    if any(overlaps.values()):
        raise AssertionError(f"Murcko scaffold leakage detected across splits: {overlaps}")
    return overlaps


def scaffold_overlap_counts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Count nonempty Murcko scaffolds shared across splits without asserting.

    A molecule-disjoint random split deliberately permits this kind of scaffold
    overlap.  The count is therefore a diagnostic for random splits and a
    leakage assertion only for structure-disjoint scaffold splits.
    """
    scaffold_sets = [
        {_scaffold(smiles) for smiles in frame["SMILES"] if _scaffold(smiles)}
        for frame in (train_df, val_df, test_df)
    ]
    overlaps = {
        "train_val": len(scaffold_sets[0] & scaffold_sets[1]),
        "train_test": len(scaffold_sets[0] & scaffold_sets[2]),
        "val_test": len(scaffold_sets[1] & scaffold_sets[2]),
    }
    return overlaps


def split_diagnostics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Return JSON-serialisable split sizes and leakage diagnostics."""
    frames = (train_df, val_df, test_df)
    return {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "unique_train": int(train_df["SMILES"].nunique()),
        "unique_val": int(val_df["SMILES"].nunique()),
        "unique_test": int(test_df["SMILES"].nunique()),
        "molecule_overlap": validate_split_integrity(*frames),
        "murcko_scaffold_overlap": scaffold_overlap_counts(*frames),
        "murcko_scaffolds_train": len({_scaffold(s) for s in train_df["SMILES"] if _scaffold(s)}),
        "murcko_scaffolds_val": len({_scaffold(s) for s in val_df["SMILES"] if _scaffold(s)}),
        "murcko_scaffolds_test": len({_scaffold(s) for s in test_df["SMILES"] if _scaffold(s)}),
    }


def scaffold_split(
    df: pd.DataFrame,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a leakage-controlled Murcko-scaffold split.

    Cyclic molecules sharing a Bemis–Murcko scaffold always stay together.
    Acyclic molecules are grouped by canonical molecule rather than one giant
    empty scaffold group; this convention is reported as
    ``Murcko scaffold split with molecule grouping for acyclic compounds``.
    """
    keys = [_scaffold_group_key(smiles) for smiles in df["SMILES"]]
    train_df, val_df, test_df = _split_by_groups(
        df, keys, frac_train=frac_train, frac_val=frac_val, seed=seed
    )
    validate_scaffold_integrity(train_df, val_df, test_df)
    return train_df, val_df, test_df


def scaffold_kfold(
    df: pd.DataFrame,
    fold: int,
    *,
    n_splits: int = 5,
    seed: int = 20260930,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffled whole-scaffold folds with disjoint tests covering every molecule.

    Randomly ordered groups go to the currently smallest fold. Validation is
    selected from the remaining groups (target 1/8 of the outer training rows),
    giving approximately 70/10/20 train/validation/test with five outer folds.
    Neither assignment uses targets. This is a new protocol, not legacy v2.
    """
    if n_splits < 3 or not 0 <= fold < n_splits:
        raise ValueError("Require n_splits >= 3 and 0 <= fold < n_splits.")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smiles in enumerate(df["SMILES"]):
        groups[_scaffold_group_key(smiles)].append(i)
    if len(groups) < n_splits + 1:
        raise ValueError("Too few scaffold groups for outer folds and validation.")
    rng = np.random.RandomState(seed)
    keys = sorted(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    bins: list[list[str]] = [[] for _ in range(n_splits)]
    counts = np.zeros(n_splits, dtype=int)
    for key in keys:
        chosen = int(rng.choice(np.flatnonzero(counts == counts.min())))
        bins[chosen].append(key)
        counts[chosen] += len(groups[key])
    remaining = [key for f, bucket in enumerate(bins) if f != fold for key in bucket]
    rng = np.random.RandomState(seed + 1009 * (fold + 1))
    remaining = [remaining[i] for i in rng.permutation(len(remaining))]
    val_keys, train_keys = [], []
    target = sum(len(groups[key]) for key in remaining) / 8
    count = 0
    for i, key in enumerate(remaining):
        if i < len(remaining) - 1 and (
            not val_keys or abs(count + len(groups[key]) - target) < abs(count - target)
        ):
            val_keys.append(key)
            count += len(groups[key])
        else:
            train_keys.append(key)

    def frame(selected):
        return (
            df.iloc[sorted(i for key in selected for i in groups[key])]
            .reset_index(drop=True)
            .copy()
        )

    splits = frame(train_keys), frame(val_keys), frame(bins[fold])
    if any(part.empty for part in splits):
        raise ValueError("Scaffold folds produced an empty partition.")
    validate_split_integrity(*splits)
    validate_scaffold_integrity(*splits)
    return splits


def random_split(
    df: pd.DataFrame,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a random *molecule-disjoint* split for in-domain comparison.

    The group assignment is random, but duplicate canonical molecules are never
    split across training, validation, and test sets.
    """
    keys = [f"molecule::{canonical_smiles(smiles)}" for smiles in df["SMILES"]]
    return _split_by_groups(df, keys, frac_train=frac_train, frac_val=frac_val, seed=seed)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate_mols(
    samples: list[dict[str, torch.Tensor]], targets: list[float]
) -> dict[str, torch.Tensor]:
    """Collate a list of per-molecule graph dicts into a batched dict."""
    x_list, ei_list, ew_list, pos_list = [], [], [], []
    eigvals_list, eigvecs_list = [], []
    batch_list = []
    y_list = []

    offset = 0
    for i, (g, t) in enumerate(zip(samples, targets)):
        n = g["n_atoms"]
        x_list.append(g["x"])
        pos_list.append(g["pos"])
        if g["edge_index"].numel():
            ei_list.append(g["edge_index"] + offset)
            ew_list.append(g["edge_weight"])
        batch_list.append(torch.full((n,), i, dtype=torch.long))
        y_list.append(torch.full((n, 1), t, dtype=torch.float32))
        # Pass through precomputed spectral data for the default heat operator.
        eigvals_list.append(g.get("eigvals", torch.empty(0)))
        eigvecs_list.append(g.get("eigvecs", torch.empty(0, 0)))
        offset += n

    x = torch.cat(x_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    batch = torch.cat(batch_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(ei_list, dim=1) if ei_list else torch.empty(2, 0, dtype=torch.long)
    edge_weight = torch.cat(ew_list, dim=0) if ew_list else torch.empty(0, dtype=torch.float32)

    return {
        "x": x,
        "pos": pos,
        "batch": batch,
        "y": y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "eigvals_list": eigvals_list,
        "eigvecs_list": eigvecs_list,
    }
