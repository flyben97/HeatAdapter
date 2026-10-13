"""Generate demonstration targets from molecular mass; these are not experimental data."""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def main():
    smiles = ["C" * length + end for length in range(1, 9) for end in ("", "O", "N", "Cl", "Br")]
    smiles += [
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "C1CC1",
        "OC1CC1",
        "C1CCC1",
        "C1CCCC1",
        "c1ccoc1",
        "c1ccsc1",
        "C1COCCN1",
        "C1CCNCC1",
    ]
    records = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        canonical = Chem.MolToSmiles(mol)
        records[canonical] = round(Descriptors.MolWt(mol), 6)
    output = Path(__file__).with_name("toy_molecules.csv")
    pd.DataFrame(sorted(records.items()), columns=["smiles", "target"]).to_csv(output, index=False)
    print(f"Wrote {len(records)} synthetic examples to {output.name}")


if __name__ == "__main__":
    main()
