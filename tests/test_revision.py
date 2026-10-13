"""Scientific regression tests: equivariance, numerical equivalence, disjoint OOF."""

import unittest
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from graphheat.mol_data import mol_to_graph, collate_mols, load_dataset, scaffold_kfold
from graphheat.models_pyg import GCNHeatNet, GINEHeatNet
from graphheat.operators import GraphHeatOperator, normalized_laplacian


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_permutation_invariance(self):
        rng = np.random.RandomState(20260930)
        smiles = [
            "C=CCN(C(=O)Cc1ccc(S(C)(=O)=O)cc1)C1CCN(CCC(c2ccccc2)c2ccccc2)CC1",
            "CC(=O)N(C)c1ccc(Nc2ncc3cc(-c4ccncc4)ccc3n2)cc1",
            "c1ccccc1.CC",
            "C",
        ]
        for topk in (0, 20):
            for cls in (GCNHeatNet, GINEHeatNet):
                torch.manual_seed(177)
                model = cls(in_dim=46, hidden=32, depth=3).eval()
                for smi in smiles:
                    mol = Chem.MolFromSmiles(smi)
                    for _ in range(5):
                        other = Chem.RenumberAtoms(mol, rng.permutation(mol.GetNumAtoms()).tolist())
                        graphs = [mol_to_graph(m, spectral_topk=topk) for m in (mol, other)]
                        with torch.no_grad():
                            preds = []
                            for g in graphs:
                                preds.append(model(collate_mols([g], [0.0])).mean().item())
                        self.assertLess(abs(preds[0] - preds[1]), 1e-5, (smi, topk, cls))

    def test_heat_equals_dense_matrix_exponential(self):
        graphs = [mol_to_graph(Chem.MolFromSmiles(s)) for s in ["CCO", "c1ccccc1", "C", "CCCC"]]
        data = collate_mols(graphs, [0.0] * len(graphs))
        op = GraphHeatOperator(46).eval()
        x = data["x"].requires_grad_()
        actual = op(
            x,
            data["edge_index"],
            data["batch"],
            data["edge_weight"],
            data["eigvals_list"],
            data["eigvecs_list"],
        )
        outputs = []
        for g in graphs:
            L = normalized_laplacian(
                g["n_atoms"], g["edge_index"], g["edge_weight"], torch.device("cpu"), torch.float64
            )
            K = sum(
                w * torch.matrix_exp(-t.double() * L)
                for t, w in zip(op.taus, torch.softmax(op.kernel_logits, 0))
            )
            outputs.append(K.float() @ g["x"])
        expected = op.norm(op.proj(torch.cat(outputs)) + x)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(op.log_taus.grad).all())

    def test_scaffold_oof_partition(self):
        df = load_dataset(
            "toy", csv_path=Path(__file__).resolve().parents[1] / "examples/toy_molecules.csv"
        )
        tests = [set(scaffold_kfold(df, f)[2].SMILES) for f in range(5)]
        self.assertEqual(set.union(*tests), set(df.SMILES))
        for i in range(5):
            for j in range(i):
                self.assertFalse(tests[i] & tests[j])
        self.assertEqual(tests[0], set(scaffold_kfold(df, 0)[2].SMILES))
        self.assertNotEqual(tests[0], set(scaffold_kfold(df, 0, seed=20260931)[2].SMILES))

    def test_mixed_rank_batch_and_eager_controls(self):
        # Same node count, different boundary multiplicity: rank 3 vs rank 2.
        graphs = [
            mol_to_graph(Chem.MolFromSmiles(s), spectral_topk=2) for s in ("c1ccccc1", "CCCCCC")
        ]
        self.assertNotEqual(graphs[0]["eigvals"].numel(), graphs[1]["eigvals"].numel())
        op = GraphHeatOperator(46).eval()
        data = collate_mols(graphs, [0.0, 0.0])
        batched = op(
            data["x"],
            data["edge_index"],
            data["batch"],
            data["edge_weight"],
            data["eigvals_list"],
            data["eigvecs_list"],
        )
        separate = []
        for g in graphs:
            d = collate_mols([g], [0.0])
            separate.append(
                op(
                    d["x"],
                    d["edge_index"],
                    d["batch"],
                    d["edge_weight"],
                    d["eigvals_list"],
                    d["eigvecs_list"],
                )
            )
        torch.testing.assert_close(batched, torch.cat(separate), atol=2e-6, rtol=2e-5)
        full = [
            mol_to_graph(Chem.MolFromSmiles(s), spectral_topk=0) for s in ("CCO", "c1ccccc1.CC")
        ]
        d = collate_mols(full, [0.0, 0.0])
        for mode in ("heat", "identity", "zero"):
            op.kernel_mode = mode
            cached = op(
                d["x"],
                d["edge_index"],
                d["batch"],
                d["edge_weight"],
                d["eigvals_list"],
                d["eigvecs_list"],
            )
            eager = op(d["x"], d["edge_index"], d["batch"], d["edge_weight"])
            torch.testing.assert_close(cached, eager, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
