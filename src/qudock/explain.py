"""Which atoms made the model say inhibitor?

A kernel SVM has no coefficients to read: the decision value is a sum over
training molecules, not over features. So instead of opening the model we
ask it questions. Each fingerprint bit a molecule sets stands for one
substructure; we switch that bit off, push the molecule through the same
compression and the same kernel, and record how far the score moves. A bit
whose removal drops the score was holding the prediction up.

Every bit comes with the atoms whose environment set it, so each bit's
contribution is spread over those atoms and the molecule can be drawn with
the responsible atoms highlighted: green for atoms arguing inhibitor, purple
for atoms arguing against.

This is occlusion, the same idea as leave-one-feature-out, and it is honest
about the model that is actually running: the quantum kernel, the PCA or PLS
compression and the SVM all stay in the loop, and nothing is approximated.
One decision value per set bit is needed, so a molecule costs roughly fifty
extra kernel evaluations, which is a fraction of a second at six qubits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

POSITIVE = (0.10, 0.60, 0.33)      # green: pushes towards inhibitor
NEGATIVE = (0.48, 0.25, 0.65)      # purple: pushes away


def environment_atoms(mol, centre: int, radius: int) -> set[int]:
    """The atoms inside the circular environment of ``radius`` around ``centre``."""
    if radius == 0:
        return {centre}
    atoms = {centre}
    for b in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre):
        bond = mol.GetBondWithIdx(b)
        atoms |= {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
    return atoms


def bit_atoms(mol, radius: int = 2, n_bits: int = 2048) -> dict[int, set[int]]:
    """For every fingerprint bit this molecule sets, the atoms that set it."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    extra = rdFingerprintGenerator.AdditionalOutput()
    extra.AllocateBitInfoMap()
    gen.GetFingerprintAsNumPy(mol, additionalOutput=extra)
    out = {}
    for bit, envs in extra.GetBitInfoMap().items():
        atoms: set[int] = set()
        for centre, rad in envs:
            atoms |= environment_atoms(mol, centre, rad)
        out[int(bit)] = atoms
    return out


def substructure_smiles(mol, atoms) -> str:
    """A readable name for the fragment those atoms form."""
    atoms = sorted(atoms)
    if len(atoms) == 1:
        return mol.GetAtomWithIdx(atoms[0]).GetSymbol()
    bonds = [b.GetIdx() for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    return Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, bondsToUse=bonds)


def angles_from_raw(pipe, raw: np.ndarray) -> np.ndarray:
    """Run already-built raw features through a fitted pipeline's compression."""
    reduced = pipe.pca_.transform(pipe.scaler_.transform(raw))
    return np.clip(pipe.angles_.transform(reduced), 0, np.pi)


@dataclass
class Explanation:
    """Per-atom and per-substructure attributions for one molecule."""
    smiles: str
    score: float
    atom_weights: np.ndarray            # one signed weight per atom
    bits: list[dict]                    # bit, attribution, atoms, substructure

    def top_substructures(self, k: int = 5) -> list[dict]:
        return sorted(self.bits, key=lambda r: -abs(r["attribution"]))[:k]

    def table(self, k: int = 5):
        import pandas as pd

        rows = [{"Substructure": r["substructure"], "Atoms": len(r["atoms"]),
                 "Pushes towards": "inhibitor" if r["attribution"] > 0 else "not an inhibitor",
                 "Score change if removed": round(-r["attribution"], 3)}
                for r in self.top_substructures(k)]
        return pd.DataFrame(rows)


def explain_molecule(smiles: str, pipe, model, fingerprints: bool = False) -> Explanation:
    """Attribute a model's score for one molecule to its atoms by occlusion.

    ``model`` is anything with ``decision_function``; set ``fingerprints`` when it
    expects angles and a fingerprint joined by :func:`qudock.kernels.stack`,
    as the final-round models do, so the occluded bit is removed from both.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    atoms_of = bit_atoms(mol, pipe.radius, pipe.n_bits)
    raw = pipe._raw([smiles])
    rows = np.repeat(raw, 1 + len(atoms_of), axis=0)
    bits = sorted(atoms_of)
    for i, bit in enumerate(bits, start=1):
        rows[i, bit] = 0.0              # the molecule as if that substructure were absent
    X = angles_from_raw(pipe, rows)
    if fingerprints:
        X = np.hstack([X, rows[:, :pipe.n_bits]])
    scores = np.asarray(model.decision_function(X), dtype=float).ravel()

    weights = np.zeros(mol.GetNumAtoms())
    table = []
    for bit, score in zip(bits, scores[1:]):
        attribution = float(scores[0] - score)      # positive: the bit held the score up
        table.append({"bit": bit, "attribution": attribution, "atoms": atoms_of[bit],
                      "substructure": substructure_smiles(mol, atoms_of[bit])})
        for atom in atoms_of[bit]:
            weights[atom] += attribution / len(atoms_of[bit])
    return Explanation(smiles, float(scores[0]), weights, table)


def draw_explanation(explanation: Explanation, size=(420, 320), floor: float = 0.1) -> bytes:
    """PNG of the molecule with the responsible atoms highlighted."""
    mol = Chem.MolFromSmiles(explanation.smiles)
    weights = explanation.atom_weights
    peak = float(np.abs(weights).max()) if len(weights) else 0.0
    colours, radii = {}, {}
    for atom, weight in enumerate(weights):
        share = abs(weight) / peak if peak else 0.0
        if share < floor:
            continue
        base = POSITIVE if weight > 0 else NEGATIVE
        # blend towards white, so weak atoms stay readable
        colours[atom] = tuple(1 - (1 - c) * (0.25 + 0.75 * share) for c in base)
        radii[atom] = 0.25 + 0.3 * share
    canvas = rdMolDraw2D.MolDraw2DCairo(*size)
    canvas.drawOptions().fillHighlights = True
    rdMolDraw2D.PrepareAndDrawMolecule(canvas, mol, highlightAtoms=list(colours),
                                       highlightAtomColors=colours, highlightAtomRadii=radii)
    canvas.FinishDrawing()
    return canvas.GetDrawingText()


def run_explain_study(n_qubits: int = 6, n_train: int = 400, n_molecules: int = 12,
                      seed: int = 0, df=None):
    """Explain the test molecules the final model is most sure about, and check the atoms matter.

    For each molecule the atoms are ranked by attribution and the top fifth are
    removed from the fingerprint together. If the explanation points at the
    right atoms, the score falls much further than removing a random fifth, and
    the gap between the two is the number worth reporting.
    """
    import pandas as pd

    from qudock.data import load_bace, scaffold_split
    from qudock.features import FeaturePipeline, morgan_fingerprints
    from qudock.kernels import KernelSVM, stack, tune_bandwidth

    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    rng = np.random.default_rng(seed)
    keep = rng.choice(split.train, size=min(n_train, len(split.train)), replace=False)
    pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
    A_tr = pipe.fit_transform(smiles[keep], y[keep])
    X_tr = stack(A_tr, morgan_fingerprints(smiles[keep]))
    bandwidth = tune_bandwidth(A_tr, y[keep], n_qubits, seed=seed)
    model = KernelSVM(kind="hybrid", n_angles=n_qubits, C=1, weight=0.25,
                      bandwidth=bandwidth).fit(X_tr, y[keep])

    te = split.test
    A_te = pipe.transform(smiles[te])
    scores = model.decision_function(stack(A_te, morgan_fingerprints(smiles[te])))
    order = np.argsort(-np.abs(scores))[:n_molecules]

    rows = []
    for i in order:
        smi = smiles[te[i]]
        exp = explain_molecule(smi, pipe, model, fingerprints=True)
        mol = Chem.MolFromSmiles(smi)
        n_drop = max(1, mol.GetNumAtoms() // 5)
        top = set(np.argsort(-exp.atom_weights * np.sign(exp.score))[:n_drop])
        random = set(rng.choice(mol.GetNumAtoms(), n_drop, replace=False))
        atoms_of = bit_atoms(mol, pipe.radius, pipe.n_bits)
        drops = {}
        for name, atoms in [("top_atoms", top), ("random_atoms", random)]:
            raw = pipe._raw([smi])
            hit = [bit for bit, bit_set in atoms_of.items() if bit_set & atoms]
            raw[0, hit] = 0.0
            X = np.hstack([angles_from_raw(pipe, raw), raw[:, :pipe.n_bits]])
            drops[name] = float(np.sign(exp.score) *
                                (exp.score - model.decision_function(X)[0]))
            # how many bits each removal broke, so a control that broke more is visible
            drops[f"bits_{name}"] = len(hit)
        rows.append({"smiles": smi, "label": int(y[te[i]]), "score": float(exp.score),
                     "n_atoms": mol.GetNumAtoms(), "n_removed": n_drop,
                     "top_substructure": exp.top_substructures(1)[0]["substructure"], **drops})
    out = pd.DataFrame(rows)
    out["gap"] = out["top_atoms"] - out["random_atoms"]
    return out


def summarise_explain(results) -> dict[str, float]:
    """How often the highlighted atoms matter more than the same number of random atoms."""
    return {"n_molecules": len(results),
            "mean_drop_top": results["top_atoms"].mean(),
            "mean_drop_random": results["random_atoms"].mean(),
            "mean_gap": results["gap"].mean(),
            "share_explanation_wins": float((results["gap"] > 0).mean()),
            "mean_bits_top": results["bits_top_atoms"].mean(),
            "mean_bits_random": results["bits_random_atoms"].mean()}
