"""Turn SMILES into qubit-ready feature vectors.

Morgan fingerprint (optionally plus RDKit descriptors) -> standardise ->
compress to ``n_qubits`` components -> rescale each to [0, pi], the angle
range of the ZZFeatureMap. Compression is PCA, or PLS, which uses the labels
to keep the directions that separate inhibitors. Everything is fit on
training molecules only.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def morgan_fingerprints(smiles, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP fingerprints (radius 2 = ECFP4) as a dense 0/1 matrix."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"could not parse SMILES: {s}")
        fps[i] = gen.GetFingerprintAsNumPy(mol)
    return fps


# physicochemical properties that matter for BACE-1 inhibitors and brain-penetrant drugs
DESCRIPTORS = {
    "mol_weight": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "h_donors": Descriptors.NumHDonors,
    "h_acceptors": Descriptors.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "fraction_sp3": Descriptors.FractionCSP3,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "rings": Descriptors.RingCount,
}


def descriptors(smiles) -> np.ndarray:
    """RDKit descriptors in ``DESCRIPTORS`` as a float matrix."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for s, m in zip(smiles, mols):
        if m is None:
            raise ValueError(f"could not parse SMILES: {s}")
    return np.array([[f(m) for f in DESCRIPTORS.values()] for m in mols], dtype=np.float64)


METHODS = ("pca", "pls")


class FeaturePipeline:
    """Fingerprint (+ descriptors) -> scale -> PCA or PLS -> [0, pi] angles."""

    def __init__(self, n_qubits: int = 4, radius: int = 2, n_bits: int = 2048, seed: int = 0,
                 method: str = "pca", use_descriptors: bool = False):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be at least 1, got {n_qubits}")
        self.n_qubits = n_qubits
        self.radius = radius
        self.n_bits = n_bits
        self.seed = seed
        self.method = method
        self.use_descriptors = use_descriptors

    def _raw(self, smiles) -> np.ndarray:
        fps = morgan_fingerprints(smiles, self.radius, self.n_bits)
        return np.hstack([fps, descriptors(smiles)]) if self.use_descriptors else fps

    def fit(self, smiles, y=None) -> FeaturePipeline:
        raw = self._raw(smiles)
        self.scaler_ = StandardScaler().fit(raw)
        scaled = self.scaler_.transform(raw)
        if self.method == "pls":
            if y is None:
                raise ValueError("PLS compression needs the training labels")
            self.pca_ = PLSRegression(n_components=self.n_qubits, scale=False).fit(scaled, y)
        else:
            self.pca_ = PCA(n_components=self.n_qubits, random_state=self.seed).fit(scaled)
        self.angles_ = MinMaxScaler(feature_range=(0, np.pi)).fit(self.pca_.transform(scaled))
        return self

    def transform(self, smiles) -> np.ndarray:
        reduced = self.pca_.transform(self.scaler_.transform(self._raw(smiles)))
        # test molecules can land outside the training range
        return np.clip(self.angles_.transform(reduced), 0, np.pi)

    def fit_transform(self, smiles, y=None) -> np.ndarray:
        return self.fit(smiles, y).transform(smiles)

    @property
    def explained_variance(self) -> float:
        """Fraction of fingerprint variance kept by the PCA components."""
        if self.method != "pca":
            raise AttributeError("explained_variance is only defined for PCA compression")
        return float(self.pca_.explained_variance_ratio_.sum())
