"""Turn SMILES into qubit-ready feature vectors.

Morgan fingerprint -> standardise -> PCA to ``n_qubits`` components ->
rescale each component to [0, pi], the angle range of the ZZFeatureMap.
Everything is fit on training molecules only.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
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


class FeaturePipeline:
    """Fingerprint -> scale -> PCA -> [0, pi] angles."""

    def __init__(self, n_qubits: int = 4, radius: int = 2, n_bits: int = 2048, seed: int = 0):
        self.n_qubits = n_qubits
        self.radius = radius
        self.n_bits = n_bits
        self.seed = seed

    def fit(self, smiles) -> FeaturePipeline:
        fps = morgan_fingerprints(smiles, self.radius, self.n_bits)
        self.scaler_ = StandardScaler().fit(fps)
        self.pca_ = PCA(n_components=self.n_qubits, random_state=self.seed)
        reduced = self.pca_.fit_transform(self.scaler_.transform(fps))
        self.angles_ = MinMaxScaler(feature_range=(0, np.pi)).fit(reduced)
        return self

    def transform(self, smiles) -> np.ndarray:
        reduced = self.pca_.transform(self.scaler_.transform(
            morgan_fingerprints(smiles, self.radius, self.n_bits)))
        # test molecules can land outside the training range
        return np.clip(self.angles_.transform(reduced), 0, np.pi)

    def fit_transform(self, smiles) -> np.ndarray:
        return self.fit(smiles).transform(smiles)

    @property
    def explained_variance(self) -> float:
        """Fraction of fingerprint variance kept by the PCA components."""
        return float(self.pca_.explained_variance_ratio_.sum())
