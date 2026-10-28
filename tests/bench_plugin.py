"""A user-written model, loaded as ``bench_plugin:nearest`` when tests/ is on the path."""

import numpy as np

from qudock.features import morgan_fingerprints
from qudock.kernels import tanimoto_kernel


class Nearest:
    """Score = similarity to the closest active minus to the closest inactive."""

    def fit(self, smiles, y):
        self.fp_, self.y_ = morgan_fingerprints(smiles), np.asarray(y)
        return self

    def decision_function(self, smiles):
        sim = tanimoto_kernel(morgan_fingerprints(smiles), self.fp_)
        return sim[:, self.y_ == 1].max(1) - sim[:, self.y_ == 0].max(1)


def nearest(info):
    return Nearest()
