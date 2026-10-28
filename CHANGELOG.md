# Changelog

## Unreleased

### New
- `qudock screen`: ranks a library by BACE-1 activity (hybrid quantum + Tanimoto kernel with
  conformal p-values), blood-brain barrier penetration (BBBP model), QED and PAINS alerts, with
  an applicability-domain flag. Without a library it ranks the BACE test set and reports how
  each ranking enriches known actives.
- `qudock bench` (QuDock-Bench): every model on the same tasks (BACE, BBBP, ClinTox), scaffold
  splits, training subsamples and seeds, with a leaderboard. Any model can be plugged in as
  `module.path:function`.
- `qudock graph`: a feature map built from the pharmacophore graph, where each qubit is a
  pharmacophore family and entanglement follows how close the groups sit in the molecule,
  compared with classical models on the same 21 numbers.
- `qudock serve`: HTTP API around the screener (`pip install .[api]`).
- Nystroem approximation for quantum kernels (`NystroemSVM`): n x m kernel entries instead of
  n^2 / 2, exact when every training molecule is a landmark.
- Classical-shadow estimates of one- and two-qubit Pauli expectations (`qudock.shadows`).
- Gradient boosting on fingerprint + descriptors as a classical baseline in the final study.
- `hardware-svm --mitigation none|readout|zne`.
- `provenance.json` next to every result: command, git commit, package versions, data hashes.
- Web app: Alzheimer's ranking toggle on the library screen.
- ClinTox dataset loader; `qudock data --dataset`; `qudock --version`.
- Documentation site (mkdocs), citation file, PyPI metadata and a tag-triggered release
  workflow, Makefile, pre-commit hooks, dependabot.

### Faster
- The linear ZZ feature map is simulated in batch with numpy: over 40x faster than one Qiskit
  statevector per molecule, checked against Qiskit to machine precision.
- Morgan fingerprints and descriptors are cached per SMILES, so the many refits in each study
  reuse them.
- The trainable kernel applies each layer's RZ and ZZ phases in one step.

### Fixed
- `qudock reproduce` no longer crashes when saved device data for a reanalysis is missing.
- The hardware time cap now applies even with error mitigation switched off.
- Interrupted dataset downloads no longer leave a half-written file in the cache.
- Clear errors for bad split fractions, unknown model names, malformed list options,
  bootstrap resamples with one class, and all-zero kernel matrices.
