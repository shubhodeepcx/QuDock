# Reproducibility

Every command that writes results also writes `provenance.json` into the same
folder:

* the exact command and every option
* the git commit, and whether the working tree had uncommitted changes
* the Python version and the versions of Qiskit, RDKit, scikit-learn and friends
* a SHA-256 of each raw dataset file in the cache
* when it ran

Two folders with the same commit, options and data hashes should hold the same
tables; if they don't, the package versions are the next place to look.
`qudock reproduce` reruns every study that makes a table or figure, from saved
device data where a real quantum computer was involved.

Splits are deterministic (scaffold split), training subsamples are stratified and
seeded, and every model in a comparison sees the same molecules.
