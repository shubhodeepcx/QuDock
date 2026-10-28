# Command line

Run `qudock --help`, or `qudock <command> --help` for options.

| Command | What it does |
| :--- | :--- |
| `data` | Download a dataset and show scaffold and random split sizes |
| `benchmark` | Learning curves for QSVC against classical models |
| `sweep` | ROC-AUC and kernel geometry from 2 to 10 qubits |
| `noise` | QSVC with shot-sampled kernels on a fake IBM backend |
| `hardware` | Measure a small kernel on an IBM Quantum device |
| `final` | Final-round study: PLS, tuned bandwidth, projected, hybrid, gradient boosting |
| `scaling` | Kernel concentration and accuracy from 4 to 14 qubits |
| `trainable` | Train the feature map's own parameters by alignment |
| `confidence` | Conformal prediction sets and accuracy by similarity |
| `explain` | Which atoms drove each prediction, and a check that they did |
| `graph` | Pharmacophore-graph feature map against classical twins |
| `bench` | QuDock-Bench leaderboard |
| `screen` | Rank a library by activity, brain penetration and drug-likeness |
| `serve` | HTTP API |
| `scorecard` | One chart of every setting measured, from the result files |
| `hardware-svm` | Projected-kernel SVM trained and tested on a device (`--mitigation none, readout or zne`) |
| `reproduce` | Rerun every study that makes a table or figure |
