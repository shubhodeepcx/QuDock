# QuDock-Bench

Numbers from different quantum ML papers rarely sit in one table: different
datasets, splits, training sizes, metrics. QuDock-Bench fixes all of them.

* **Tasks:** `bace` (BACE-1 inhibition), `bbbp` (blood-brain barrier
  penetration), `clintox` (failed trials for toxicity, 6% positive)
* **Split:** Bemis-Murcko scaffold split, 80/20, rarest scaffolds in the test set
* **Training sizes and seeds:** stratified subsamples, identical for every model
* **Metrics:** ROC-AUC, PR-AUC, enrichment at 1% and 5%, BEDROC
* **Leaderboard:** mean ROC-AUC with a 95% interval per task, and the mean rank across tasks

```bash
qudock bench                                   # every built-in model, all tasks, 3 seeds
qudock bench --models rf_fingerprint,qsvc_pls --tasks bace --sizes 100,full
```

## Built-in models

| Name | Kind |
| :--- | :--- |
| `rf_fingerprint` | Random Forest on ECFP4 |
| `gb_fingerprint` | Histogram gradient boosting on ECFP4 + RDKit descriptors |
| `tanimoto_svm` | SVM on the Tanimoto kernel |
| `qsvc_pls` | ZZ fidelity kernel on 6 PLS angles |
| `hybrid_pls` | Weighted ZZ + Tanimoto kernel |
| `nystroem_projected` | Projected quantum kernel through a 150-landmark Nystroem map |
| `graph_quantum` | Pharmacophore-graph feature map |

## Add your own model

Write a function that receives a `TaskInfo` (task name, training size, seed) and
returns an object with `fit(smiles, labels)` and `decision_function(smiles)`:

```python
# my_models.py
from sklearn.svm import SVC
from qudock.bench import SmilesModel
from qudock.features import morgan_fingerprints

def my_svm(info):
    return SmilesModel(morgan_fingerprints, SVC(), {"C": [0.1, 1, 10]}, info)
```

```bash
qudock bench --models my_models:my_svm,rf_fingerprint,qsvc_pls
```

`SmilesModel` handles featurising and inner cross-validation; anything else with
those two methods works too.
