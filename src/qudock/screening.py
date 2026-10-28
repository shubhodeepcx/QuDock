"""Virtual-screening metrics and paired model comparisons on a fixed test set.

ROC-AUC rewards ranking every molecule well; a screening campaign only tests
the top of the list, so we also report enrichment (how many more inhibitors
land in the top x% than random picking would give), BEDROC (early recognition)
and PR-AUC.
"""

from __future__ import annotations

import numpy as np
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import average_precision_score, roc_auc_score


def screening_metrics(y_true, score, alpha: float = 20.0) -> dict[str, float]:
    y_true, score = np.asarray(y_true), np.asarray(score)
    ranked = [[int(y)] for y in y_true[np.argsort(-score, kind="stable")]]
    ef1, ef5 = Scoring.CalcEnrichment(ranked, 0, [0.01, 0.05])
    return {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "ef_1": ef1,
        "ef_5": ef5,
        "bedroc": Scoring.CalcBEDROC(ranked, 0, alpha),
    }


def paired_bootstrap(y_true, score_a, score_b, n_boot: int = 2000, seed: int = 0):
    """Mean ROC-AUC difference (a - b), its 95% interval and a two-sided p-value.

    Both models are scored on the same resampled molecules, so the interval
    reflects the test set, which is what a DeLong test measures too.
    """
    rng = np.random.default_rng(seed)
    y_true, a, b = map(np.asarray, (y_true, score_a, score_b))
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) == 2:
            diffs.append(roc_auc_score(y_true[i], a[i]) - roc_auc_score(y_true[i], b[i]))
    diffs = np.array(diffs)
    if not len(diffs):
        raise ValueError("every bootstrap resample had a single class; need both labels")
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"delta": roc_auc_score(y_true, a) - roc_auc_score(y_true, b),
            "ci_low": lo, "ci_high": hi, "p_value": min(p, 1.0)}
