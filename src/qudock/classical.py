"""Classical baselines, tuned by stratified inner cross-validation."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

GRIDS = {
    "svm_rbf": (
        lambda seed: SVC(kernel="rbf", random_state=seed),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1]},
    ),
    "random_forest": (
        lambda seed: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        {"max_depth": [None, 10], "min_samples_leaf": [1, 3]},
    ),
    "logreg": (
        lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
        {"C": [0.01, 0.1, 1, 10]},
    ),
}


def make_classical(name: str, seed: int = 0, n_train: int | None = None) -> GridSearchCV:
    """Return a GridSearchCV wrapper for one of the baselines in ``GRIDS``.

    ``n_train`` lets small training sets use fewer folds so every fold still
    has both classes.
    """
    build, grid = GRIDS[name]
    folds = 5 if n_train is None or n_train >= 100 else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return GridSearchCV(build(seed), grid, cv=cv, scoring="roc_auc", n_jobs=-1)


def scores(model, X):
    """Continuous scores for ROC-AUC, whatever the model exposes."""
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except AttributeError:  # GridSearchCV over a model without one
            pass
    return model.predict_proba(X)[:, 1]
