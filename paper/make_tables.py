"""Write the LaTeX tables the paper inputs, from the result files.

    python paper/make_tables.py        # after `qudock bench --out results/bench`
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "rf_fingerprint": "Random Forest, ECFP4",
    "gb_fingerprint": "Gradient boosting, ECFP4 + descriptors",
    "tanimoto_svm": "Tanimoto SVM",
    "qsvc_pls": "QSVC, PLS (quantum)",
    "hybrid_pls": "Hybrid ZZ + Tanimoto (quantum)",
    "nystroem_projected": "Projected kernel, Nystr\\\"om (quantum)",
    "graph_quantum": "Pharmacophore-graph map (quantum)",
}


def bench_table(runs: pd.DataFrame) -> str:
    full = runs[runs["n_train"] == runs.groupby("task")["n_train"].transform("max")]
    g = full.groupby(["model", "task"])["roc_auc"]
    mean, ci = g.mean().unstack(), (1.96 * g.sem().fillna(0)).unstack()
    tasks = [t for t in ("bace", "bbbp", "clintox") if t in mean]
    rank = mean[tasks].rank(ascending=False).mean(axis=1).sort_values()
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{QuDock-Bench: test ROC-AUC (mean $\pm$ 95\% interval over "
        f"{full['seed'].nunique()} seeds) on scaffold splits, full training sets.}}",
        r"\label{tab:bench}", r"\small",
        r"\begin{tabular}{l" + "c" * len(tasks) + "c}", r"\toprule",
        "Model & " + " & ".join(t.upper() for t in tasks) + r" & Mean rank\\", r"\midrule",
    ]
    for model in rank.index:
        cells = [f"{mean.loc[model, t]:.3f} $\\pm$ {ci.loc[model, t]:.3f}" for t in tasks]
        lines.append(f"{NAMES.get(model, model)} & " + " & ".join(cells)
                     + f" & {rank[model]:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    runs = pd.read_csv(ROOT / "results" / "bench" / "bench_runs.csv")
    out = ROOT / "paper" / "bench_table.tex"
    out.write_text(bench_table(runs))
    print(f"wrote {out}")
