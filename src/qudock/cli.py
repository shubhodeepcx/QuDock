"""Command line entry point: ``qudock data``, ``benchmark`` and ``sweep``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qudock.benchmark import BenchmarkConfig, compare_to_best_classical, run_benchmark, summarise
from qudock.data import load_bace, random_split, scaffold_split
from qudock.plots import learning_curve, qubit_sweep
from qudock.sweep import qubit_sweep as run_sweep


def cmd_data(args):
    df = load_bace()
    print(f"molecules: {len(df)}  inhibitors: {df['label'].mean():.1%}")
    for name, split in [("scaffold", scaffold_split(df["smiles"], args.test_frac)),
                        ("random", random_split(df["label"], args.test_frac, args.seed))]:
        y = df["label"].values
        print(f"{name:>8}: train {len(split.train)} ({y[split.train].mean():.1%} active)  "
              f"test {len(split.test)} ({y[split.test].mean():.1%} active)")


def cmd_benchmark(args):
    sizes = [None if s == "full" else int(s) for s in args.sizes.split(",")]
    cfg = BenchmarkConfig(n_qubits=args.qubits, sizes=sizes,
                          seeds=list(range(args.seeds)), reps=args.reps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(cfg)
    results.to_csv(out / "benchmark.csv", index=False)
    summary = summarise(results)
    summary.to_csv(out / "summary.csv", index=False)
    stats = compare_to_best_classical(results)
    stats.to_csv(out / "qsvc_vs_classical.csv", index=False)
    learning_curve(summary, out / "learning_curve.png")

    print(summary.pivot(index="n_train", columns="model", values="mean").round(3))
    print(stats.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def cmd_sweep(args):
    lo, hi = (int(v) for v in args.qubits.split("-"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    perf, diag = run_sweep(range(lo, hi + 1), seeds=range(args.seeds))
    perf.to_csv(out / "qubit_sweep.csv", index=False)
    diag.to_csv(out / "kernel_diagnostics.csv", index=False)
    qubit_sweep(perf, diag, out / "qubit_sweep.png")

    print(perf.groupby(["n_qubits", "model"])["roc_auc"].mean().unstack().round(3))
    print(diag.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qudock")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download BACE and show split sizes")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("benchmark", help="learning curves for QSVC vs classical models")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--sizes", default="25,50,100,200,400,full")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("sweep", help="ROC-AUC and kernel geometry across qubit counts")
    p.add_argument("--qubits", default="2-10", help="range, e.g. 2-10")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
