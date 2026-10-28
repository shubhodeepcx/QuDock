"""Command line entry point: ``qudock data``, ``benchmark``, ``sweep``, ``noise``,
``hardware``, ``final``, ``scaling``, ``trainable``, ``confidence``, ``scorecard``,
``explain``, ``graph``, ``screen``, ``hardware-svm`` and ``reproduce``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qudock import __version__, provenance
from qudock.benchmark import BenchmarkConfig, compare_to_best_classical, run_benchmark, summarise
from qudock.conformal import run_confidence_study
from qudock.data import DATASETS, DEFAULT_CACHE, load_bace, random_split, scaffold_split
from qudock.explain import run_explain_study, summarise_explain
from qudock.final_study import FinalConfig, compare_pairs, run_final_study, summarise_final
from qudock.hardware import HardwareConfig, reanalyse, run_hardware_study, save_hardware_study
from qudock.noise_study import NoiseConfig, run_noise_study
from qudock.plots import (
    confidence_study,
    explain_study,
    final_study,
    learning_curve,
    noise_study,
    qubit_sweep,
    scaling_study,
    trainable_study,
)
from qudock.plots import (
    scorecard as plot_scorecard,
)
from qudock.scaling import run_scaling_study
from qudock.scorecard import build_scorecard, verdicts
from qudock.sweep import qubit_sweep as run_sweep
from qudock.trainable import run_trainable_study, summarise_trainable


def parse_sizes(text: str) -> list[int | None]:
    """'25,50,full' -> [25, 50, None]; 'full' means the whole training set."""
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if part == "full":
            sizes.append(None)
        elif part.isdigit() and int(part) > 0:
            sizes.append(int(part))
        else:
            raise argparse.ArgumentTypeError(
                f"bad training size {part!r}: use positive integers or 'full'")
    return sizes


def parse_ints(text: str) -> list[int]:
    """'256,1024' -> [256, 1024]."""
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}") from None


def parse_range(text: str) -> range:
    """'2-10' -> range(2, 11)."""
    try:
        lo, hi = (int(v) for v in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a range like 2-10, got {text!r}") from None
    if not 1 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"range must be increasing and start at 1+, got {text}")
    return range(lo, hi + 1)


def cmd_data(args):
    df = DATASETS[args.dataset]()
    print(f"{args.dataset} molecules: {len(df)}  positives: {df['label'].mean():.1%}")
    for name, split in [("scaffold", scaffold_split(df["smiles"], args.test_frac)),
                        ("random", random_split(df["label"], args.test_frac, args.seed))]:
        y = df["label"].values
        print(f"{name:>8}: train {len(split.train)} ({y[split.train].mean():.1%} active)  "
              f"test {len(split.test)} ({y[split.test].mean():.1%} active)")


def cmd_benchmark(args):
    cfg = BenchmarkConfig(n_qubits=args.qubits, sizes=args.sizes,
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
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    perf, diag = run_sweep(args.qubits, seeds=range(args.seeds))
    perf.to_csv(out / "qubit_sweep.csv", index=False)
    diag.to_csv(out / "kernel_diagnostics.csv", index=False)
    qubit_sweep(perf, diag, out / "qubit_sweep.png")

    print(perf.groupby(["n_qubits", "model"])["roc_auc"].mean().unstack().round(3))
    print(diag.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def cmd_noise(args):
    cfg = NoiseConfig(n_qubits=args.qubits, n_train=args.n_train, n_test=args.n_test,
                      shots=args.shots,
                      backend=args.backend, seeds=list(range(args.seeds)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = run_noise_study(cfg)
    results.to_csv(out / "noise_study.csv", index=False)
    noise_study(results, out / "noise_study.png")

    summary = (results.groupby(["reps", "noise", "shots", "psd_repair"])
               [["rel_frobenius", "roc_auc"]].mean().round(3))
    print(summary)
    print(f"wrote results to {out}/")


def cmd_hardware(args):
    out = Path(args.out)
    cfg = HardwareConfig(n_molecules=args.molecules, shots=args.shots, backend=args.backend)
    if args.reanalyse and not (out / "hardware_kernels.npz").exists():
        raise SystemExit(f"nothing to reanalyse: {out}/hardware_kernels.npz is missing")
    if args.reanalyse:
        rows, kernels, info = reanalyse(out, cfg)
    else:
        backend = None
        if args.fake:
            from qiskit_ibm_runtime.fake_provider import FakeTorino
            backend = FakeTorino()
        rows, kernels, info = run_hardware_study(cfg, backend=backend)
    save_hardware_study(rows, kernels, info, out)

    print(rows.round(3).to_string(index=False))
    print(f"{info['backend']}  job {info['job_id']}  quantum seconds {info['quantum_seconds']}")
    print(f"wrote results to {out}/")


PAIRS = [("hybrid_pca", "tanimoto_svm"), ("hybrid_pls", "tanimoto_svm"),
         ("hybrid_pls", "rf_fingerprint"), ("hybrid_pls", "gb_fingerprint"),
         ("qsvc_pls", "qsvc_pca"),
         ("qsvc_pls", "svm_rbf_pls"), ("qsvc_pls", "rf_pls"), ("qsvc_pls", "svm_rbf_pca"),
         ("qsvc_pls", "rf_pca"), ("projected_pls", "qsvc_pls")]


def cmd_final(args):
    models = args.models.split(",") if args.models else None
    cfg = FinalConfig(n_qubits=args.qubits, sizes=args.sizes, seeds=list(range(args.seeds)),
                      **({"models": models} if models else {}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, test_scores = run_final_study(cfg, DATASETS[args.dataset]())
    results.to_csv(out / "final_runs.csv", index=False)
    summary = summarise_final(results)
    summary.to_csv(out / "final_summary.csv", index=False)
    if test_scores is not None:
        test_scores.to_csv(out / "full_size_test_scores.csv", index=False)
        pairs = compare_pairs(test_scores, PAIRS)
        pairs.to_csv(out / "paired_tests.csv", index=False)
        print(pairs.round(3).to_string(index=False))
    final_study(summary, out / f"{'final' if args.dataset == 'bace' else args.dataset}_study.png")

    print(summary.pivot(index="n_train", columns="model", values="roc_auc").round(3))
    print(f"wrote results to {out}/")


def _read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def cmd_scorecard(args):
    """One chart of every measured setting, built from the result files."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = _read(Path("results/final/final_runs.csv"))
    if final is None:
        raise SystemExit("run `qudock final` first: results/final/final_runs.csv is missing")
    table = build_scorecard(final, _read(Path("results/bbbp/final_runs.csv")),
                            _read(Path("results/scaling/scaling.csv")),
                            _read(Path("results/trainable/trainable_runs.csv")))
    table["verdict"] = verdicts(table)
    table.to_csv(out / "scorecard.csv", index=False)
    plot_scorecard(table, table["verdict"], out / "scorecard.png")
    print(table.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def cmd_trainable(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = run_trainable_study(seeds=range(args.seeds), n_qubits=args.qubits)
    runs.to_csv(out / "trainable_runs.csv", index=False)
    summary = summarise_trainable(runs)
    summary.to_csv(out / "trainable_summary.csv", index=False)
    trainable_study(summary, out / "trainable_study.png")
    print(summary.pivot(index="n_train", columns="feature_map",
                        values=["roc_auc", "alignment"]).round(3))
    print(f"wrote results to {out}/")


def cmd_explain(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_explain_study(n_qubits=args.qubits, n_train=args.n_train,
                                n_molecules=args.molecules, seed=args.seed)
    results.to_csv(out / "explain_study.csv", index=False)
    explain_study(results, out / "explain_study.png")
    summary = summarise_explain(results)
    pd.Series(summary).to_csv(out / "explain_summary.csv", header=False)
    save_examples(results, out, args)
    print(results[["label", "score", "n_atoms", "n_removed", "top_substructure",
                   "top_atoms", "random_atoms", "gap"]].round(3).to_string(index=False))
    print(pd.Series(summary).round(3).to_string())
    print(f"wrote results to {out}/")


def save_examples(results, out, args, n: int = 3):
    """Draw the highlighted molecules themselves, for the poster and the README."""
    from qudock.explain import draw_explanation, explain_molecule
    from qudock.features import FeaturePipeline, morgan_fingerprints
    from qudock.kernels import KernelSVM, stack, tune_bandwidth

    df = load_bace()
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(split.train, size=min(args.n_train, len(split.train)), replace=False)
    pipe = FeaturePipeline(args.qubits, seed=args.seed, method="pls", use_descriptors=True)
    A_tr = pipe.fit_transform(smiles[keep], y[keep])
    bandwidth = tune_bandwidth(A_tr, y[keep], args.qubits, seed=args.seed)
    model = KernelSVM(kind="hybrid", n_angles=args.qubits, C=1, weight=0.25,
                      bandwidth=bandwidth).fit(stack(A_tr, morgan_fingerprints(smiles[keep])),
                                               y[keep])
    for i, smi in enumerate(results.sort_values("gap", ascending=False)["smiles"].head(n)):
        png = draw_explanation(explain_molecule(smi, pipe, model, fingerprints=True))
        (out / f"example_{i + 1}.png").write_bytes(png)


def cmd_confidence(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets, domain = run_confidence_study(n_qubits=args.qubits, seed=args.seed)
    sets.to_csv(out / "conformal_sets.csv", index=False)
    domain.to_csv(out / "applicability_domain.csv", index=False)
    confidence_study(sets, domain, out / "confidence_study.png")
    print(sets.round(3).to_string(index=False))
    print(domain.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


BBBP_MODELS = ("qsvc_pca,qsvc_pls,svm_rbf_pls,rf_pls,tanimoto_svm,rf_fingerprint,"
               "gb_fingerprint,hybrid_pca")

REPRODUCE = [
    ("benchmark", ["--out", "results/full"]),
    ("sweep", ["--out", "results/full"]),
    ("noise", ["--out", "results/full"]),
    ("final", ["--out", "results/final"]),
    ("final", ["--dataset", "bbbp", "--sizes", "100,400,full", "--seeds", "5",
               "--models", BBBP_MODELS, "--out", "results/bbbp"]),
    ("scaling", ["--out", "results/scaling"]),
    ("trainable", ["--out", "results/trainable"]),
    ("confidence", ["--out", "results/confidence"]),
    ("explain", ["--out", "results/explain"]),
    ("scorecard", ["--out", "results/scorecard"]),
    ("hardware", ["--reanalyse"]),          # saved device data, no quantum time needed
    ("hardware-svm", ["--reanalyse"]),
]

# saved device data each reanalysis needs; without it the step is skipped, not failed
SAVED_DEVICE_DATA = {
    "hardware": Path("results/hardware/hardware_kernels.npz"),
    "hardware-svm": Path("results/hardware_svm/hardware_svm_features.npz"),
}


def cmd_reproduce(args):
    """Regenerate every table and figure a reader sees, from saved data only."""
    for name, extra in REPRODUCE:
        if args.skip and name in args.skip.split(","):
            continue
        saved = SAVED_DEVICE_DATA.get(name)
        if "--reanalyse" in extra and saved is not None and not saved.exists():
            print(f"\n=== skipping qudock {name}: no saved device data at {saved}")
            continue
        print(f"\n=== qudock {name} {' '.join(extra)}")
        main([*(["--verbose"] if args.verbose else []), name, *extra])


def cmd_scaling(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_scaling_study(qubits=args.qubits, n_train=args.n_train, seed=args.seed)
    results.to_csv(out / "scaling.csv", index=False)
    scaling_study(results, out / "scaling_study.png")
    print(results[["n_qubits", "kind", "bandwidth", "off_diagonal_spread", "roc_auc"]]
          .round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def cmd_graph(args):
    """Pharmacophore-graph feature map against its classical twins."""
    from qudock.graphmap import run_graph_study
    from qudock.plots import graph_study

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_graph_study(seeds=range(args.seeds), sizes=args.sizes, reps=args.reps)
    results.to_csv(out / "graph_runs.csv", index=False)
    graph_study(results, out / "graph_study.png")
    print(results.groupby(["n_train", "model"])["roc_auc"].mean().unstack().round(3))
    print(f"wrote results to {out}/")


def cmd_screen(args):
    """Rank a library by activity, brain penetration and drug-likeness."""
    from qudock.data import load_bbbp
    from qudock.screen import ScreenConfig, Screener, holdout_check, read_library, screen_library

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ScreenConfig(n_qubits=args.qubits, confidence=args.confidence, seed=args.seed)
    bace, bbbp = load_bace(), load_bbbp()
    if args.library is None:
        table, summary = holdout_check(bace, bbbp, cfg)
        table.sort_values("combined", ascending=False).to_csv(out / "holdout_ranking.csv",
                                                              index=False)
        summary.to_csv(out / "holdout_summary.csv")
        print(summary.round(3).to_string())
    else:
        ranked = screen_library(Screener(cfg).fit(bace, bbbp), read_library(args.library))
        ranked.to_csv(out / "ranking.csv", index=False)
        cols = [c for c in ("rank", "name", "smiles", "combined", "p_active", "activity_call",
                            "bbb_probability", "qed", "pains", "in_domain") if c in ranked]
        print(ranked[cols].head(args.top).round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def cmd_hardware_svm(args):
    from qudock.hardware_svm import HardwareSVMConfig, reanalyse, run_hardware_svm, save

    out = Path(args.out)
    cfg = HardwareSVMConfig(n_train=args.n_train, n_test=args.n_test, backend=args.backend)
    if args.reanalyse and not (out / "hardware_svm_features.npz").exists():
        raise SystemExit(f"nothing to reanalyse: {out}/hardware_svm_features.npz is missing")
    if args.reanalyse:
        rows = reanalyse(out, cfg)
        rows.to_csv(out / "hardware_svm.csv", index=False)
    else:
        backend = None
        if args.fake:
            from qiskit_ibm_runtime.fake_provider import FakeTorino
            backend = FakeTorino()
        rows, saved, info = run_hardware_svm(cfg, backend=backend)
        save(rows, saved, info, out)
        print(f"{info['backend']}  job {info['job_id']}  quantum seconds {info['quantum_seconds']}")
    print(rows.round(3).to_string(index=False))
    print(f"wrote results to {out}/")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qudock")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="download a dataset and show split sizes")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bace")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("benchmark", help="learning curves for QSVC vs classical models")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--sizes", type=parse_sizes, default="25,50,100,200,400,full")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("sweep", help="ROC-AUC and kernel geometry across qubit counts")
    p.add_argument("--qubits", type=parse_range, default="2-10", help="range, e.g. 2-10")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("noise", help="QSVC with shot-sampled kernels on a fake IBM backend")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--shots", type=parse_ints, default="256,1024,4096")
    p.add_argument("--backend", default="FakeTorino")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_noise)

    p = sub.add_parser("hardware", help="measure a small kernel on an IBM Quantum device")
    p.add_argument("--backend", default=None, help="device name, default: least busy")
    p.add_argument("--molecules", type=int, default=12)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--fake", action="store_true", help="run on FakeTorino locally instead")
    p.add_argument("--reanalyse", action="store_true",
                   help="redo the comparison from saved device kernels, no device needed")
    p.add_argument("--out", default="results/hardware")
    p.set_defaults(func=cmd_hardware)

    p = sub.add_parser("final", help="final-round study: tuned, projected and hybrid kernels")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="bace")
    p.add_argument("--models", default=None, help="comma-separated subset of the models")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--sizes", type=parse_sizes, default="50,100,200,400,full")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/final")
    p.set_defaults(func=cmd_final)

    p = sub.add_parser("reproduce", help="rerun every study that makes a table or figure")
    p.add_argument("--skip", default=None, help="comma-separated commands to leave out")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("scaling",
                       help="kernel concentration and accuracy from 4 to 14 qubits")
    p.add_argument("--qubits", type=parse_ints, default="4,6,8,10,12,14")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/scaling")
    p.set_defaults(func=cmd_scaling)

    p = sub.add_parser("scorecard", help="one chart of every setting we measured")
    p.add_argument("--out", default="results/scorecard")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("trainable",
                       help="train the feature map's own parameters by kernel alignment")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results/trainable")
    p.set_defaults(func=cmd_trainable)

    p = sub.add_parser("confidence",
                       help="conformal prediction sets and accuracy by chemical similarity")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/confidence")
    p.set_defaults(func=cmd_confidence)

    p = sub.add_parser("explain",
                       help="which atoms drove each prediction, and whether they really did")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--molecules", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/explain")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("graph",
                       help="feature map built from the pharmacophore graph, vs classical twins")
    p.add_argument("--sizes", type=parse_sizes, default="100,400,full")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--out", default="results/graph")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("screen",
                       help="rank a library by BACE-1 activity, brain penetration and QED")
    p.add_argument("library", nargs="?", default=None,
                   help="CSV with a smiles column, or .smi file; leave out to rank the "
                        "BACE test set and check the ranking against known labels")
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--top", type=int, default=20, help="rows to print")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/screen")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("hardware-svm",
                       help="train and test a projected-kernel SVM on an IBM Quantum device")
    p.add_argument("--backend", default=None, help="device name, default: least busy")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--fake", action="store_true", help="run on FakeTorino locally instead")
    p.add_argument("--reanalyse", action="store_true",
                   help="redo the analysis from saved device features, no device needed")
    p.add_argument("--out", default="results/hardware_svm")
    p.set_defaults(func=cmd_hardware_svm)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    args.func(args)
    if getattr(args, "out", None) and Path(args.out).is_dir():
        config = {k: v for k, v in vars(args).items() if k != "func"}
        provenance.write(Path(args.out), command=["qudock", *(argv or sys.argv[1:])],
                         config=config, data_files=sorted(DEFAULT_CACHE.glob("*.csv")))


if __name__ == "__main__":
    main()
