"""Command line entry point: ``qudock data``, ``benchmark``, ``sweep``, ``noise``,
``hardware``, ``final`` and ``hardware-svm``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qudock.benchmark import BenchmarkConfig, compare_to_best_classical, run_benchmark, summarise
from qudock.data import load_bace, random_split, scaffold_split
from qudock.final_study import FinalConfig, compare_pairs, run_final_study, summarise_final
from qudock.hardware import HardwareConfig, reanalyse, run_hardware_study, save_hardware_study
from qudock.noise_study import NoiseConfig, run_noise_study
from qudock.plots import final_study, learning_curve, noise_study, qubit_sweep
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


def cmd_noise(args):
    cfg = NoiseConfig(n_qubits=args.qubits, n_train=args.n_train, n_test=args.n_test,
                      shots=[int(s) for s in args.shots.split(",")],
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
         ("hybrid_pls", "rf_fingerprint"), ("qsvc_pls", "qsvc_pca"),
         ("qsvc_pls", "svm_rbf_pls"), ("qsvc_pls", "rf_pls"), ("qsvc_pls", "svm_rbf_pca"),
         ("qsvc_pls", "rf_pca"), ("projected_pls", "qsvc_pls")]


def cmd_final(args):
    sizes = [None if s == "full" else int(s) for s in args.sizes.split(",")]
    cfg = FinalConfig(n_qubits=args.qubits, sizes=sizes, seeds=list(range(args.seeds)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, test_scores = run_final_study(cfg)
    results.to_csv(out / "final_runs.csv", index=False)
    summary = summarise_final(results)
    summary.to_csv(out / "final_summary.csv", index=False)
    if test_scores is not None:
        test_scores.to_csv(out / "full_size_test_scores.csv", index=False)
        pairs = compare_pairs(test_scores, PAIRS)
        pairs.to_csv(out / "paired_tests.csv", index=False)
        print(pairs.round(3).to_string(index=False))
    final_study(summary, out / "final_study.png")

    print(summary.pivot(index="n_train", columns="model", values="roc_auc").round(3))
    print(f"wrote results to {out}/")


def cmd_hardware_svm(args):
    from qudock.hardware_svm import HardwareSVMConfig, reanalyse, run_hardware_svm, save

    out = Path(args.out)
    cfg = HardwareSVMConfig(n_train=args.n_train, n_test=args.n_test, backend=args.backend)
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

    p = sub.add_parser("noise", help="QSVC with shot-sampled kernels on a fake IBM backend")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--shots", default="256,1024,4096")
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
    p.add_argument("--qubits", type=int, default=6)
    p.add_argument("--sizes", default="50,100,200,400,full")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/final")
    p.set_defaults(func=cmd_final)

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


if __name__ == "__main__":
    main()
