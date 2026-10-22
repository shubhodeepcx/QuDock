"""Measure a small ZZ kernel on a real IBM Quantum device.

Kept deliberately small (a dozen molecules, 4 qubits) so it fits in a few
minutes of free-plan quantum time. The same kernel is also computed exactly and
on the FakeTorino noise model, so the three can be compared directly. The
device kernels are saved, so the comparison can be redone later without
spending more quantum time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.transpiler import generate_preset_pass_manager

from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.noise import (
    SampledZZKernel,
    fidelity_circuit,
    kernel_error,
    make_simulator,
    nearest_psd,
)
from qudock.quantum import ZZKernel

log = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    n_qubits: int = 4
    n_molecules: int = 12
    reps: list[int] = field(default_factory=lambda: [2])
    bandwidth: float = 0.5
    shots: int = 1024
    backend: str | None = None          # None = least busy device
    max_execution_time: int = 240       # hard cap on quantum seconds billed
    seed: int = 0


def pick_molecules(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Half inhibitors, half non-inhibitors, chosen at random."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.sort(np.r_[rng.choice(pos, n // 2, replace=False),
                         rng.choice(neg, n - n // 2, replace=False)])


def prepare(cfg: HardwareConfig, df: pd.DataFrame | None = None):
    """Features and labels of the molecules sent to the device."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    pipe = FeaturePipeline(cfg.n_qubits, seed=cfg.seed).fit(smiles[split.train])
    idx = split.train[pick_molecules(y[split.train], cfg.n_molecules, cfg.seed)]
    return pipe.transform(smiles[idx]), y[idx]


def kernel_circuits(X: np.ndarray, reps: int, bandwidth: float):
    """One bound compute-uncompute circuit per upper-triangle pair of X."""
    qc, xp, yp = fidelity_circuit(X.shape[1], reps)
    pairs = list(zip(*np.triu_indices(len(X), k=1)))
    circuits = [qc.assign_parameters({**dict(zip(xp, bandwidth * X[i])),
                                      **dict(zip(yp, bandwidth * X[j]))}) for i, j in pairs]
    return circuits, pairs


def counts_to_kernel(counts: list[dict[str, int]], pairs, n: int, n_qubits: int) -> np.ndarray:
    zeros = "0" * n_qubits
    K = np.eye(n)
    for c, (i, j) in zip(counts, pairs):
        K[i, j] = K[j, i] = c.get(zeros, 0) / sum(c.values())
    return K


def run_on_backend(backend, circuits, shots: int, max_execution_time: int | None, seed: int):
    """Transpile for the device and run everything as one Sampler job."""
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=seed)
    isa = pm.run(circuits)
    sampler = SamplerV2(mode=backend)
    if max_execution_time:
        sampler.options.max_execution_time = max_execution_time
    job = sampler.run(isa, shots=shots)
    log.info("submitted job %s to %s", job.job_id(), backend.name)
    result = job.result()
    counts = [r.data.meas.get_counts() for r in result]
    depth = max(c.depth(lambda g: g.operation.num_qubits == 2) for c in isa)
    return counts, job, depth


def select_backend(name: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if name:
        return service.backend(name)
    return service.least_busy(operational=True, simulator=False)


def measure_device_kernels(X: np.ndarray, cfg: HardwareConfig, backend):
    """Device kernel for every reps setting, from a single job."""
    all_circuits, layout = [], []
    for reps in cfg.reps:
        circuits, pairs = kernel_circuits(X, reps, cfg.bandwidth)
        layout.append((reps, len(all_circuits), len(circuits), pairs))
        all_circuits += circuits

    t0 = time.time()
    counts, job, depth = run_on_backend(backend, all_circuits, cfg.shots,
                                        cfg.max_execution_time, cfg.seed)
    kernels = {reps: counts_to_kernel(counts[start:start + n], pairs, len(X), cfg.n_qubits)
               for reps, start, n, pairs in layout}
    # local fake backends have no usage record
    usage = job.usage() if hasattr(job, "usage") else None
    info = {"backend": backend.name, "job_id": job.job_id(), "shots": cfg.shots,
            "n_molecules": len(X), "n_circuits": len(all_circuits),
            "max_two_qubit_depth": depth, "quantum_seconds": usage,
            "wall_seconds": round(time.time() - t0, 1),
            "date": time.strftime("%Y-%m-%d %H:%M %Z")}
    return kernels, info


def compare(X: np.ndarray, device: dict[int, np.ndarray], cfg: HardwareConfig, name: str):
    """Kernel error of the simulated and measured kernels against the exact one."""
    from qudock.noise_study import fake_backend

    fake = make_simulator(fake_backend("FakeTorino"), seed=cfg.seed)
    rows, kernels = [], {}
    for reps, K_dev in device.items():
        exact = ZZKernel(X.shape[1], reps=reps, bandwidth=cfg.bandwidth)(X)
        sim = SampledZZKernel(X.shape[1], reps=reps, bandwidth=cfg.bandwidth,
                              shots=cfg.shots, simulator=fake, seed=cfg.seed)(X)
        kernels.update({f"exact_reps{reps}": exact, f"faketorino_reps{reps}": sim,
                        f"device_reps{reps}": K_dev})
        for label, K in [("FakeTorino (simulated)", sim), (name, K_dev),
                         (f"{name}, PSD repaired", nearest_psd(K_dev))]:
            rows.append({"reps": reps, "kernel": label, **kernel_error(K, exact)})
    return pd.DataFrame(rows), kernels


def run_hardware_study(cfg: HardwareConfig, backend=None, df: pd.DataFrame | None = None):
    """Exact vs FakeTorino vs device kernels for the same molecules.

    Returns (summary rows, kernels dict, run info dict).
    """
    X, labels = prepare(cfg, df)
    backend = backend or select_backend(cfg.backend)
    device, info = measure_device_kernels(X, cfg, backend)
    rows, kernels = compare(X, device, cfg, backend.name)
    kernels["features"] = X
    info["labels"] = labels.tolist()
    return rows, kernels, info


def reanalyse(out: Path, cfg: HardwareConfig | None = None):
    """Redo the comparison from saved device kernels, without the device."""
    saved = np.load(out / "hardware_kernels.npz")
    info = json.loads((out / "hardware_run.json").read_text())
    cfg = cfg or HardwareConfig()
    X = saved["features"] if "features" in saved else prepare(cfg)[0]
    device = {int(k.removeprefix("device_reps")): saved[k]
              for k in saved.files if k.startswith("device_reps")}
    rows, kernels = compare(X, device, cfg, info["backend"])
    kernels["features"] = X
    return rows, kernels, info


def save_hardware_study(rows: pd.DataFrame, kernels: dict, info: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "hardware_kernel.csv", index=False)
    np.savez(out / "hardware_kernels.npz", **kernels)
    (out / "hardware_run.json").write_text(json.dumps(info, indent=2) + "\n")
