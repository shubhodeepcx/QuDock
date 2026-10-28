"""Where did this number come from? A small record saved next to every result.

Each study writes ``provenance.json`` into its output folder with the exact
command, the code version (git commit, and whether the tree had uncommitted
changes), the package versions that matter, a hash of each raw dataset file
the study could read, and when it ran. Anyone holding a results folder can
then rerun the same thing and check they get the same table.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from qudock import __version__

PACKAGES = ("qiskit", "qiskit-aer", "qiskit-machine-learning", "qiskit-ibm-runtime", "rdkit",
            "scikit-learn", "numpy", "pandas", "scipy")


def git_state(root: Path | None = None) -> dict[str, str | bool | None]:
    """Current commit and whether there are uncommitted changes; None outside a checkout."""
    root = root or Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hash(df: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of cleaned molecules and labels, for data built in memory."""
    digest = hashlib.sha256()
    for smiles, label in zip(df["smiles"], df["label"]):
        digest.update(f"{smiles}\t{label}\n".encode())
    return digest.hexdigest()


def record(command: list[str] | None = None, config: dict | None = None,
           data_files: list[Path] | None = None) -> dict:
    """Everything needed to tell whether two results folders should match."""
    return {
        "qudock_version": __version__,
        "command": command if command is not None else sys.argv,
        "config": config or {},
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "data_files": {Path(p).name: file_hash(Path(p))
                       for p in (data_files or []) if Path(p).exists()},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write(out: Path, **kwargs) -> Path:
    """Save :func:`record` as ``out/provenance.json`` and return its path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "provenance.json"
    path.write_text(json.dumps(record(**kwargs), indent=2, default=str) + "\n")
    return path
