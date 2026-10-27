"""QuDock in the browser: train the models, then screen a molecule.

Run from the repository root with:

    streamlit run app/app.py
"""

from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel

ROOT = Path(__file__).resolve().parents[1]
# lets the app run from a fresh clone (or a hosting service) without `pip install -e .`
sys.path.insert(0, str(ROOT / "src"))

from qudock.analysis import geometric_difference, kernel_target_alignment
from qudock.classical import make_classical, scores
from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.quantum import _STATE_CACHE, QSVC, ZZKernel

RESULTS = ROOT / "results"
NAMES = {"qsvc": "QSVC (quantum kernel)", "svm_rbf": "SVM-RBF", "random_forest": "Random Forest",
         "logreg": "Logistic Regression"}

st.set_page_config(page_title="QuDock", page_icon="⚛️", layout="wide")


@st.cache_data(show_spinner="Loading the BACE dataset...")
def dataset():
    df = load_bace(ROOT / "data" / "cache")
    split = scaffold_split(df["smiles"])
    return df, split.train, split.test


# keep only the last few settings, so visitors trying many combinations
# do not push the hosted app over its memory limit
@st.cache_resource(max_entries=3, show_spinner="Training the quantum and classical models...")
def train(n_qubits: int, n_train: int, reps: int, bandwidth: float):
    return q(_train, dataset(), n_qubits, n_train, reps, bandwidth)


def _train(data, n_qubits, n_train, reps, bandwidth):
    # statevectors cached for earlier settings are never reused here
    _STATE_CACHE.clear()
    df, train_idx, test_idx = data
    rng = np.random.default_rng(0)
    if n_train < len(train_idx):
        train_idx = rng.choice(train_idx, n_train, replace=False)
    smiles, y = df["smiles"].values, df["label"].values
    pipe = FeaturePipeline(n_qubits=n_qubits).fit(smiles[train_idx])
    X_train, X_test = pipe.transform(smiles[train_idx]), pipe.transform(smiles[test_idx])
    models = {"qsvc": QSVC(C=1, reps=reps, bandwidth=bandwidth).fit(X_train, y[train_idx])}
    for name in ["svm_rbf", "random_forest", "logreg"]:
        models[name] = make_classical(name, n_train=len(train_idx)).fit(X_train, y[train_idx])
    aucs = {name: roc_auc_score(y[test_idx], scores(m, X_test)) for name, m in models.items()}
    return {"pipe": pipe, "models": models, "aucs": aucs, "X_train": X_train,
            "train_idx": train_idx, "test_idx": test_idx}


def mol_image(smi: str, size=(320, 240)):
    return Draw.MolToImage(Chem.MolFromSmiles(smi), size=size)


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_resource
def qiskit_worker():
    # Streamlit runs every rerun on a fresh thread, and Qiskit's Rust core can
    # crash when circuits are bound on short-lived threads; keep it on one
    return ThreadPoolExecutor(max_workers=1)


def q(fn, *args):
    """Run a function that touches Qiskit on the dedicated worker thread."""
    return qiskit_worker().submit(fn, *args).result()


def main():
    # ---------------------------------------------------------------- sidebar
    st.sidebar.title("QuDock")
    st.sidebar.caption("Quantum kernels for Alzheimer's drug screening (BACE-1)")
    st.sidebar.subheader("Model settings")
    n_qubits = st.sidebar.slider("Qubits (one feature per qubit)", 2, 8, 4)
    n_train = st.sidebar.select_slider("Training molecules", [100, 300, 600, 1210], value=300)
    reps = st.sidebar.radio("Feature map repetitions", [1, 2], index=1, horizontal=True)
    bandwidth = st.sidebar.slider("Kernel bandwidth", 0.1, 1.0, 0.5, 0.1)
    st.sidebar.caption("Defaults match the demo notebook. Larger settings take longer to train.")

    df, _, _ = dataset()
    run = train(n_qubits, n_train, reps, bandwidth)
    pipe, models = run["pipe"], run["models"]

    st.title("QuDock")
    st.subheader("Does a quantum kernel find Alzheimer's drug candidates?")
    st.write("Each molecule is turned into a fingerprint, compressed to one angle per qubit and "
             "loaded into a ZZ feature map. A quantum kernel SVM is compared with three classical "
             "models on exactly the same features. Test molecules come from chemical families the "
             "models have never seen (scaffold split).")

    tab_screen, tab_compare, tab_kernel, tab_results = st.tabs(
        ["1. Screen a molecule", "2. Compare models", "3. Inside the kernel", "4. Full results & IBM hardware"])

    # ---------------------------------------------------------------- screen a molecule
    with tab_screen:
        test_idx = run["test_idx"]
        y = df["label"].values
        examples = {}
        for label, tag in [(1, "known inhibitor"), (0, "known non-inhibitor")]:
            for i in test_idx[y[test_idx] == label][:6]:
                examples[f"Test molecule #{i} ({tag})"] = i
        # open on the first example the quantum model gets right; the others include misses
        hits = [k for k, i in examples.items()
                if q(models["qsvc"].predict, pipe.transform([df["smiles"].values[i]]))[0] == y[i]]
        options = ["Paste my own SMILES", *examples]
        choice = st.selectbox("Pick an unseen test molecule, or paste your own SMILES below", options,
                              index=options.index(max(hits, key=lambda k: "non" not in k)) if hits else 1)
        if choice == "Paste my own SMILES":
            smi = st.text_input("SMILES", "CC(=O)Oc1ccccc1C(=O)O").strip()
            true = None
        else:
            smi = df["smiles"].values[examples[choice]]
            true = int(y[examples[choice]])
            st.code(smi, language=None)

        if Chem.MolFromSmiles(smi) is None:
            st.error("That SMILES string could not be read. Check it and try again.")
            st.stop()

        x = pipe.transform([smi])
        col_mol, col_pred = st.columns([1, 2])
        with col_mol:
            st.image(mol_image(smi), caption="Molecule")
            st.write("**Qubit angles** (radians, 0 to π)")
            st.dataframe(pd.DataFrame([x[0].round(3)], columns=[f"q{i}" for i in range(n_qubits)]),
                         hide_index=True)
        with col_pred:
            rows = []
            for name, m in models.items():
                s = float(q(scores, m, x)[0])
                hit = q(m.predict, x)[0] == 1
                rows.append({"Model": NAMES[name], "Prediction": "Inhibitor" if hit else "Not an inhibitor",
                             "Score": round(s, 3), "Test ROC-AUC": round(run["aucs"][name], 3)})
            top = rows[0]
            if top["Prediction"] == "Inhibitor":
                st.success(f"Quantum kernel SVM: likely BACE-1 inhibitor (score {top['Score']})")
            else:
                st.info(f"Quantum kernel SVM: probably not a BACE-1 inhibitor (score {top['Score']})")
            if true is not None:
                st.write(f"Lab result for this molecule: **{'inhibitor' if true else 'not an inhibitor'}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Score: decision value for the SVMs and Logistic Regression (above 0 means "
                       "inhibitor), probability for Random Forest (above 0.5). Test ROC-AUC is measured "
                       "on all unseen test molecules.")

            st.write("**Most similar training molecules according to the quantum kernel**")
            k = q(models["qsvc"].kernel_, x, run["X_train"])[0]
            cols = st.columns(3)
            for c, j in zip(cols, np.argsort(k)[::-1][:3]):
                i = run["train_idx"][j]
                c.image(mol_image(df["smiles"].values[i], (220, 160)))
                c.caption(f"overlap {k[j]:.2f} · {'inhibitor' if y[i] else 'non-inhibitor'}")

        with st.expander("Show the quantum circuit for this molecule"):
            def draw():
                circ = ZZKernel(n_qubits, reps=reps, bandwidth=bandwidth).circuit
                return fig_png(circ.assign_parameters(bandwidth * x[0]).decompose().draw("mpl", fold=-1))
            st.image(q(draw), width="stretch")
            st.caption("ZZ feature map: Hadamards, then Z rotations by each angle and ZZ "
                       "entangling gates between neighbouring qubits, repeated. The kernel between "
                       "two molecules is the overlap of their quantum states.")

    # ---------------------------------------------------------------- compare models
    with tab_compare:
        aucs = pd.Series({NAMES[k]: v for k, v in run["aucs"].items()}, name="Test ROC-AUC")
        c1, c2, c3 = st.columns(3)
        c1.metric("Molecules in BACE", f"{len(df):,}")
        c2.metric("Training molecules", f"{len(run['train_idx']):,}")
        c3.metric("Unseen test molecules", f"{len(run['test_idx']):,}")
        ranked = aucs.sort_values()
        fig, ax = plt.subplots(figsize=(8, 2.8))
        colors = ["#5B3BA8" if n.startswith("QSVC") else "#B8BDEB" for n in ranked.index]
        ax.barh(ranked.index, ranked.values, color=colors)
        for i, v in enumerate(ranked.values):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center")
        ax.axvline(0.5, ls="--", color="grey", lw=1)
        ax.set_xlim(0.4, 0.8)
        ax.set_xlabel("Test ROC-AUC on unseen molecules (0.5 = random guessing)")
        ax.spines[["top", "right"]].set_visible(False)
        st.image(fig_png(fig), width=760)
        best = aucs.idxmax()
        if best.startswith("QSVC"):
            note = ("Here the quantum kernel SVM comes out on top. Over the full experiments with "
                    "several seeds, though, Random Forest stays ahead (see tab 4).")
        else:
            note = ("The quantum kernel SVM is usually close to the RBF SVM, and Random Forest "
                    "stays ahead in the full experiments.")
        st.write(f"With these settings the best model is **{best}**. {note} Change the settings "
                 "in the sidebar to see how this moves.")

    # ---------------------------------------------------------------- kernel view
    with tab_kernel:
        X = run["X_train"]
        y_train = df["label"].values[run["train_idx"]]
        order = np.argsort(y_train)[::-1][:60]
        K_q = q(models["qsvc"].kernel_, X)
        K_c = rbf_kernel(X, gamma=1 / (4 * X.var()))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, K, title in [(axes[0], K_q, "Quantum (ZZ) kernel"), (axes[1], K_c, "Classical RBF kernel")]:
            im = ax.imshow(K[np.ix_(order, order)], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8)
        st.image(fig_png(fig), width=760)
        c1, c2, c3 = st.columns(3)
        c1.metric("Label alignment, quantum", f"{kernel_target_alignment(K_q, y_train):.3f}")
        c2.metric("Label alignment, RBF", f"{kernel_target_alignment(K_c, y_train):.3f}")
        c3.metric("Geometric difference", f"{geometric_difference(K_c, K_q):.1f}")
        st.caption("60 training molecules, inhibitors first. Each pixel is how similar two molecules "
                   "look to the kernel. Higher alignment means the kernel groups inhibitors together "
                   "more clearly; a large geometric difference means the two kernels see the data in "
                   "very different ways.")

    # ---------------------------------------------------------------- full results
    with tab_results:
        st.subheader("Final round study (from results/final)")
        st.image(str(RESULTS / "final" / "final_study.png"),
                 caption="Mean test ROC-AUC over 10 seeds, 6 qubits, scaffold split")
        summary = pd.read_csv(RESULTS / "final" / "final_summary.csv")
        full = summary[summary["n_train"] == summary["n_train"].max()]
        st.dataframe(full.drop(columns=["n_train"]).sort_values("roc_auc", ascending=False)
                     .round(3), hide_index=True)
        st.caption("PLS features and a tuned bandwidth lift QSVC from 0.708 to 0.743, but the "
                   "same features help classical models more. Full-fingerprint models stay best.")

        st.subheader("Full experiments (from results/full)")
        c1, c2 = st.columns(2)
        c1.image(str(RESULTS / "full" / "learning_curve.png"),
                 caption="Sample efficiency: no small-data advantage for the quantum kernel")
        c2.image(str(RESULTS / "full" / "qubit_sweep.png"),
                 caption="Qubit sweep: QSVC peaks at 8 qubits (0.719)")
        st.image(str(RESULTS / "full" / "noise_study.png"),
                 caption="Simulated IBM Torino noise: 8 to 13% kernel error, accuracy barely moves")

        st.subheader("On a real IBM quantum computer")
        hw = RESULTS / "hardware"
        info = json.loads((hw / "hardware_run.json").read_text())
        st.write(f"Backend **{info['backend']}** · {info['n_molecules']} molecules · "
                 f"{info['n_circuits']} circuits · {info['shots']:,} shots · "
                 f"{info['quantum_seconds']} s of quantum time")
        saved = np.load(hw / "hardware_kernels.npz")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (key, title) in zip(axes, [("exact_reps2", "Exact (ideal)"),
                                           ("faketorino_reps2", "FakeTorino simulation"),
                                           ("device_reps2", f"{info['backend']} (real device)")]):
            im = ax.imshow(saved[key], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8)
        st.image(fig_png(fig), width="stretch")
        st.dataframe(pd.read_csv(hw / "hardware_kernel.csv").round(3), hide_index=True)
        st.caption("The real device adds more error than the simulation (17.5% vs 10.3%), but the "
                   "kernel keeps its shape, so the ranking of molecules barely changes.")


main()
