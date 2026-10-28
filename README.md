<div align="center">

# BlackOut

**VTU-QFF-HC-0140** &nbsp;|&nbsp; Tier: **Online**

</div>

---

<div align="center">

| | |
| :--- | :--- |
| **Team** | BlackOut |
| **Team ID** | VTU-QFF-HC-0140 |
| **Tier** | Online |
| **Repository** | [vtu-hc-blackout](https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-blackout.git) |
| **Organisation** | [vtu-qiskit-fall-fest-2026](https://github.com/vtu-qiskit-fall-fest-2026) |
| **Event** | VTU-QFF Hackathon, VTU-Qiskit Fall Fest 2026 |

</div>

---

## 🧭 Overview

QuDock tests whether a quantum kernel can find BACE-1 inhibitors (a key Alzheimer's drug target)
better than classical machine learning when data is scarce and hardware is noisy. Molecules from
the MoleculeNet BACE dataset (1,513) are turned into fingerprints, compressed to one angle per
qubit and loaded into a ZZ feature map. A quantum kernel SVM is compared with SVM-RBF, Random
Forest and Logistic Regression on identical features, using a scaffold split, 25 to 1,210
training molecules, 2 to 10 qubits and multiple seeds. The quantum model matches the RBF SVM
(best ROC-AUC 0.719 at 8 qubits) but does not beat Random Forest (0.729) and shows no small-data
advantage. Simulated IBM Torino noise distorts the kernel by 8 to 13%, and a real run on
ibm_marrakesh gives 17.5%, yet the kernel keeps its shape (correlation 0.996) and accuracy barely
changes.

**Demo video:** [watch on Google Drive](https://drive.google.com/file/d/17vJ5pzTXjhN4cXxzwQyV3GzaghxDYzhk/view?usp=sharing)

**Live web app:** [qudock-blackout.streamlit.app](https://qudock-blackout.streamlit.app/)

**Presentation:** [VTU-QFF-HC-0140.pptx](docs/VTU-QFF-HC-0140.pptx)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Team Members](#-team-members-github-ids)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Approach & Architecture](#-approach--architecture)
- [Results](#-results)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [How to Push Code](#-how-to-push-code)
- [Submission Checklist](#-submission-checklist)
- [Resources](#-resources)
- [License](#-license)

---

## 👥 Team Members (GitHub IDs)

> Members are added to this repository as **collaborators** and must accept the
> GitHub invitation before they can push. Add your GitHub username below.

| # | Name | GitHub Username | Role |
| :-- | :--- | :--- | :--- |
| 1 | Shubhodeep Chatterjee | [@shubhodeepcx](https://github.com/shubhodeepcx) | Team Lead |
| 2 | Srijita Banerjee | [@srijita29-stack](https://github.com/srijita29-stack) | Developer |
| 3 | Rounak Pal | [@Rounak639](https://github.com/Rounak639) | Developer |

<!--
Optional: add a plain list instead of the table above.
- @username1
- @username2
-->

---

## ❓ Problem Statement

<!--
State the problem you are solving. Be specific: what is broken or missing,
who experiences it, and why does it matter. Link to the official problem
statement from the brief if one applies.
-->

**Problem:** Early-stage drug discovery involves screening thousands of molecules to identify compounds that can inhibit a specific target protein. However, obtaining experimentally measured activity data is expensive and time-consuming, meaning machine learning models often have to learn from limited labelled examples. Quantum machine learning, particularly quantum kernel methods, offers a potential approach by representing molecular data in quantum feature spaces that may capture complex patterns. However, claims that these methods learn more effectively from less data remain uncertain. Their performance must be evaluated against properly tuned classical models using realistic molecular data splits, while accounting for data leakage, computational costs, and quantum hardware noise.

**Why it matters:** β-secretase 1 (BACE-1) is an important enzyme studied as a potential therapeutic target in Alzheimer's disease research. Predicting molecular activity against BACE-1 provides a practical setting for evaluating quantum machine learning in drug discovery. This project will compare quantum kernel methods with classical models using molecular fingerprints, scaffold-based data splitting, and sample-efficiency experiments. It will also investigate how training data size, quantum circuit design, and hardware noise affect predictive performance. The goal is to determine whether quantum kernels offer a measurable advantage over classical approaches in data-limited virtual screening and whether that advantage, if present, remains meaningful under realistic conditions.

**Track / Category:** Online · Quantum machine learning · Drug discovery

---

## 💡 Proposed Solution

<!--
Describe your solution. Keep it concrete: the core idea, the user flow, and
the quantum/ML technique you are applying.
-->

**Solution:** We built **QuDock**, a hybrid quantum and classical pipeline that predicts whether
a molecule inhibits BACE-1. Molecular fingerprints are compressed and loaded into a ZZ feature
map, and a quantum kernel SVM (QSVC) is trained on the result. We compare it with SVM-RBF,
Random Forest and Logistic Regression on the same features, with 25 to 1,210 training
molecules, 2 to 10 qubits, with simulated IBM device noise, and on a real IBM quantum computer.

**Key idea:** Use the fidelity kernel $k(x, y) = |\langle\phi(x)|\phi(y)\rangle|^2$ of a ZZ
feature map, and test it with a fair benchmark plus a few checks on the kernel itself
(geometric difference, kernel-target alignment and concentration).

**Why this works:** All models get exactly the same features. The data is split by molecular
scaffold, so the test set contains molecule families the models have never seen.
Hyperparameters are tuned with cross-validation, and every result is averaged over several
seeds and checked with a paired significance test. So if quantum and classical results differ,
it is because of the models and not because of how we set up the comparison.

---

## 🧩 Approach & Architecture

<!--
Walk through how it works. A simple pipeline diagram in text or Mermaid is ideal.
-->

```mermaid
flowchart LR
    A[BACE SMILES<br/>1,513 molecules] --> B[Clean + scaffold split]
    B --> C[ECFP4 fingerprint<br/>2048 bits]
    C --> D[Standardise + PCA<br/>n = qubits, scaled to 0..π]
    D --> E[ZZ feature map<br/>fidelity kernel]
    D --> F[Classical baselines<br/>SVM-RBF, RF, LogReg]
    E --> G[QSVC]
    F --> H[Benchmark, kernel analysis,<br/>noise study]
    G --> H
```

1. **Input:** the MoleculeNet BACE dataset, 1,513 SMILES strings, each labelled as inhibitor or
   non-inhibitor (`qudock.data`).
2. **Preprocessing:** we canonicalise the SMILES, drop molecules that can't be parsed or have
   conflicting labels, and split by Bemis-Murcko scaffold (1,210 train, 303 test). Each molecule
   becomes a Morgan (ECFP4) fingerprint, which is standardised, reduced with PCA to one value per
   qubit and rescaled to [0, π]. All of this is fit on the training data only
   (`qudock.features`).
3. **Quantum core:** a ZZFeatureMap with linear entanglement, 1 or 2 reps and a tunable
   bandwidth. For the benchmark the kernel is computed exactly from statevectors (and cached).
   For the noise study it is estimated from compute-uncompute circuits with a finite number of
   shots on AerSimulator, with and without the FakeTorino noise model. Finally, the same
   circuits for 12 molecules are run on a real IBM device, ibm_marrakesh (`qudock.quantum`,
   `qudock.noise`, `qudock.hardware`).
4. **Post-processing:** an SVM is trained on the precomputed kernel, with C and bandwidth chosen
   by stratified inner cross-validation. Noisy kernels can optionally be repaired to the nearest
   positive semi-definite (PSD) matrix.
5. **Output:** test ROC-AUC, accuracy, F1 and MCC for each model, training size and seed,
   Wilcoxon tests of QSVC against the best classical model, kernel diagnostics, and figures in
   `results/full/`.

---

## 📊 Results

All numbers are test ROC-AUC on the scaffold split. The full tables and figures are in
[`results/full/`](results/full), and the [demo notebook](notebooks/demo.ipynb) goes through them
step by step.

**Sample efficiency** (6 qubits, average of 5 seeds):

| Training molecules | QSVC | SVM-RBF | Random Forest | Logistic Regression |
| --: | --: | --: | --: | --: |
| 25 | 0.592 | 0.644 | 0.625 | 0.623 |
| 50 | 0.588 | 0.653 | 0.640 | 0.594 |
| 100 | 0.649 | 0.666 | 0.700 | 0.573 |
| 200 | 0.673 | 0.692 | 0.704 | 0.593 |
| 400 | 0.688 | 0.703 | 0.723 | 0.596 |
| 1,210 | 0.687 | 0.683 | 0.740 | 0.607 |

![Learning curve](results/full/learning_curve.png)

**Qubit sweep** (full training set, average of 3 seeds): QSVC does best with 8 qubits, at 0.719.
That is better than SVM-RBF (0.705) and Logistic Regression (0.626) but still below Random
Forest (0.729).

![Qubit sweep](results/full/qubit_sweep.png)

**Noise study** (4 qubits, 80 train and 40 test molecules, 3 seeds, simulated IBM Torino):
here each kernel entry is estimated from measurement shots, with and without the noise model of
the FakeTorino device. Circuits are transpiled at optimization level 3, so they sit on well
chosen qubits, as on a real device. Each cell shows relative kernel error / test ROC-AUC, using
the raw kernel without PSD repair.

| Setting | reps = 1 | reps = 2 |
| :-- | --: | --: |
| Exact kernel | 0.000 / 0.650 | 0.000 / 0.626 |
| Shots only, 256 | 0.055 / 0.616 | 0.064 / 0.628 |
| Shots only, 1024 | 0.027 / 0.649 | 0.032 / 0.622 |
| Shots only, 4096 | 0.014 / 0.639 | 0.016 / 0.623 |
| FakeTorino, 256 | 0.096 / 0.633 | 0.132 / 0.631 |
| FakeTorino, 1024 | 0.082 / 0.659 | 0.119 / 0.644 |
| FakeTorino, 4096 | 0.078 / 0.640 | 0.115 / 0.633 |

![Noise study](results/full/noise_study.png)

**Real IBM quantum computer** (ibm_marrakesh, 12 training-set molecules with 6 inhibitors, 4 qubits,
reps = 2, 66 circuits at 1,024 shots in one job, 20 seconds of quantum time): we measured the
kernel on the device and compared it with the exact kernel and with FakeTorino simulating the
same circuits.

| Kernel | Relative error vs exact | Mean off-diagonal value | Correlation with exact |
| :-- | --: | --: | --: |
| Exact | 0.000 | 0.262 | 1.000 |
| FakeTorino simulation | 0.103 | 0.238 | 0.999 |
| ibm_marrakesh (real device) | 0.175 | 0.218 | 0.996 |

Details, including the job ID, are in [`results/hardware/`](results/hardware).

**What we learned**

- QSVC does about as well as an RBF SVM on the same features and beats Logistic Regression
  from 100 molecules up, but it never beats Random Forest. It is 0.03 to 0.07 ROC-AUC behind
  the best classical model at every training size. With only 5 seeds none of these gaps is
  statistically significant (the smallest possible Wilcoxon p-value is 0.06).
- We found no advantage with little data. With 25 or 50 training molecules QSVC is actually the
  weakest model.
- The quantum kernel is very different from the RBF kernel (geometric difference up to about
  100), and at bandwidth 0.5 it lines up better with the labels (alignment 0.15 to 0.18, against
  0.09 to 0.10 for RBF). That didn't lead to better predictions on this dataset.
- Kernel concentration shows up clearly. At bandwidth 1.0 the variance of the off-diagonal
  kernel values falls from 0.075 at 2 qubits to 0.005 at 10 qubits, so the bandwidth has to be
  tuned.
- Device noise changes the kernel more than shot noise does. Simulated Torino noise causes 8 to
  13% error that more shots don't fix, while shot noise alone causes 1.4 to 6.4%. Even so, test
  ROC-AUC stays between 0.62 and 0.66 in every setting, and with the variation between seeds
  (about 0.05 to 0.10) we can't measure a difference at this sample size. Repairing the noisy
  kernel to be PSD makes almost no difference.
- The real device is noisier than the simulation (17.5% kernel error against 10.3% for
  FakeTorino on the same molecules), but the kernel keeps its shape: its values correlate 0.996
  with the exact kernel. The noise mostly shrinks every value a little, which is why the SVM's
  ranking of molecules, and so its accuracy, barely changes.
- Our first noise study placed circuits on poorly chosen qubits and reported 25 to 28% error.
  Checking it against the real device exposed this, and the numbers above are from the
  corrected run.

### Final round: can a better-fed or hybrid quantum kernel close the gap?

For the final round we went after the two weak points above: the quantum kernel only ever saw
6 unsupervised PCA numbers, and the bandwidth was fixed by hand. Every model below runs on the
same scaffold split, the same training sizes and the same 10 seeds
(`qudock final --out results/final`, under an hour on a laptop CPU). Besides ROC-AUC we now
report screening metrics: enrichment of inhibitors in the top 1% and 5% of the ranking, BEDROC
and PR-AUC. All files are in [`results/final/`](results/final).

What we added:

- **Supervised compression:** PLS on the fingerprint plus 10 RDKit descriptors (weight, logP,
  TPSA, H-bond donors and acceptors and so on) instead of PCA, so the 6 qubits get the
  directions that matter for activity.
- **Adaptive bandwidth:** chosen per run by kernel-target alignment on the training set only.
  This is our answer to kernel concentration, and it picks 0.19 to 0.30 for PLS features and
  0.55 to 0.80 for PCA features.
- **Projected quantum kernel** (Huang et al., 2021): each molecule's single-qubit X, Y and Z
  expectations go into an RBF kernel. It needs one circuit per molecule instead of one per pair,
  which is what makes a full classifier on real hardware affordable.
- **Hybrid kernel:** w x quantum + (1 - w) x Tanimoto on the full fingerprint, with w chosen by
  inner cross-validation.
- **Fair baselines:** SVM-RBF and Random Forest on exactly the same PLS features, and Tanimoto SVM
  and Random Forest on the full 2,048-bit fingerprint, which is the classical ceiling.

![Final round study](results/final/final_study.png)

**Mean test ROC-AUC over 10 seeds** (6 qubits):

| Training molecules | QSVC, PCA | QSVC, PLS + tuned bw | Projected, PLS | SVM-RBF, PLS | RF, PLS | Hybrid, PCA | Tanimoto SVM | RF, fingerprint |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 50 | 0.607 | 0.656 | 0.581 | 0.675 | 0.666 | 0.651 | 0.655 | 0.684 |
| 100 | 0.647 | 0.687 | 0.633 | 0.710 | 0.717 | 0.702 | 0.712 | 0.735 |
| 200 | 0.663 | 0.712 | 0.687 | 0.737 | 0.737 | 0.726 | 0.742 | 0.756 |
| 400 | 0.693 | 0.732 | 0.699 | 0.742 | 0.750 | 0.757 | 0.762 | 0.769 |
| 1,210 | 0.708 | 0.743 | 0.717 | 0.769 | 0.768 | 0.776 | 0.781 | 0.782 |

**Paired bootstrap on the full-size test set** (2,000 resamples of the same test molecules):

| Comparison | ROC-AUC difference | 95% interval | p |
| :-- | --: | --: | --: |
| QSVC PLS vs QSVC PCA | +0.046 | -0.015 to 0.106 | 0.15 |
| QSVC PLS vs SVM-RBF PCA (first-round baseline) | +0.102 | 0.033 to 0.170 | <0.001 |
| QSVC PLS vs SVM-RBF PLS (same features) | -0.026 | -0.050 to -0.002 | 0.03 |
| QSVC PLS vs RF PLS (same features) | -0.024 | -0.049 to 0.001 | 0.06 |
| Hybrid PCA vs Tanimoto SVM | -0.020 | -0.054 to 0.013 | 0.27 |

**What we learned in the final round**

- Better features and a tuned bandwidth lift QSVC by 0.035 to 0.049 ROC-AUC at every training
  size, and it now clearly beats the first-round SVM-RBF baseline (p < 0.001).
- Most of that gain comes from the features, not from the quantum kernel. On the very same PLS
  features, SVM-RBF and Random Forest still do 0.01 to 0.03 better at every size, and the difference against
  SVM-RBF is significant (p = 0.03). We report this rather than the flattering comparison.
- The hybrid kernel keeps a real quantum share (inner CV picks w between 0.1 and 0.5) and
  reaches 0.776, within 0.006 of the best classical model, but it does not beat Tanimoto SVM
  (p = 0.27). The quantum part adds information without hurting, but not enough to win.
- With PLS inside the hybrid, inner CV overrates the quantum part, because PLS has already seen
  the training labels. The PCA hybrid has no such leak and does better (0.776 against 0.755).
- The projected kernel is slightly weaker than the fidelity kernel here, but it is the one we
  can run as a full classifier on hardware (`qudock hardware-svm`).

### Does it survive more qubits? (`qudock scaling`)

The judges asked how we keep classes separable as qubits grow, since quantum kernels concentrate:
every pair of molecules looks equally distant, and the SVM has nothing left to learn from. We
measured it directly from 4 to 14 qubits on the same 400 molecules, reporting the spread
(standard deviation) of the off-diagonal kernel values next to test ROC-AUC.

![Scaling study](results/scaling/scaling_study.png)

| Qubits | Spread, fixed bandwidth | Spread, tuned | ROC-AUC, fixed | ROC-AUC, tuned | ROC-AUC, projected |
| --: | --: | --: | --: | --: | --: |
| 4 | 0.117 | 0.273 | 0.589 | 0.681 | 0.584 |
| 6 | 0.054 | 0.197 | 0.584 | 0.679 | 0.646 |
| 8 | 0.028 | 0.144 | 0.606 | 0.661 | 0.655 |
| 10 | 0.018 | 0.117 | 0.633 | 0.666 | 0.643 |
| 12 | 0.012 | 0.089 | 0.638 | 0.663 | 0.629 |
| 14 | 0.010 | 0.173 | 0.636 | 0.671 | 0.634 |

At a fixed bandwidth the kernel does concentrate: the spread falls by a factor of 11 between 4
and 14 qubits. Tuning the bandwidth by kernel-target alignment keeps it 9 to 17 times larger,
and that shows up as 0.03 to 0.09 more ROC-AUC at every width. The alignment picks 0.2 at 4 to
12 qubits and 0.1 at 14, which is the adaptive scaling the question was about. The projected
kernel concentrates as much as the plain one by this measure, because it is the RBF on top, not
the overlap, that keeps it usable.

### Does it hold on another task? (`qudock final --dataset bbbp`)

A method tuned until it works on one dataset proves little, so we ran the same pipeline,
unchanged, on MoleculeNet BBBP: can a molecule cross the blood-brain barrier, which every
Alzheimer's drug has to do. 1,965 molecules, the same scaffold split, 5 seeds.

![BBBP study](results/bbbp/bbbp_study.png)

| Training molecules | QSVC, PLS | SVM-RBF, PLS | RF, PLS | Hybrid, PCA | Tanimoto SVM | RF, fingerprint |
| --: | --: | --: | --: | --: | --: | --: |
| 100 | 0.740 | 0.754 | 0.729 | 0.768 | 0.770 | 0.773 |
| 400 | 0.768 | 0.766 | 0.785 | 0.810 | 0.811 | 0.809 |
| 1,572 | 0.769 | 0.781 | 0.800 | 0.827 | 0.826 | 0.829 |

The BACE story repeats: the hybrid kernel is level with the best classical model (0.827 against
0.826 and 0.829, differences of 0.002 to 0.004 with p = 0.68 and 0.72, so no measurable gap),
while the quantum kernel alone stays behind Random Forest on the same features (-0.027,
p = 0.02). Two datasets, same conclusion.

### How much can you trust one prediction? (`qudock confidence`)

A screening model is only useful if it says when it does not know. We added conformal
prediction, which turns a score into a set of labels: {inhibitor}, {not an inhibitor}, both
("don't know") or neither. Its guarantee assumes new molecules resemble the calibration ones,
so we calibrate on the rarest training scaffolds, the way the test set was built, and then
measure whether the promise holds.

![Confidence study](results/confidence/confidence_study.png)

| Model | Asked for 90% | Measured | Single-label predictions | Right when it commits |
| :-- | --: | --: | --: | --: |
| QSVC, PLS | 0.90 | 0.657 | 0.911 | 0.721 |
| Hybrid, PLS | 0.90 | 0.680 | 0.957 | 0.710 |
| Tanimoto SVM | 0.90 | 0.858 | 0.571 | 0.751 |

Asking for 90% confidence really delivers 66 to 86% under a scaffold split. That is not a bug in
the method: it is what happens when test molecules are genuinely new chemistry, and it is worth
reporting, because a confidence number taken from a random split would have looked fine and
been wrong. Accuracy tracks the same thing: on the third of test molecules least similar to the
training set, every model loses 0.05 to 0.10 ROC-AUC. The app shows this similarity for each
molecule you screen.

---

## 🎬 Demo

**Demo video (about 5.5 minutes, with voiceover: project demo, then how to use the app):**
[watch on Google Drive](https://drive.google.com/file/d/17vJ5pzTXjhN4cXxzwQyV3GzaghxDYzhk/view?usp=sharing)

**Try it live:** [qudock-blackout.streamlit.app](https://qudock-blackout.streamlit.app/) (the first visit takes a minute while the models train)

**Presentation:** [VTU-QFF-HC-0140.pptx](docs/VTU-QFF-HC-0140.pptx)

A screenshot from the [demo notebook](notebooks/demo.ipynb), run end to end on a laptop with 4
qubits and 300 training molecules. It shows the ZZ feature map circuit, the quantum kernel next
to the classical RBF kernel, and the test ROC-AUC of every model on the same features.

![Demo notebook](docs/demo.png)

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.11+ |
| Quantum SDK | Qiskit 2.x, Qiskit Aer, Qiskit IBM Runtime (fake backends and a real device) |
| Quantum ML | ZZ feature map fidelity kernel + scikit-learn SVM (QSVC) |
| Chemistry | RDKit (SMILES parsing, Morgan fingerprints, Murcko scaffolds) |
| Classical ML | scikit-learn (SVM-RBF, Random Forest, Logistic Regression, PCA) |
| Simulation | Statevector (exact kernel), AerSimulator with FakeTorino noise model |
| Hardware | IBM Quantum Platform, ibm_marrakesh (156 qubits), open plan |
| Testing / CI | pytest, ruff, GitHub Actions |
| Environment | JupyterLab / VS Code, `venv`, Docker |
| Version control | Git + GitHub |

---

## 📁 Project Structure

```text
vtu-hc-blackout/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package metadata and the `qudock` command
├── Dockerfile
├── docs/demo.png          # Demo screenshot
├── .github/workflows/     # CI: ruff + pytest on every push
├── notebooks/
│   └── demo.ipynb         # End-to-end walkthrough of the pipeline and results
├── src/qudock/
│   ├── data.py            # BACE and BBBP download, cleaning, scaffold split
│   ├── features.py        # ECFP fingerprints (+ descriptors) -> PCA or PLS -> qubit angles
│   ├── classical.py       # SVM-RBF, Random Forest, Logistic Regression
│   ├── quantum.py         # ZZ fidelity kernel and QSVC
│   ├── benchmark.py       # Sample-efficiency benchmark and statistics
│   ├── analysis.py        # Geometric difference, alignment, concentration
│   ├── sweep.py           # Qubit sweep
│   ├── noise.py           # Shot-based kernel estimation, PSD repair
│   ├── noise_study.py     # Noise study on a fake IBM backend
│   ├── hardware.py        # Kernel on a real IBM device
│   ├── kernels.py         # Tanimoto, projected and hybrid kernels, bandwidth tuning
│   ├── screening.py       # Enrichment, BEDROC, PR-AUC, paired bootstrap
│   ├── conformal.py       # Calibrated confidence and the applicability domain
│   ├── scaling.py         # Kernel concentration and accuracy from 4 to 14 qubits
│   ├── final_study.py     # Final-round study
│   ├── hardware_svm.py    # Projected-kernel SVM trained and tested on a real device
│   ├── plots.py           # Figures
│   └── cli.py             # `qudock` command line
├── app/app.py             # Streamlit web app: screen molecules, compare models
├── results/full/          # Benchmark, sweep and noise study outputs
├── results/hardware/      # Real-device kernel run
├── results/final/         # Final-round study outputs
├── results/scaling/       # Kernel concentration study
├── results/confidence/    # Conformal prediction and applicability domain
├── results/bbbp/          # The same pipeline on a second dataset
├── data/                  # Downloaded dataset cache (not committed)
└── tests/                 # Unit tests
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.11 or newer
- [Git](https://git-scm.com/downloads)
- A GitHub account (you need collaborator access to this repo)

### 1. Clone the repository

```bash
git clone https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-blackout.git
cd vtu-hc-blackout
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -e .[dev]            # or: pip install -r requirements.txt
```

> **Linux only:** if importing Qiskit Aer fails with *"cannot allocate memory in static TLS
> block"*, run `export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096000` first, together
> with `export RUST_MIN_STACK=33554432` so the transpiler's threads keep enough stack.

### 4. Run

```bash
qudock data                          # download BACE, show split sizes
qudock benchmark --out results/full  # learning curves, ~15 min
qudock sweep --out results/full      # 2-10 qubit sweep + kernel diagnostics, ~15 min
qudock noise --out results/full      # shot + device noise study, ~30 min
qudock hardware --reanalyse          # redo the real-device comparison from saved results
qudock hardware                      # rerun on a real IBM device (needs a saved IBM account)
qudock final --out results/final     # final-round study: PLS, tuned bandwidth, hybrid, under an hour
qudock final --dataset bbbp          # the same pipeline on blood-brain barrier data
qudock scaling --out results/scaling # kernel concentration from 4 to 14 qubits, ~10 min
qudock confidence                    # conformal prediction sets and accuracy by similarity
qudock reproduce                     # rerun every study that makes a table or figure
qudock hardware-svm --fake           # projected-kernel SVM on a simulated IBM device
qudock hardware-svm                  # same on a real IBM device, capped at 5 min of quantum time
```

Or try the web app, live at [qudock-blackout.streamlit.app](https://qudock-blackout.streamlit.app/) or locally, where you can screen your own molecules and change the settings:

```bash
pip install -e ".[app]"
streamlit run app/app.py         # opens http://localhost:8501
```

Or open the walkthrough notebook:

```bash
jupyter lab notebooks/demo.ipynb
```

Or with Docker:

```bash
docker build -t qudock .
docker run --rm -v "$PWD/results:/app/results" qudock benchmark --out results/full
```

### 5. Run the tests

```bash
pytest -q
```

---

## 🔁 How to Push Code

> Full walkthrough: **[Git & Collaboration Handbook](../vtu-hc-guide-git-and-collaboration)**, covering
> the collaboration workflow, branches, pull requests, reviews and conflict resolution.

### Configure Git (first time only)

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

### Everyday workflow

```bash
# 1. Get the latest changes
git checkout main
git pull origin main

# 2. Create a branch for your work
git checkout -b feature/my-task

# 3. Commit your work
git add .
git commit -m "feat: add variational circuit for the search problem"

# 4. Push the branch
git push -u origin feature/my-task

# 5. Open a Pull Request on GitHub and request a review
#    https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-blackout/compare/main...feature/my-task

# 6. After approval, merge into main
git checkout main
git pull origin main
git branch -d feature/my-task
```

### Commit message convention

| Prefix | Meaning |
| :--- | :--- |
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or fixing tests |
| `refactor:` | Code restructure, no behaviour change |
| `chore:` | Build/config/dependency changes |

### Useful commands

```bash
git status              # what changed
git log --oneline -10   # recent commits
git diff                # review changes before committing
git stash               # shelve changes temporarily
git pull --rebase       # update your branch on top of main
```

### Collaborating on the same repo

- **Never push directly to `main`.** Open a Pull Request instead.
- **Pull before you push** to avoid rejected commits.
- **Commit often** with clear messages.
- **Resolve conflicts** locally: `git pull --rebase`, fix the files, then `git add` + `git rebase --continue`.
- **Ask for a review** before merging. A second pair of eyes catches bugs fast.

---

## ✅ Submission Checklist

- [x] Repository created and **public**
- [x] All team members added as **collaborators** and invitations accepted
- [x] `README.md` updated with project details, problem statement and solution
- [x] Team members listed under [Team Members](#-team-members-github-ids)
- [x] `requirements.txt` committed and project runs from a clean clone
- [x] Code pushed to `main` and the latest `main` builds successfully
- [x] Notebooks committed with **cleared outputs** (or use `nbstrip`)
- [x] `README` includes a demo screenshot / video link
- [ ] Short project summary submitted on the hackathon portal

---

## 📚 Resources

| Resource | Link |
| :--- | :--- |
| 🔁 Git & collaboration handbook | [Git & Collaboration Handbook](../vtu-hc-guide-git-and-collaboration) |
| ❓ Problem statements, solutions & terms | [Problem Statements, Solutions & Terms](../vtu-hc-guide-problem-and-solution) |
| 📄 README writing guide | [README Guide & Template](../vtu-hc-guide-readme-template) |
| ⚛️ Qiskit documentation | https://docs.quantum.ibm.com/ |
| ⚛️ Qiskit Runtime & primitives | https://quantum.cloud.ibm.com/docs/ |
| 🧪 Qiskit Aer simulator | https://docs.quantum.ibm.com/guides/simulator |
| 💻 IBM Quantum Platform | https://quantum.cloud.ibm.com/ |
| 🎓 Qiskit textbook | https://qiskit.qotlabs.org/learning/ |
| 📋 GitHub Docs | https://docs.github.com/ |

---

## 📄 License

Released under the [MIT License](LICENSE). See `LICENSE` for details.

---

<div align="center">

**BlackOut** · Team ID `VTU-QFF-HC-0140` · Tier `Online`
· Built with ⚛️ Qiskit for the VTU-QFF Hackathon

[⬆ Back to Top](#blackout)

</div>
