# QuDock

QuDock asks one question carefully: **does a quantum kernel help find BACE-1
inhibitors**, a key Alzheimer's drug target, when data is scarce and hardware is noisy?

It compares quantum kernels (ZZ fidelity, projected, trainable, and a feature map built
from the molecule's pharmacophore graph) against strong classical baselines on
Bemis-Murcko scaffold splits, with multiple seeds, paired statistics, and runs on real
IBM Quantum hardware. Around that study it ships tools that are useful whatever the
answer is:

| Tool | What it does |
| :--- | :--- |
| [`qudock screen`](screening.md) | Ranks any library by BACE-1 activity, brain penetration and drug-likeness, with calibrated confidence |
| [`qudock bench`](bench.md) | Scores any model, yours included, on fixed tasks, splits and seeds |
| [`qudock serve`](api.md) | The screening pipeline as an HTTP API |
| [Kernels](kernels.md) | Nystroem scaling, classical shadows, graph feature map, trainable map |

## Install

```bash
git clone https://github.com/vtu-qiskit-fall-fest-2026/vtu-hc-blackout.git
cd vtu-hc-blackout
pip install -e .[dev]          # add [app] for the web app, [api] for the HTTP API
```

On Linux, if importing Qiskit Aer fails with *"cannot allocate memory in static TLS
block"*, export `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096000` and
`RUST_MIN_STACK=33554432` first (`make test` does this for you).

## Live demo

[qudock-blackout.streamlit.app](https://qudock-blackout.streamlit.app/)
