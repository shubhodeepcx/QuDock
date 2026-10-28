# Quantum kernels

All kernels are simulated exactly with batched numpy statevectors (the linear ZZ map
is checked against Qiskit to machine precision), so studies run on a laptop.

## ZZ fidelity kernel (`qudock.quantum.ZZKernel`)

k(x, y) = |⟨φ(x)|φ(y)⟩|² for the ZZ feature map on PCA or PLS angles, with a
bandwidth that scales the angles and is tuned by kernel-target alignment.

## Projected kernel (`qudock.kernels.ProjectedKernel`)

An RBF kernel on each qubit's X, Y and Z expectations (Huang et al., 2021). Local
measurements resist concentration, and hardware needs one circuit per molecule
instead of one per pair.

## Nystroem approximation (`qudock.kernels.Nystroem`, `NystroemSVM`)

φ(x) = k(x, L) K_LL^{-1/2} with m landmark molecules. The Gram matrix costs n·m
kernel entries instead of n²/2, which is the difference between linear and
quadratic circuit counts on hardware. It is exact when every training molecule is a
landmark, and tests check the error falls as landmarks are added.

## Classical shadows (`qudock.shadows.ShadowFeatures`)

Every shot measures each qubit in a random Pauli basis. The same snapshots give
unbiased estimates of all single-qubit expectations and all 9·n(n−1)/2 two-qubit
correlators, so the projected kernel can see entanglement for no extra
measurement settings. A finite shot budget can be simulated before spending
quantum time.

## Pharmacophore-graph feature map (`qudock.graphmap.GraphKernel`)

One qubit per pharmacophore family (donor, acceptor, positive and negative
ionizable, aromatic, hydrophobe). Each qubit's angle grows with how many of that
group the molecule has, and every pair of qubits is entangled in proportion to how
close those groups sit in the molecule. The entanglement pattern changes from
molecule to molecule. `qudock graph` compares it with an RBF SVM and a Random
Forest on exactly the same 21 numbers, so any difference is due to the quantum
map, not to the features.

## Trainable feature map (`qudock.trainable`)

Per-qubit encoding weights, per-pair entangling weights and extra RY rotations,
trained by maximising kernel-target alignment, starting from the plain ZZ map.
