# Screening a library

A BACE-1 hit only matters if it can also reach the brain and look like a drug, so
`qudock screen` gives every molecule four independent readouts:

| Column | Meaning |
| :--- | :--- |
| `p_active` | Conformal p-value for "active" from the hybrid quantum + Tanimoto kernel SVM, calibrated on the rarest training scaffolds |
| `activity_call` | The prediction set at the chosen confidence: active, inactive, don't know, or unlike anything seen |
| `bbb_probability` | Random Forest trained on BBBP: probability of crossing the blood-brain barrier |
| `qed` | RDKit's quantitative estimate of drug-likeness |
| `pains` | Name of the first PAINS alert, empty when clean |
| `nn_similarity`, `in_domain` | Tanimoto similarity to the closest BACE training molecule, and whether it clears 0.35 |
| `combined` | Weighted geometric mean of `p_active` (weight 2), `bbb_probability` and `qed` |

A geometric mean means a molecule has to be decent on all three to rise. PAINS hits
and out-of-domain molecules are flagged, not dropped, so you decide.

```bash
qudock screen library.csv          # CSV with a smiles column (and optionally name)
qudock screen library.smi --top 50 # or one SMILES per line, name after a space
qudock screen                      # no library: rank the BACE test set and check it
```

The last form trains on the scaffold-split training set, ranks the held-out test
molecules, and reports enrichment for the activity score alone and for the combined
rank, so you can see what the multi-objective ranking costs or gains on molecules
whose answer is known.

Results go to `results/screen/ranking.csv`, with a `provenance.json` next to it.
