# ESM2 logistic-regression heads

Vendored classifier weights for the two sequence-level metrics used by
`esm_scores.py`. Each file is a tiny torch checkpoint holding one logistic
regression: `embedding_dim` (2560), `weight_full`, `bias`, with the training
scaler already folded into the weights. A score is
`sigmoid(weight_full . e + bias)`, where `e` is the mean-pooled ESM2-3B
(layer 36) embedding of the peptide.

| File | Metric | Trained on |
|------|--------|-----------|
| `general_model_latest.pt` | amyloid | peptides up to 20 aa |
| `6aa_model_latest.pt` | amyloid | hexapeptides (WALTZ-style benchmark) |
| `6aa_FETA_model_latest.pt` | amyloid | hexapeptides, FETA feature subset |
| `10aa_model_latest.pt` | amyloid | 10 aa fragments |
| `15aa_model_latest.pt` | amyloid | 15 aa tau fragments |
| `LLPS_model_latest.pt` | LLPS | IDR drivers vs. non-drivers (CD-CODE / IDRome) |

## Provenance

Copied verbatim (SHA256-verified) from the `v1.0.0-manuscript` tag of:

- <https://github.com/samlobe/amyloid-predict> tag `v1.0.0-manuscript`
  (commit `e07bd33ba01bc59bad4c9f018170cb25f72ae276`), path
  `model_development/models/` -- Zenodo <https://doi.org/10.5281/zenodo.18882460>
- <https://github.com/samlobe/LLPS-predict> tag `v1.0.0-manuscript`
  (commit `53986d7157c5d40f6fea9bc37f4937b1942147d1`), path
  `model_development/LLPS_model_latest.pt` -- Zenodo
  <https://doi.org/10.5281/zenodo.18882439>

They are vendored (rather than fetched at setup time) because the upstream
packages ship only code in their wheels -- the checkpoints live outside the
Python package, so `pip install git+...` does not deliver them. Vendoring keeps
`uv sync` sufficient for a working environment. The upstream packages are still
declared as dependencies, and `esm_scores.py` scores through their own inference
functions rather than reimplementing the arithmetic; the resulting values agree
with the published `amyloid-predict` / `llps-predict` CLIs to <1e-6 on the cases
tested (VQIVYK through the 6aa and general heads, the tau 73mer through the LLPS
head).

The ESM2-3B weights themselves (~5.7 GB) are *not* vendored; `fair-esm`
downloads them into the torch hub cache on first use.

## License and citation

Both upstream repositories are MIT licensed (Copyright (c) 2026 Sam Lobo).
If you use these weights or scores derived from them, cite:

Lobo S, Griem L, Shell MS, Shea J-E (2026) amyloid-predict and LLPS-predict:
Predicting phase separation propensities in the intrinsically disordered
proteome. Proc Natl Acad Sci USA 123: e2531932123.
<https://doi.org/10.1073/pnas.2531932123>
