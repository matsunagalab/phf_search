# PHF Search

Sequence optimization for protein structures using AlphaFold2 via ColabDesign. Supports monomers and homooligomers from any PDB target.

## Project Structure

```
predict.py           - AF2 prediction wrapper (ColabDesign hallucination protocol, configurable length/copies)
mc_search.py         - Monte Carlo search engine (mutate -> evaluate -> accept/reject)
evaluate.py          - SequenceEvaluator: sequence -> all metrics + fitness (search-strategy agnostic)
esm_scores.py        - ESMScorer: amyloid / LLPS probabilities from ESM2-3B embeddings
aggrescan.py         - AGGRESCAN reimplementation (a3v scale + sliding window + hot spots)
fitness.py           - Fitness function (w_plddt * pLDDT - w_rmsd * RMSD + w_amyloid * amyloid + w_llps * LLPS + w_aggrescan * AGGRESCAN)
utils.py             - Kabsch RMSD, chain-permutation RMSD minimization, sequence mutation
prepare_reference.py - Extract reference CA coordinates and sequence from any PDB
run_search.py        - CLI entry point (--pdb-id, --chains for target selection)
download_params.sh   - AF2 parameter download
models/esm_heads/    - Vendored logistic-regression checkpoints for the two sequence scores
pyproject.toml       - Project definition (managed by uv)
```

## Setup

```bash
uv sync --extra cuda                # Everything, including torch + fair-esm (CPU: --extra cpu)
bash download_params.sh             # AF2 parameters -> params/
uv run python prepare_reference.py  # Default: 5O3L chains A,C,E,G,I -> data/5o3l_acegi_*
uv run python prepare_reference.py --pdb-id 6ELM --chains A  # -> data/6elm_a_*
```

ESM2-3B (~5.7 GB) downloads itself into the torch hub cache on first use of the
sequence scores; the classifier heads are already in `models/esm_heads/`.

## Usage

```bash
# PHF tau (default)
uv run python run_search.py --n-steps 100

# WNK2 CCT1 monomer
uv run python run_search.py --pdb-id 6ELM --chains A --n-steps 100
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdb-id` | `5O3L` | Target PDB ID |
| `--chains` | `A,C,E,G,I` | Comma-separated chain IDs |
| `--data-dir` | `params` | AF2 parameters directory |
| `--ref-coords` | Auto-derived | Reference CA coordinates (.npy) |
| `--initial-seq` | Auto-loaded | Starting sequence |
| `--n-steps` | 1000 | Number of MC steps |
| `--temperature` | 1.0 | MC temperature |
| `--n-mutations` | 1 | Mutations per step |
| `--num-recycles` | 3 | AF2 recycles |
| `--w-plddt` | 1.0 | pLDDT weight |
| `--w-rmsd` | 1.0 | RMSD weight |
| `--w-aggrescan` | 0.0 | AGGRESCAN weight; negative penalizes |
| `--aggrescan-metric` | `na4vss` | Which AGGRESCAN scalar enters the fitness: `na4vss` / `a3vsa` / `thsar` (nhs/nnhs are recorded but too coarse to optimize) |
| `--esm-scores` | `auto` | `auto` (on when a weight is nonzero) / `on` / `off` |
| `--w-amyloid` | 0.0 | Amyloidogenicity weight; negative penalizes |
| `--w-llps` | 0.0 | LLPS propensity weight; negative penalizes |
| `--amyloid-probe-lengths` | `6,15` | Window lengths, or `none` for the whole sequence |
| `--amyloid-agg` | `mean` | Which aggregate of the per-residue profile enters the fitness: `mean` or `max` |
| `--esm-cpu` | off | Run ESM2 on CPU |
| `--esm-toks-per-batch` | 4096 | ESM2 batch size in tokens |
| `--log-interval` | 10 | Log every N steps |
| `--save-interval` | 1 | Save PDB every N steps |
| `--structures-dir` | `structures` | PDB output directory |
| `--output` | `results.json` | Output JSON path |

## Output

- `results.json` -- Full trajectory (sequence, pLDDT, RMSD, fitness per step)
- `structures/step_0000.pdb` -- Predicted structure for the initial sequence
- `structures/step_NNNN.pdb` -- Structures at subsequent steps (per `--save-interval`)

## Technical Details

- **Prediction**: ColabDesign hallucination protocol, non-multimer (ptm models), configurable `copies` (5 for PHF, 1 for monomers)
- **Structure comparison**: Kabsch RMSD over all n_chains! chain permutations, taking the minimum (trivial for monomers)
- **Fitness**: `w_plddt * pLDDT - w_rmsd * RMSD + w_amyloid * amyloid + w_llps * LLPS + w_aggrescan * AGGRESCAN` (higher is better; a metric passed as `None` drops out of the sum). Everything past `w_rmsd` in `compute_fitness` is keyword-only: each added metric used to shift the positional arguments of the ones after it, silently reassigning a caller's weights
- **Sequence scores**: ESM2-3B (layer 36, mean-pooled) -> logistic regression, per Lobo et al., PNAS 123:e2531932123 (2026). ESM2 is loaded once and shared by both heads, the windows of a candidate are batched together, and results are cached per sequence. Agrees with the upstream `amyloid-predict` / `llps-predict` CLIs to <1e-6 on the three values tested (VQIVYK via the 6aa and general heads, the tau 73mer via the LLPS head) -- re-check after changing batching, dtype or device, since the equivalence is empirical, not structural
- **Recorded metrics**: `amyloid_mean` and `amyloid_max` (likewise for LLPS) are always both written to every record, and `amyloid`/`llps` hold the aggregate the fitness used (`--amyloid-agg`; the LLPS term always uses the mean, which equals the score under whole-sequence LLPS scoring). Selecting one aggregate must never cost the other, since a rerun means redoing every ESM evaluation
- **CLI surface**: only the knobs that change results or unblock a machine are flags. LLPS windowing, the amyloid classifier policy and the checkpoint directory are `ESMScorer` constructor arguments, whose defaults are the lengths and heads the published classifiers were trained for
- **Windowing**: the amyloid heads were trained on 6/10/15 aa peptides, so longer sequences are scored by sliding windows, averaging window scores onto residues, then `mean`/`max` over the profile. The LLPS head was trained on whole IDRs, so it scores the sequence in one shot by default
- **AGGRESCAN**: reimplemented from Conchillo-Sole et al., BMC Bioinformatics 8:65 (2007); a3v scale from its additional file 1, `HST = -0.02` from additional file 2. ~80 us per candidate, so always computed, never gated. NOT the official server, and two details are unreproducible from the publication: the charge correction applied to the 2 extreme residues is never given (they inherit the nearest window centre here), and the areas are specified as "trapezoidal integration (midpoint rule)", which names two different methods (`aat`/`ta`/`thsa` are the trapezoid of the threshold-clipped profile). a3v, a4v and hot-spot detection are checked against hand computation; the areas are not, since there is nothing unambiguous to check them against. Values must not be reported as server output
- **AGGRESCAN scale trap**: `na4vss` is a sum of order -13, not a probability, so `--w-aggrescan 1.0` outweighs the whole pLDDT term thirteen times over. Sensible weights start around 0.01-0.05. `nhs`/`nnhs` are recorded but excluded from `METRICS`: over a fixed-length target they take a handful of distinct values (three across all 1387 single mutants of the 73-residue PHF core) and the steps are threshold noise. Every hot spot carries a `margin` for exactly that reason -- on native PHF tau one hot spot clears `HST` by 0.50 and the other by 0.0026
- **TANGO / WALTZ**: deliberately absent. TANGO ships as a licensed compiled binary (no source), WALTZ is web-server only and its score depends on a FoldX-derived matrix. AGGRESCAN stands in for TANGO's role and the `6aa` head for WALTZ's; neither reproduces the original numbers, so do not compare results here against published TANGO/WALTZ values
- **GPU memory**: ESM2-3B moves onto the GPU at the first scored candidate and peaks at ~11.1 GiB there (measured, default batching, RTX A6000) on top of AF2's allocation, so `run_search.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` when the scores are on and ESM2 is on the GPU (before importing `predict`); set the variable yourself to override. Cost per candidate ~3 s against ~10 s for the AF2 prediction; two ESM forward passes at the default `--esm-toks-per-batch 4096`, three at 1024
- **MC acceptance**: Metropolis criterion `exp(delta_fitness / temperature)`
- **PDB saving**: `model.save_pdb(filename=None, get_best=False)` returns PDB string (note: `save_current_pdb` has a missing `return` in ColabDesign)
- **Reference files**: `data/<pdb_id>_<chains>_ca_coords.npy`, `data/<pdb_id>_<chains>_sequence.txt`

## Development

- Package management via `uv`
- Python >= 3.10
- `data/`, `params/`, `structures/`, `results.json` are in `.gitignore`
