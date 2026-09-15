# PHF Search

Sequence optimization for protein structures using AlphaFold2 via ColabDesign. Supports monomers and homooligomers from any PDB target.

## Project Structure

```
predict.py           - AF2 prediction wrapper (ColabDesign hallucination protocol); returns pLDDT, pTM, ipTM, iPAE, coords
mc_search.py         - Monte Carlo search engine (mutate -> evaluate -> accept/reject)
evaluate.py          - SequenceEvaluator: sequence -> all metrics + fitness (search-strategy agnostic)
esm_scores.py        - ESMScorer: amyloid / LLPS probabilities from ESM2-3B embeddings
homology.py          - Sequence vs the target's sequence: seq_recovery (ProteinMPNN's), BLOSUM62
shape.py             - Shape fidelity vs the reference: TM-score, lDDT, Fnat, per-chain variants
aggrescan.py         - AGGRESCAN reimplementation (a3v scale + sliding window + hot spots)
mpnn_score.py        - MPNNScorer: ProteinMPNN inverse-folding score via ColabDesign's bundled model
ddg.py               - DDGLookup: candidate ddG by summing a precomputed ThermoMPNN matrix
precompute_ddg.py    - One-time: run ThermoMPNN site-saturation -> models/ddg/*.npz
fitness.py           - Fitness function (structural terms + amyloid + LLPS + AGGRESCAN + ddG)
utils.py             - Kabsch RMSD, chain-permutation RMSD minimization, sequence mutation
prepare_reference.py - Extract reference CA coordinates and sequence from any PDB
run_search.py        - CLI entry point (--pdb-id, --chains for target selection)
download_params.sh   - AF2 parameter download
models/esm_heads/    - Vendored logistic-regression checkpoints for the two sequence scores
models/ddg/          - Precomputed ThermoMPNN ddG matrices (~30 kB each, committed)
pyproject.toml       - Project definition (managed by uv)
```

## Setup

```bash
uv sync --extra cuda                # Everything, including torch + fair-esm (CPU: --extra cpu)
bash download_params.sh             # AF2 parameters -> params/
uv run python prepare_reference.py --pdb-id 5O3L --chains A,C,E,G,I  # -> data/5o3l_acegi_*
uv run python prepare_reference.py --pdb-id 6ELM --chains A          # -> data/6elm_a_*
```

ESM2-3B (~5.7 GB) downloads itself into the torch hub cache on first use of the
sequence scores; the classifier heads are already in `models/esm_heads/`.

## Usage

```bash
# PHF tau (the default target, written out)
uv run python run_search.py --pdb-id 5O3L --chains A,C,E,G,I --n-steps 100

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
| `--w-ptm` | 0.0 | pTM weight |
| `--w-iptm` | 0.0 | Interface pTM weight; higher is better |
| `--w-seq-recovery` | 0.0 | Fraction of positions matching the target's sequence (ProteinMPNN's `seq_recovery`) |
| `--w-blosum62` | 0.0 | Mean BLOSUM62 score per position against it |
| `--w-tm-score` | 0.0 | TM-score weight; higher is better |
| `--w-lddt` | 0.0 | lDDT weight; higher is better |
| `--w-fnat` | 0.0 | Fnat weight; higher is better |
| `--w-ipae` | 0.0 | Interface PAE weight; LOWER is better, so the weight should be NEGATIVE |
| `--w-aggrescan` | 0.0 | AGGRESCAN weight; negative penalizes |
| `--mpnn-scores` | `auto` | `auto` (on when `--w-mpnn-score` is nonzero) / `on` / `off` |
| `--w-mpnn-score` | 0.0 | ProteinMPNN weight. Score is a negative log probability, so LOWER is better and the weight should be NEGATIVE |
| `--mpnn-n-eval` | 10 | Decoding orders averaged; the score is stochastic |
| `--w-ddg` | 0.0 | ThermoMPNN ddG weight. Positive ddG = destabilizing, so use a NEGATIVE weight for stable designs |
| `--ddg-chain-reduction` | `mean` | Combine chain copies: `mean` / `middle` / `sum` |
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
- **Fitness**: a weighted sum over every metric below, `w_plddt * pLDDT - w_rmsd * RMSD` by default with all other weights 0 (higher is better; a metric passed as `None` drops out of the sum). Everything past `w_rmsd` in `compute_fitness` is keyword-only: each added metric used to shift the positional arguments of the ones after it, silently reassigning a caller's weights
- **Multimer confidences**: `ptm`, `i_ptm` and `i_pae` come out of the same AF2 forward pass as pLDDT and were previously discarded by `predict.py`, so recording them costs nothing. For a fibril `i_ptm` is the more meaningful confidence -- the structure *is* its inter-chain stacking -- and native PHF tau scores `i_ptm` 0.093 against pLDDT 0.235, i.e. AF2 misses the interface even harder than the fold. Bennett et al. 2023 found interface PAE the best in-silico filter for designed interfaces. **ColabDesign divides PAE by 31 A** (`af/loss.py`), so `i_pae` is in [0,1]; `i_pae_angstrom` is recorded for comparison against the literature's ~10 A thresholds (native: 27.8 A)
- **Confidence vs fidelity**: pLDDT/pTM/ipTM/iPAE are AF2's opinion of itself and need no reference, so they can be high for a confidently wrong shape. `shape.py` adds the reference-based half -- TM-score (iterative superposition search, so the >0.5 same-fold threshold applies), lDDT (superposition-free, and the quantity pLDDT predicts), Fnat (fraction of the reference's inter-chain CA contacts recovered), plus per-chain variants. pure numpy on coordinates already in hand. Cost is not constant -- the TM-score refinement iterates more as structures converge, so it is ~250 ms at the current starting point (TM 0.10), ~530 ms at TM 0.58, ~150 ms on native itself: succeeding makes it more expensive
- **Why the decomposition earns its cost**: on native PHF tau, global RMSD 34.97 A says only "not similar", while per-chain RMSD 20 A / per-chain lDDT 0.443 / global lDDT 0.127 / Fnat 0.0009 localize it -- the monomer C-shape is wrong *and* there is no stacking. The 6ELM monomer reaches TM-score 0.648 through the same pipeline, so the PHF failure is target-specific, not a general AF2 failure
- **Sequence homology**: `homology.py`, ~37 us, no model. `seq_recovery` is ProteinMPNN's own quantity, verified against `protein_mpnn_run.py` (one-hot inner product of native and designed over designable positions; equals plain identity for a fully designable homo-oligomer, to float32 rounding). `blosum62` follows Gadhe et al.'s "blosum62 distances to the WT sequence", but they give no formula, so the mean-per-position convention is ours and their numbers are not reproduced. Measured against the target's **native** sequence (`data/<target>_sequence.txt`), not `--initial-seq`. Neither has an imposed direction: staying near native or moving away are both legitimate objectives
- **Why `mpnn_identity` is gone**: it named where the quantity was computed rather than what it was -- it needs no model at all -- and being returned by `MPNNScorer` meant it vanished when ProteinMPNN was switched off. It is `seq_recovery` in `homology.py` now, always computed
- **Metric presence**: a metric that does not apply is absent, not a placeholder. ipTM/iPAE/Fnat only exist for `copies > 1`; ddG only when `models/ddg/` has a matrix for the target; ProteinMPNN and amyloid/LLPS behind their `auto/on/off` flags. `results.json` records non-finite floats as `null` (`_convert` in run_search.py) because `json.dump` would emit a bare `NaN` that strict parsers reject
- **Naming rule**: every weight flag is `--w-` plus its record key with `_` replaced by `-` (`tm_score` -> `--w-tm-score`). `predict.py` renames ColabDesign's `i_ptm`/`i_pae` to `iptm`/`ipae` to keep that rule total
- **Fnat is not DockQ**: DockQ defines Fnat over all heavy atoms at 5 A, which is not comparable between sequences with different side chains. Here it is CA pairs at 8 A. Do not report these as DockQ values
- **Interface metrics need an interface**: `i_ptm`/`i_pae`/`i_pae_angstrom` are emitted only for `copies > 1` (ColabDesign adds `i_pae` to the losses only then, and pops `i_ptm` from its own log for a single chain because it returns a meaningless 0.0), and `fnat` is `nan` for one chain
- **Weight 0 skips the term rather than multiplying by zero**: `0.0 * nan` is `nan`, which combined with the non-finite-fitness guard would abort a run over a metric nobody asked to optimize
- **Sequence scores**: ESM2-3B (layer 36, mean-pooled) -> logistic regression, per Lobo et al., PNAS 123:e2531932123 (2026). ESM2 is loaded once and shared by both heads, the windows of a candidate are batched together, and results are cached per sequence. Agrees with the upstream `amyloid-predict` / `llps-predict` CLIs to <1e-6 on the three values tested (VQIVYK via the 6aa and general heads, the tau 73mer via the LLPS head) -- re-check after changing batching, dtype or device, since the equivalence is empirical, not structural
- **Recorded metrics**: `amyloid_mean` and `amyloid_max` (likewise for LLPS) are always both written to every record, and `amyloid`/`llps` hold the aggregate the fitness used (`--amyloid-agg`; the LLPS term always uses the mean, which equals the score under whole-sequence LLPS scoring). Selecting one aggregate must never cost the other, since a rerun means redoing every ESM evaluation
- **CLI surface**: only the knobs that change results or unblock a machine are flags. LLPS windowing, the amyloid classifier policy and the checkpoint directory are `ESMScorer` constructor arguments, whose defaults are the lengths and heads the published classifiers were trained for
- **Windowing**: the amyloid heads were trained on 6/10/15 aa peptides, so longer sequences are scored by sliding windows, averaging window scores onto residues, then `mean`/`max` over the profile. The LLPS head was trained on whole IDRs, so it scores the sequence in one shot by default
- **AGGRESCAN**: reimplemented from Conchillo-Sole et al., BMC Bioinformatics 8:65 (2007); a3v scale from its additional file 1, `HST = -0.02` from additional file 2. ~80 us per candidate, so always computed, never gated. NOT the official server, and two details are unreproducible from the publication: the charge correction applied to the 2 extreme residues is never given (they inherit the nearest window centre here), and the areas are specified as "trapezoidal integration (midpoint rule)", which names two different methods (`aat`/`ta`/`thsa` are the trapezoid of the threshold-clipped profile). a3v, a4v and hot-spot detection are checked against hand computation; the areas are not, since there is nothing unambiguous to check them against. Values must not be reported as server output
- **AGGRESCAN scale trap**: `na4vss` is a sum of order -13, not a probability, so `--w-aggrescan 1.0` outweighs the whole pLDDT term thirteen times over. Sensible weights start around 0.01-0.05. `nhs`/`nnhs` are recorded but excluded from `METRICS`: over a fixed-length target they take a handful of distinct values (three across all 1387 single mutants of the 73-residue PHF core) and the steps are threshold noise. Every hot spot carries a `margin` for exactly that reason -- on native PHF tau one hot spot clears `HST` by 0.50 and the other by 0.0026
- **TANGO / WALTZ**: deliberately absent. TANGO ships as a licensed compiled binary (no source), WALTZ is web-server only and its score depends on a FoldX-derived matrix. AGGRESCAN stands in for TANGO's role and the `6aa` head for WALTZ's; neither reproduces the original numbers, so do not compare results here against published TANGO/WALTZ values
- **GPU memory**: ESM2-3B moves onto the GPU at the first scored candidate and peaks at ~11.1 GiB there (measured, default batching, RTX A6000) on top of AF2's allocation, so `run_search.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` when the scores are on and ESM2 is on the GPU (before importing `predict`); set the variable yourself to override. Cost per candidate ~3 s against ~10 s for the AF2 prediction; two ESM forward passes at the default `--esm-toks-per-batch 4096`, three at 1024
- **ddG**: ThermoMPNN (Dieckhaus et al., PNAS 121:e2314851121, 2024; MIT) site-saturation is precomputed once per target by `precompute_ddg.py` into `models/ddg/<prefix>_thermompnn.npz` (~27 kB, committed); the search only sums table entries (~7 us/candidate, no torch). **Positive ddG = destabilizing**, so `--w-ddg` must be NEGATIVE to favour stable designs; native is exactly 0. `precompute_ddg.py` loads ThermoMPNN's `TransferModel` directly rather than through its Lightning wrapper, which keeps pytorch-lightning, wandb, omegaconf and tqdm out of the dependency set -- `_Cfg` there is a 6-field stand-in for its OmegaConf cfg, and `load_state_dict` is strict so a hyperparameter mismatch fails loudly
- **ddG gaps are fatal, loudly**: `precompute_ddg.py` leaves NaN where the structure does not model a residue, and a NaN fitness makes both branches of the Metropolis criterion false -- every candidate rejected, run reports success, nothing searched. `ddg.py` therefore raises on a NaN lookup, `precompute_ddg.py` warns with a NaN count, and `SequenceEvaluator` refuses any non-finite fitness whatever produced it. The shipped 5O3L matrix has 0 NaN of 7300; 6GX5 (275-305 unobserved) is the kind of target that would trip it
- **ddG chain reductions**: `mean` (default) and `middle` are interchangeable here (correlation 0.98), but `sum` multiplies the term by the chain count -- 8.08 vs 1.62 for I3D on the 5-chain target -- so a weight tuned under one is wrong under the other. `middle` takes index `n//2`, which for an even chain count is the upper of the two central chains
- **ddG limits**: a k-mutation candidate is the sum of k single-mutant predictions (no epistasis, error grows with k); ThermoMPNN was trained on Megascale globular domains, so a fibril core is out of distribution; and every number is for a mutation of the reference backbone, not of the candidate's predicted structure. For an academic licence, FoldX or PyRosetta is the more defensible physics-based route
- **ProteinMPNN score**: `mean(-log p(residue | backbone))` from the ProteinMPNN bundled in ColabDesign (`v_48_020`), so no new dependency. Lower is better -> `--w-mpnn-score` must be NEGATIVE. Native scores 0.588 on the PHF reference; poly-alanine 0.733, native reversed 0.949, poly-tryptophan 1.246, so the usable span is ~0.6 (poly-alanine alone would suggest ~0.15 and underestimate it). Stochastic: SD 0.019 at n_eval=1, 0.006 at 10, ~18 ms per evaluation. Differences below ~0.01 are noise. The isolated 180 ms at n_eval=10 understates the in-run cost -- a full run went from ~11.9 s to ~13 s per step, untraced
- **ProteinMPNN gotchas found by measurement**: (a) ColabDesign's score differs from the original PyTorch tool by 0.0084 on the same input, which is about 2x the standard error of the 30-order average being compared (~0.004) and therefore a resolvable systematic difference, not noise -- the 0.021 figure is the spread of a *single* evaluation and must not be used as the comparison scale; cause unidentified, and it is NOT the 20-vs-21 letter normalization; (b) ColabDesign orders residues as AlphaFold does (`ARNDCQEGHILKMFPSTWYV`), not as ProteinMPNN does, so decoding `_inputs["S"]` with the wrong table yields a wrong-but-plausible reference sequence; (c) `score()` overwrites `_inputs["S"]` in place because `copy_dict` shares leaf arrays, which is why its `seqid` is always 1.0 and why `MPNNScorer` snapshots the reference before the first call
- **Homo-oligomers only**: ColabDesign's `_prep_hallucination` takes a scalar `length` and hardcodes `get_multi_id(..., homooligomer=True)`, so hetero-oligomers cannot be searched. Two metrics encode that assumption: `min_permutation_rmsd` permutes all n_chains! chains (valid only when they share a sequence), and `precompute_ddg.py` refuses chains that differ. Both are correct for homo and fail loudly rather than silently for hetero
- **MC acceptance**: Metropolis criterion `exp(delta_fitness / temperature)`
- **PDB saving**: `model.save_pdb(filename=None, get_best=False)` returns PDB string (note: `save_current_pdb` has a missing `return` in ColabDesign)
- **Reference files**: `data/<pdb_id>_<chains>_ca_coords.npy`, `data/<pdb_id>_<chains>_sequence.txt`

## Development

- Package management via `uv`
- Python >= 3.10
- `data/`, `params/`, `structures/`, `results.json` are in `.gitignore`
