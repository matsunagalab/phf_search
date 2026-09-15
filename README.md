# PHF Search

Sequence optimization for protein structures using AlphaFold2 as a black-box evaluator. The current implementation uses Metropolis Monte Carlo, but the codebase is structured so you can **replace the search strategy** (e.g., with a genetic algorithm) without touching the prediction or evaluation layers.

## How It Works

The evaluator is [AlphaFold2](https://github.com/google-deepmind/alphafold) (AF2), a deep learning model that predicts a protein's 3D structure from its amino acid sequence. We access AF2 through [ColabDesign](https://github.com/sokrypton/ColabDesign), which provides a differentiable interface to the model. For each candidate sequence, AF2 runs a forward pass and outputs a predicted 3D structure along with per-residue confidence scores. This prediction is the bottleneck (~seconds per evaluation on GPU), making it a classic expensive black-box optimization setting.

## The Optimization Problem

| Aspect | Details |
|--------|---------|
| **Search space** | Strings of length *L* over a 20-letter alphabet (amino acids) |
| **Objective** | Maximize `fitness = w_plddt * pLDDT - w_rmsd * RMSD + w_amyloid * amyloid + w_llps * LLPS` |
| **Evaluation cost** | ~seconds per call on GPU (AlphaFold2 forward pass) |

## What the metrics mean

No biology background needed. Two metrics come from the 3D structure AF2
predicts for a candidate and are always computed; two are read off the sequence
alone by a separate, much cheaper model.

| Metric | Range | Direction | What it asks |
|--------|-------|-----------|--------------|
| **pLDDT** | 0 to 1 | higher | "How sure is AF2 that the shape it just predicted is right?" |
| **RMSD** | 0 to inf (angstroms) | lower | "How far is that predicted shape from the experimental target?" |
| **amyloid** | 0 to 1 | higher | "Does this sequence read like one that forms amyloid fibrils?" |
| **LLPS** | 0 to 1 | usually lower | "Does it read like one that condenses into liquid droplets instead?" |

**pLDDT** is the predictor's own confidence, averaged over residues. Above ~0.8
AF2 is asserting a definite fold; below ~0.5 it is effectively saying "I don't
know", and the coordinates it returns should not be trusted.

**RMSD** is the distance between predicted and target coordinates after optimal
superposition, in angstroms. Below ~2 A is the same structure; below ~5 A is
recognizably the same fold; above ~15 A the two have nothing in common.

**amyloid** and **LLPS** are probabilities from binary classifiers, so **0.5 is
the decision boundary**: above 0.5 the sequence is on the "forms amyloid" /
"drives phase separation" side, below 0.5 on the other. They describe the
*sequence*, not the predicted structure, which is what makes them worth having
-- a sequence can land on the target coordinates in AF2's hands while reading
nothing like a real amyloid former.

### What counts as a good value

The goal of this project is concrete: **find a sequence whose AF2 prediction is
the PHF filament structure, in as few evaluations as possible.** The metrics
serve that goal in different roles, so their target values are not symmetric.

| Metric | Aim for | Role in this goal |
|--------|---------|-------------------|
| **RMSD** | **< 5 A**, ideally < 2 A | **The success criterion.** It is the only metric that directly answers "is this the PHF fold?" |
| **pLDDT** | **> 0.8** | **Necessary companion.** A low RMSD with pLDDT ~0.3 means AF2 is not actually asserting that fold, so the match is not to be trusted |
| **amyloid** (max) | **>= ~0.99** | **Plausibility check.** The target is an amyloid fibril; a hit whose amyloidogenic core has been mutated away is suspect even if AF2 likes it |
| **amyloid** (mean) | **>= ~0.23** | Keep it at or above the real filament sequence; there is no reason to push it toward 1.0 (see below) |
| **LLPS** | **<= ~0.50** | Droplets are the competing fate of tau. Lower than native steers toward the fibril -- a hypothesis you choose to encode, not an established correction |

**Where you start.** The same metrics, measured here on the native PHF tau
sequence itself (5O3L, 73 residues) -- the sequence that does form the target
filament in reality:

| Metric | Native PHF tau | Reading |
|--------|---------------|---------|
| pLDDT | 0.23 | AF2 has no confidence in *any* fold for this sequence |
| RMSD | 35 A | the predicted shape is nothing like the real filament |
| amyloid (mean) | 0.23 | most of the sequence is not amyloidogenic on its own |
| amyloid (max) | 0.996 | but it holds an almost maximally amyloidogenic window (VQIVYK, residues 1-6) |
| LLPS | 0.50 | right on the boundary; tau is known to do both |

Three consequences worth internalizing before tuning a search:

1. **AF2 fails on the correct answer.** Given the very sequence that builds the
   PHF filament, this pipeline returns 35 A and pLDDT 0.23. The search is
   therefore not hunting for the native sequence -- it is hunting for a sequence
   AF2 *will* fold onto the PHF coordinates. That gap is the research problem,
   not a misconfiguration.
2. **With default weights, you start at fitness -34.7 and success is above
   about -4.2.** (`pLDDT 0.8 - RMSD 5.0 = -4.2`; native gives
   `0.23 - 34.97 = -34.74`.) RMSD dominates the objective by an order of
   magnitude, so in practice the search is an RMSD minimizer with a confidence
   tiebreaker.
3. **A mean amyloid score near 1.0 is the wrong target.** The real filament
   sequence averages 0.23, because amyloidogenicity is carried by a few short
   stretches rather than spread evenly. Demanding a high mean would push the
   search away from realistic sequences -- which is why `--amyloid-agg max`
   (is there a strong stretch?) usually asks the better question than the mean
   (is it uniformly sticky?).

**On "in as few evaluations as possible".** The sequence scores cost ~3 s
against AF2's ~10 s and need no structure at all, so they can also be read as a
cheap surrogate: a candidate that has lost its amyloidogenic core is unlikely to
be a PHF hit, and could be rejected before AF2 ever runs. The current
`SequenceEvaluator` does not do this -- it scores AF2 first and always pays for
both -- but it is the obvious place for a search strategy to save wall-clock.

The two sequence terms carry **weight 0 by default**, so the objective is
unchanged until you ask for them. `--esm-scores on` reports them without
letting them steer the search; a nonzero `--w-amyloid` / `--w-llps` puts them in
the objective, and a negative weight penalizes instead of rewards.

## Supported Targets

The pipeline works with any PDB structure. Two targets are pre-configured:

| Target | PDB | Chains | Residues | Type | Description |
|--------|-----|--------|----------|------|-------------|
| PHF tau | [5O3L](https://www.rcsb.org/structure/5O3L) | A,C,E,G,I | 73 | 5-chain homooligomer | Paired helical filament (default) |
| WNK2 CCT1 | [6ELM](https://www.rcsb.org/structure/6ELM) | A | 98 | Monomer | WNK2 kinase CCT1 domain |

To add a new target, run `prepare_reference.py` with its PDB ID and chains (see below).

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync --extra cpu    # CPU-only (for testing)
uv sync --extra cuda   # GPU (for real runs)

# 2. Download AlphaFold2 model parameters (~3.5 GB)
bash download_params.sh

# 3. Prepare reference target structure(s)
uv run python prepare_reference.py                          # PHF tau (default: 5O3L chains A,C,E,G,I)
uv run python prepare_reference.py --pdb-id 6ELM --chains A # WNK2 CCT1 monomer
```

One `uv sync` installs everything needed for both the structural and the
sequence-level metrics; nothing else has to be pip-installed or cloned. The
`cpu`/`cuda` choice covers PyTorch as well as JAX, so `--extra cpu` stays a
genuinely CPU-only install rather than pulling CUDA wheels it will never use.

Two sets of model weights are fetched separately because of their size: AF2's,
by the script above, and ESM2-3B's (~5.7 GB), downloaded automatically into the
torch cache the first time you run with `--esm-scores on`. The small classifier
heads that turn ESM2 embeddings into the amyloid and LLPS scores ship with the
repository in `models/esm_heads/`.

After setup, you should have:
- `params/` -- AF2 model weights
- `data/5o3l_acegi_ca_coords.npy` -- PHF target coordinates, shape (5, 73, 3)
- `data/5o3l_acegi_sequence.txt` -- PHF native sequence (73 residues)
- `data/6elm_a_ca_coords.npy` -- WNK2 target coordinates, shape (1, 98, 3)
- `data/6elm_a_sequence.txt` -- WNK2 native sequence (98 residues)

## Quick Start

```bash
# PHF tau 5-chain homooligomer (default)
uv run python run_search.py --n-steps 100

# WNK2 CCT1 monomer
uv run python run_search.py --pdb-id 6ELM --chains A --n-steps 100

# Any other PDB target (after running prepare_reference.py for it)
uv run python run_search.py --pdb-id <PDB_ID> --chains <CHAIN_IDS> --n-steps 100
```

Results are saved to `results.json` and predicted structures to `structures/`.

### CLI Options

```
Target selection:
--pdb-id ID          Target PDB ID (default: 5O3L)
--chains IDS         Comma-separated chain IDs (default: A,C,E,G,I)

Overrides:
--ref-coords FILE    Reference CA coordinates .npy (auto-derived from --pdb-id/--chains)
--initial-seq SEQ    Starting sequence (auto-loaded from prepared reference)

Search parameters:
--n-steps N          Number of optimization steps (default: 1000)
--temperature T      MC temperature for acceptance (default: 1.0)
--n-mutations N      Mutations per step (default: 1)
--num-recycles N     AF2 recycles (default: 3)
--w-plddt W          Weight for pLDDT in fitness (default: 1.0)
--w-rmsd W           Weight for RMSD in fitness (default: 1.0)

Sequence-level scores:
--esm-scores MODE    auto | on | off (default: auto = on if a weight is nonzero)
--w-amyloid W        Weight for amyloidogenicity; negative penalizes (default: 0.0)
--w-llps W           Weight for LLPS propensity; negative penalizes (default: 0.0)
--amyloid-agg A      Which aggregate enters the fitness: mean | max (default: mean)
--amyloid-probe-lengths L   Window lengths, or 'none' for whole sequence (default: 6,15)
--esm-cpu            Run ESM2 on CPU instead of GPU
--esm-toks-per-batch N      ESM2 batch size in tokens; lower on OOM (default: 4096)

Output:
--log-interval N     Log every N steps (default: 10)
--save-interval N    Save PDB structure every N steps (default: 1)
--structures-dir D   Directory for PDB files (default: structures)
--output FILE        Output JSON path (default: results.json)
```

## Sequence-level scores (amyloid and LLPS)

How the two sequence scores are computed. Both come from
[Lobo et al. (2026)](https://doi.org/10.1073/pnas.2531932123) and work the same
way: embed a peptide with ESM2-3B (layer 36, mean-pooled over residues), then
push the 2560-dim vector through a logistic regression trained on experimental
data.

| Score | Trained to separate |
|-------|--------------------|
| **amyloid** | peptides that form amyloid from ones that do not (WALTZ-style hexapeptides, 10 aa and 15 aa fragments including tau) |
| **LLPS** | IDRs that drive phase separation from ones that do not (CD-CODE / IDRome) |

Those training sets are short peptides and whole IDRs respectively, so a
73-residue candidate is not handed to both whole. The amyloid score slides
6-residue and 15-residue windows along the sequence (`--amyloid-probe-lengths`),
scores each with the head trained at that length, averages the window scores
back onto the residues they cover, and reduces that per-residue profile to its
mean or its peak (`--amyloid-agg`). The LLPS head, trained on entire IDRs, scores
the sequence in one shot.

```bash
# Report both scores every step without changing what is optimized
uv run python run_search.py --n-steps 100 --esm-scores on

# Reward amyloidogenicity and penalize phase separation
uv run python run_search.py --n-steps 100 --w-amyloid 1.0 --w-llps -1.0

# Score the single most amyloidogenic window instead of the average
uv run python run_search.py --n-steps 100 --w-amyloid 1.0 --amyloid-agg max
```

**Cost.** ESM2-3B is loaded once and shared by both heads, the windows of a
candidate are batched by length rather than scored one at a time (two forward
passes for the default configuration, three at `--esm-toks-per-batch 1024`), and
repeated sequences are cached. Measured on an RTX A6000 for the 73-residue PHF
target: **~3 s per candidate**, alongside the ~10 s its AF2 prediction takes.

Memory is the thing to watch. ESM2-3B moves onto the GPU at the first scored
candidate and peaks around **11.1 GiB** there under the default batching, on top
of AF2's own allocation. `run_search.py` therefore sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` when the scores are on *and* ESM2 is on
the GPU, so JAX leaves room; set the variable yourself to override.
`--esm-toks-per-batch` lowers the peak, and `--esm-cpu` takes ESM2 off the GPU
entirely at a large speed cost.

**What lands in `results.json`.** Every record carries `amyloid_mean` and
`amyloid_max` (likewise `llps_mean`, `llps_max`) regardless of which aggregate
you optimize, plus `amyloid`/`llps` holding the one the fitness used. Choosing
`--amyloid-agg max` therefore does not cost you the mean: both questions above
stay answerable from a finished run, which matters because recovering the other
one means repeating every ESM evaluation.

**Weights and citation.** The classifier heads are vendored in
`models/esm_heads/`; their provenance, licensing and citation are documented in
[`models/esm_heads/README.md`](models/esm_heads/README.md). If you publish
numbers that use them, cite Lobo et al., *PNAS* **123**, e2531932123 (2026).

## Architecture Overview

```
run_search.py          CLI entry point (--pdb-id, --chains)
    |
    v
mc_search.py           Search strategy (THE PART YOU REPLACE)
    |
    v
evaluate.py            SequenceEvaluator: sequence -> every metric + fitness
    |
    +-> predict.py     AF2 black-box evaluator (AF2Predictor, DO NOT MODIFY)
    +-> utils.py       RMSD computation + mutation operator
    +-> esm_scores.py  ESM2-3B amyloid / LLPS scores
    +-> fitness.py     Objective function

prepare_reference.py   Extract reference from any PDB
```

The key design: **`predict.py` is an expensive black-box function.** You give it a sequence string, it returns scores and 3D coordinates. Your optimizer's job is to explore the sequence space efficiently.

`evaluate.py` sits between the search loop and the metrics so that a new search
strategy inherits all of them: call `evaluator.evaluate(seq)` and you get the
same record, the same fitness and the same logging as the Monte Carlo baseline,
which keeps runs from different strategies comparable.

## Where to Modify

### 1. Replace the search strategy -- `mc_search.py`

This is the main file to replace or rewrite. The current `MonteCarloSearch` class follows a simple interface:

```python
# The evaluator -- call this for any candidate sequence
result = self.evaluator.evaluate(seq)
# -> dict with "plddt", "rmsd", "fitness", "pdb_str", and when the sequence
#    scores are switched on, "amyloid" and "llps"

# The mutation operator -- or write your own
from utils import mutate_sequence
new_seq = mutate_sequence(current_seq, n_mutations=1)
```

To implement a genetic algorithm, you would:
1. Create a new class (e.g., `GeneticAlgorithmSearch`) in a new file or replace `mc_search.py`
2. Take a `SequenceEvaluator` in the constructor and call `evaluate()` on every candidate -- that one call covers AF2, RMSD, the sequence scores and the weighted fitness
3. Wire it up in `run_search.py`

**Sequence representation:** A sequence is a Python string of length *L* where each character is one of `ACDEFGHIKLMNPQRSTVWY` (the 20 standard amino acids). The length *L* is determined by the target structure (e.g., 73 for PHF, 98 for WNK2 CCT1).

### 2. Change the fitness function -- `fitness.py`

Currently a weighted sum:

```python
fitness = w_plddt * plddt - w_rmsd * rmsd + w_amyloid * amyloid + w_llps * llps
```

You can add nonlinear terms, thresholds, or additional objectives here without changing anything else. A metric passed as `None` is dropped from the sum, which is how the sequence terms disappear when they are switched off.

### 3. Change the mutation operator -- `utils.py`

`mutate_sequence(seq, n_mutations)` does uniform random single-point mutations. For a GA, you would add crossover operators here. The alphabet is defined as:

```python
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 letters
```

### 4. The evaluator (you probably don't need to touch this) -- `predict.py`

`AF2Predictor.predict(sequence)` wraps AlphaFold2 via ColabDesign. It takes a sequence string and returns:

| Key | Type | Description |
|-----|------|-------------|
| `plddt` | float (0-1) | Mean prediction confidence |
| `ca_coords` | ndarray (copies, length, 3) | Predicted 3D coordinates |
| `pdb_str` | str | Full atomic structure in PDB format |

Think of this as `f(x) -> score` where `x` is an *L*-dimensional categorical variable. Each call takes a few seconds on GPU.

### 5. Add a new target

```bash
# 1. Prepare reference structure
uv run python prepare_reference.py --pdb-id <PDB_ID> --chains <CHAIN_IDS>

# 2. Run search
uv run python run_search.py --pdb-id <PDB_ID> --chains <CHAIN_IDS> --n-steps 100
```

The pipeline auto-derives sequence length, chain count, and initial sequence from the prepared reference files.

## Output

- `results.json` -- full optimization trajectory (sequence, pLDDT, RMSD, fitness per step; plus `amyloid_mean`, `amyloid_max`, `llps_mean`, `llps_max` when the sequence scores are on)
- `structures/step_0000.pdb` -- predicted 3D structure for the initial sequence
- `structures/step_NNNN.pdb` -- structures at subsequent steps

PDB files can be visualized with [PyMOL](https://pymol.org/), [ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or [Mol*](https://molstar.org/).
