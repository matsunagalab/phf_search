# PHF Search

Sequence optimization for protein structures using AlphaFold2 as a black-box evaluator. The current implementation uses Metropolis Monte Carlo, but the codebase is structured so you can **replace the search strategy** (e.g., with a genetic algorithm) without touching the prediction or evaluation layers.

## How It Works

The evaluator is [AlphaFold2](https://github.com/google-deepmind/alphafold) (AF2), a deep learning model that predicts a protein's 3D structure from its amino acid sequence. We access AF2 through [ColabDesign](https://github.com/sokrypton/ColabDesign), which provides a differentiable interface to the model. For each candidate sequence, AF2 runs a forward pass and outputs a predicted 3D structure along with per-residue confidence scores. This prediction is the bottleneck (~seconds per evaluation on GPU), making it a classic expensive black-box optimization setting.

## The Optimization Problem

| Aspect | Details |
|--------|---------|
| **Search space** | Strings of length *L* over a 20-letter alphabet (amino acids) |
| **Objective** | Maximize `fitness = w_plddt * pLDDT - w_rmsd * RMSD + w_amyloid * amyloid + w_llps * LLPS + w_aggrescan * AGGRESCAN + w_ddg * ddG + w_mpnn * mpnn` |
| **Evaluation cost** | ~seconds per call on GPU (AlphaFold2 forward pass) |

## What the metrics mean

No biology background needed. Two come from the 3D structure AF2 predicts for a
candidate. Three more are read off the sequence, far more cheaply, and say
nothing about that predicted structure -- the last of them, ddG, is scored
against the *reference* structure instead.

| Metric | Range | Direction | What it asks |
|--------|-------|-----------|--------------|
| **pLDDT** | 0 to 1 | higher | "How sure is AF2 that the shape it just predicted is right?" |
| **RMSD** | 0 to inf (angstroms) | lower | "How far is that predicted shape from the experimental target?" |
| **amyloid** | 0 to 1 | higher | "Does this sequence read like one that forms amyloid fibrils?" |
| **LLPS** | 0 to 1 | usually lower | "Does it read like one that condenses into liquid droplets instead?" |
| **AGGRESCAN** | unbounded | higher | "Do its residues add up to an aggregation-prone stretch?" -- the same question as *amyloid*, asked by twenty numbers instead of a neural network |
| **ddG** | kcal/mol | **lower** | "Would these substitutions make the target structure less stable?" Positive = destabilizing |
| **mpnn** | ~0 to 3 | **lower** | "Would an inverse-folding model have proposed this sequence for this backbone?" |

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

**AGGRESCAN** answers roughly the same question as *amyloid*, but by a method
old enough and simple enough to read in one sitting: each residue carries an
experimentally measured propensity value, a sliding window averages them, and
stretches that stay above a fixed threshold are called hot spots. It has **no
decision boundary** -- the default summary (`na4vss`) is a sum, so only its
relative ordering means anything. Keeping it alongside the language-model score
is the point: when a transparent index and a learned one disagree about the same
sequence, that disagreement is information about the indices.

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
| **AGGRESCAN** | **>= ~-13.5** (`na4vss`) | Second opinion on the amyloid score, from a completely different method. Read it relative to native, never as an absolute |
| **mpnn** | **<= ~0.59** | Is the sequence at least as plausible for this backbone as native? Native scores 0.588; a scrambled or homopolymer sequence runs 0.73-1.25 (see below). Read differences below ~0.01 as noise |
| **ddG** | **<= 0** | Does the design hold the reference fold together at least as well as native? Native is 0 by definition, so any positive value is a design that ThermoMPNN thinks destabilizes the filament |

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
| AGGRESCAN `na4vss` | -13.50 | one solid hot spot at residues 1-6 (VQIVYK), plus one that is an artifact -- see below |
| ProteinMPNN `mpnn_score` | 0.588 | for scale: poly-alanine 0.733, the native sequence reversed 0.949, poly-tryptophan 1.246 |

Two things to read carefully in that last row.

**The agreement is meaningful.** AGGRESCAN and the language-model amyloid score
share no method, no training data and no decade, and they independently single
out **VQIVYK** as the most aggregation-prone window of this construct. Where two
such different indices agree, the signal is probably in the sequence rather than
in the index.

**The hot-spot count is not.** AGGRESCAN reports two hot spots here, but they
are not comparable findings. VQIVYK clears the threshold by **0.50**; the second
one, residues 9-13 (DLSKV), clears it by **0.0026** -- about 190 times less.
Moving `HST` from -0.02 to -0.015 deletes it and turns `nhs` from 2 into 1. It
is an artifact of where the threshold happens to fall, not a property of the
sequence, which is why `nhs` is reported but not offered as an objective, and
why `aggrescan.score()` returns a `margin` for every hot spot. Check that margin
before believing a hot spot.

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
--w-aggrescan W      Weight for AGGRESCAN; negative penalizes (default: 0.0)
--aggrescan-metric M na4vss | a3vsa | thsar (default: na4vss)
--w-ddg W            Weight for ThermoMPNN ddG. Use a NEGATIVE weight to favour
                     stable designs, since positive ddG is destabilizing (default: 0.0)
--ddg-chain-reduction R  mean | middle | sum over the chain copies (default: mean)

Sequence-level scores (language model; AGGRESCAN is always on):
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

## Sequence-level scores (amyloid, LLPS, AGGRESCAN)

### AGGRESCAN

`aggrescan.py` is a reimplementation of the published algorithm
([Conchillo-Sole et al. 2007](https://doi.org/10.1186/1471-2105-8-65)), whose
per-residue scale was measured *in vivo* from the intracellular aggregation of
amyloid-beta central-hydrophobic-cluster mutants
([Sanchez de Groot et al. 2005](https://doi.org/10.1186/1472-6807-5-18)). Each
residue carries a propensity value (a3v), a length-dependent sliding window
averages them into a profile (a4v), and a run of 5 or more residues above the
threshold `HST = -0.02` containing no proline is a hot spot.

It costs **~80 microseconds** per candidate -- against ~10 s for AF2 -- so it is
always computed and always recorded, with no flag to switch it on. Each record
carries `aggrescan_a3vsa`, `aggrescan_na4vss`, `aggrescan_nhs`,
`aggrescan_nnhs`, `aggrescan_aatr`, `aggrescan_thsar` and `aggrescan_ta`. Of
those, `--aggrescan-metric` can send `na4vss`, `a3vsa` or `thsar` to the
fitness, weighted by `--w-aggrescan`. The per-residue profile and the
per-hot-spot detail stay out of the records but are returned by
`aggrescan.score()` for analysis, along with `a4vss`, `aat`, `thsa`, `a4vahs`
and `nhsa` -- the unnormalized quantities are omitted from records only because
they are the recorded ones times the (fixed) sequence length.

**Mind the scale when weighting it.** Unlike the two probabilities, `na4vss` is
a sum of order **-13**, so `--w-aggrescan 1.0` contributes about thirteen times
what the entire pLDDT term can and roughly a third of the RMSD term. Chosen by
analogy with `--w-amyloid 1.0`, it would quietly take over the objective.
Start two orders of magnitude lower, around `0.01`-`0.05`.

`nhs`/`nnhs` are recorded but cannot steer the search: over the 73-residue PHF
target, counting hot spots takes only three distinct values across all 1387
single mutants, so as an objective it is a step function whose steps are
threshold noise.

**This is not the official AGGRESCAN server.** The scale, window rule,
threshold, hot-spot rule and area definitions all come from the paper and its
additional files, but the server assigns the extreme terminal residues an
unspecified value "to account for charge effects" that the publication does not
give. Here those residues inherit the nearest window centre, so profiles can
differ at the ends. Do not report these numbers as server output.

### ddG (ThermoMPNN)

[ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) (Dieckhaus et al., *PNAS*
**121**, e2314851121, 2024; MIT) predicts the stability change of a point
mutation from structure. It scores every single mutation of a structure in one
pass -- 7300 mutations over the 5-chain PHF reference in ~4 s -- so it never
runs inside the search. `precompute_ddg.py` writes the whole
`(n_chains, n_residues, 20)` matrix to `models/ddg/` once, and `ddg.py` adds up
the entries for the positions a candidate changed: **~7 microseconds**, no
torch, no 39 MB checkpoint. The matrix for the default target is committed
(27 kB), so nothing needs installing; other targets need one run of
`precompute_ddg.py` against a ThermoMPNN checkout.

```bash
git clone https://github.com/Kuhlman-Lab/ThermoMPNN
uv run python precompute_ddg.py --thermompnn-dir ThermoMPNN     # once, per target
uv run python run_search.py --n-steps 100 --w-ddg -0.5          # negative weight!
```

**Positive ddG is destabilizing**, so a search after stable designs wants a
**negative** `--w-ddg`. Native scores exactly 0 -- it is the reference.

Three limits, all real:

1. **Additivity.** A candidate differing at *k* positions is scored as the sum
   of *k* independent single-mutant predictions. Epistasis is invisible and the
   error grows with *k*: treat it as a ranking signal among near neighbours, not
   a free energy. A sequence 30 mutations out is past what this can honestly
   estimate.
2. **Out of distribution.** ThermoMPNN was trained on Megascale -- small
   globular domains measured by proteolysis. A fibril core is held together by
   inter-chain stacking, and "folding stability" is not the same quantity. It is
   used here because it is cheap and structure-aware, not because it was
   validated on fibrils. (This is also why the physics-based route, FoldX or
   Rosetta, is the more defensible one when a licence is available.)
3. **Fixed reference backbone.** Every number describes a mutation of the
   reference structure, not of the structure AF2 predicts for the candidate.

The matrix is built on the full assembly, so inter-chain contacts are in the
model's input: on 5O3L, I3D scores +0.36 against a lone chain and **+1.62**
under the default `mean` reduction over the five stacked copies.

`--ddg-chain-reduction` picks how those copies combine. They are not equivalent
-- per chain that same mutation gives [1.34, 1.60, 1.76, 1.68, 1.70], lowest at
the two ends of the stack, which have one neighbour rather than two. `mean` and
`middle` differ little (the middle chain correlates 0.98 with the mean), but
**`sum` is on a different scale entirely**: it multiplies the term by the chain
count, so 8.08 instead of 1.62 here, and a weight tuned under `mean` becomes
five times stronger.

### ProteinMPNN score

The inverse-folding model's own opinion of a sequence: the mean negative log
probability it assigns, given the target coordinates. **Lower is better**, so
`--w-mpnn` must be **negative** to reward sequences ProteinMPNN likes.

It answers a genuinely different question from the rest. RMSD asks whether AF2
folds the sequence onto the target; the amyloid indices ask whether the sequence
reads like an aggregator; this asks whether a design model would have *proposed*
the sequence for this backbone -- which is the criterion the sequences were
generated under in the first place.

```bash
uv run python run_search.py --n-steps 100 --mpnn-scores on          # record only
uv run python run_search.py --n-steps 100 --w-mpnn -1.0             # negative weight!
```

This comes from the ProteinMPNN bundled inside ColabDesign, using the
`v_48_020` weights -- already a dependency, so no new package and no new
checkpoint.

**Know the scale.** Measured on the PHF reference at `n_eval=5`:

| sequence | `mpnn_score` |
|---|---|
| native | 0.588 |
| poly-alanine | 0.733 |
| poly-glycine | 0.791 |
| poly-aspartate | 0.836 |
| native reversed | 0.949 |
| poly-proline | 1.136 |
| poly-tryptophan | 1.246 |

So the usable span is roughly **0.6**. Poly-alanine is a mild perturbation, not
an extreme -- calibrating a weight against it alone would underestimate how hard
this term pulls once a search wanders somewhere genuinely implausible.

The score is also stochastic, because ProteinMPNN decodes in a random order:

| `--mpnn-n-eval` | spread (SD) | cost per candidate |
|---|---|---|
| 1 | 0.019 | ~18 ms |
| 3 | 0.013 | ~53 ms |
| 10 (default) | 0.006 | ~180 ms |

At `n_eval=1` the noise is larger than the effect of most single mutations, so
the search would be optimizing the random number generator. Even at the default,
treat a difference below ~0.01 as noise.

Those costs are measured in isolation. In a full run the per-step time went from
~11.9 s to ~13 s with this term on, i.e. closer to +1 s than the +0.18 s the
table implies; the extra was not traced (GPU contention or JAX recompilation are
both plausible). Either way it is a small fraction of the AF2 call.

Also recorded is `mpnn_identity`, the fraction of positions still matching the
reference sequence -- a plain drift counter. ColabDesign's own `seqid` output is
not used: it compares the scored sequence against itself and is always 1.0.

### Why not TANGO and WALTZ?

The amyloid design literature -- including
[Gadhe et al. 2026](https://doi.org/10.64898/2026.05.08.723915), who did exactly
this kind of ProteinMPNN design on alpha-synuclein fibrils -- characterizes
designs with TANGO and WALTZ. Neither can be bundled here:

- **TANGO** is distributed as a compiled binary under a licence agreement
  ("we do not provide or sell source code"). Free for academic use, but it has
  to be requested, and it cannot ship with this repository.
- **WALTZ** is web-server only. Its score combines a PSSM, 19 physicochemical
  properties and a structural pseudo-energy matrix derived with FoldX, which is
  itself licensed, so a faithful local reimplementation is not realistic either.

What fills their roles here, and how honestly:

| Their tool | Measures | Stand-in here | Caveat |
|-----------|----------|---------------|--------|
| TANGO | generic hydrophobic beta-sheet aggregation | **AGGRESCAN** (`aggrescan.py`) | different method and scale; same question |
| WALTZ | sequence-specific amyloid motifs | **the `6aa` amyloid head** (already in `models/esm_heads/`) | trained on the WALTZ hexapeptide benchmark, so it learned from WALTZ's data -- but it is a language-model classifier, **not** WALTZ |

Neither substitution reproduces the original numbers, so results here are not
directly comparable to published TANGO or WALTZ values. If you need the real
ones, the web servers ([tango.crg.es](https://tango.crg.es/),
[waltz.switchlab.org](https://waltz.switchlab.org/)) take a sequence or a FASTA
and are fine for a handful of selected designs -- which is how Gadhe et al. used
them, as post-hoc characterization of already-generated sequences rather than
inside the design loop.

### amyloid and LLPS

How the two language-model scores are computed. Both come from
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
