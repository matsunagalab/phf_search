# PHF Search

Sequence optimization for protein structures using AlphaFold2 as a black-box evaluator. The current implementation uses Metropolis Monte Carlo, but the codebase is structured so you can **replace the search strategy** (e.g., with a genetic algorithm) without touching the prediction or evaluation layers.

## How It Works

The evaluator is [AlphaFold2](https://github.com/google-deepmind/alphafold) (AF2), a deep learning model that predicts a protein's 3D structure from its amino acid sequence. We access AF2 through [ColabDesign](https://github.com/sokrypton/ColabDesign), which provides a differentiable interface to the model. For each candidate sequence, AF2 runs a forward pass and outputs a predicted 3D structure along with per-residue confidence scores. This prediction is the bottleneck (~seconds per evaluation on GPU), making it a classic expensive black-box optimization setting.

## The Optimization Problem

| Aspect | Details |
|--------|---------|
| **Search space** | Strings of length *L* over a 20-letter alphabet (amino acids) |
| **Objective** | Maximize a weighted sum of the metrics below; `w_plddt * pLDDT - w_rmsd * RMSD` by default, with every other term at weight 0 |
| **Evaluation cost** | ~seconds per call on GPU (AlphaFold2 forward pass) |

## What the metrics mean

No biology background needed. They fall into four groups, and the distinction
between the first two is the one that matters most:

* **AF2's confidence in its own answer** -- pLDDT, pTM, ipTM, iPAE. These need
  no reference structure, which also means **they can be high for a confidently
  wrong shape, and none of them measures shape fidelity at all.**
* **Similarity to the experimental target** -- RMSD, TM-score, lDDT, Fnat.
  These are the ones that actually say whether the shape was reproduced.
* **Read off the sequence alone** -- amyloid, LLPS, AGGRESCAN. Cheap, and blind
  to the predicted structure.
* **The sequence judged against the reference backbone** -- ddG, mpnn.

| Metric | Range | Direction | Aim for | Native PHF tau | What it asks |
|--------|-------|-----------|---------|----------------|--------------|
| **pLDDT** | 0 to 1 | higher | > 0.8 | **0.23** | "How sure is AF2 that the shape it just predicted is right?" |
| **RMSD** | 0 to inf (A) | lower | < 5 A | **35 A** | "How far is that predicted shape from the experimental target?" |
| **pTM** | 0 to 1 | higher | > 0.8 | 0.133 | "How sure is AF2 of the overall shape, as a predicted TM-score?" |
| **ipTM** | 0 to 1 | higher | > 0.8 | **0.093** | The same restricted to **inter-chain** pairs -- how sure it is of the interface |
| **iPAE** | 0 to 1 (also in A) | **lower** | < ~10 A | 27.8 A | "How large an error does AF2 expect between residues on *different* chains?" |
| **TM-score** | 0 to 1 | higher | **> 0.5** | **0.102** | "Is this the same fold as the target?" Length-normalized; 0.5 is the conventional same-fold threshold |
| **lDDT** | 0 to 1 | higher | > 0.7 | 0.127 | The same question asked **without superimposing anything** -- and the quantity pLDDT claims to predict |
| **Fnat** | 0 to 1 | higher | > 0.5 | **0.0009** | "How many of the target's inter-chain contacts did we recover?" For a fibril: is it stacked at all? |
| **amyloid** | 0 to 1 | higher | max >= ~0.99 | mean 0.23, max 0.996 | "Does this sequence read like one that forms amyloid fibrils?" |
| **LLPS** | 0 to 1 | usually lower | <= ~0.50 | 0.50 | "Does it read like one that condenses into liquid droplets instead?" |
| **AGGRESCAN** | unbounded | higher | >= native | -13.50 | "Do its residues add up to an aggregation-prone stretch?" -- the same question as *amyloid*, asked by twenty numbers instead of a neural network |
| **ddG** | kcal/mol | **lower** | <= 0 | 0 by definition | "Would these substitutions make the target structure less stable?" Positive = destabilizing |
| **mpnn** | ~0.6 to 1.3 | **lower** | <= ~0.59 | 0.588 | "Would an inverse-folding model have proposed this sequence for this backbone?" |

**pLDDT** is the predictor's own confidence, averaged over residues. Above ~0.8
AF2 is asserting a definite fold; below ~0.5 it is effectively saying "I don't
know", and the coordinates it returns should not be trusted.

**RMSD** is the distance between predicted and target coordinates after optimal
superposition, in angstroms. Below ~2 A is the same structure; below ~5 A is
recognizably the same fold; above ~15 A the two have nothing in common.

**pTM, ipTM and iPAE** are the multimer confidences, and they cost nothing --
AF2 produces them in the same forward pass as pLDDT, and this pipeline simply
used to discard them. **ipTM is the one to watch here**: a fibril *is* its
inter-chain stacking, so the confidence in the interface is more to the point
than the confidence in a single chain. The de novo binder-design literature
reaches the same conclusion from the other direction -- Bennett et al. (2023)
found interface PAE to be the single best in-silico predictor of whether a
designed interface works, with pLDDT a weaker filter.

Two traps. ColabDesign divides PAE by 31 A (its largest PAE bin), so `i_pae` is
in [0, 1], not angstroms; `i_pae_angstrom` is recorded alongside it, verified
equal to the mean of the inter-chain block of the raw PAE matrix to within
0.001 A. And this pipeline runs AF2 in **non-multimer** mode, building chains
from `copies` plus a residue-index offset, so `i_ptm` is ColabDesign's
interface-masked quantity rather than AF-Multimer's official ipTM -- the ~10 A
interface-PAE thresholds from the binder-design literature were measured with
AF-Multimer and do not transfer unexamined.

These three exist only when there is an interface: a monomer target records no
`i_ptm`, `i_pae` or `i_pae_angstrom`, and its `fnat` is `nan` rather than a
misleading 0.

**Shape fidelity** (`shape.py`) adds TM-score, lDDT and Fnat, plus per-chain
`tm_score_chain` / `lddt_chain` / `rmsd_chain` so the monomer fold can be judged
apart from the stacking. All of it runs on the CA coordinates already in hand,
with no new dependency.

**The cost is not constant, and it peaks where a working search goes.** The
TM-score refinement iterates more as the structures get closer, so measured on
this target:

| what is being scored | TM-score | `shape.compare` |
|---|---|---|
| the current starting point | 0.10 | ~250 ms |
| native + 2 A of noise | 0.82 | ~340 ms |
| native + 4 A of noise | 0.58 | ~530 ms |
| native itself | 1.00 | ~150 ms |

Against the ~10 s AF2 call that is a few percent either way, but it is worth
knowing that succeeding makes this term more expensive, not less. The
TM-score follows the `TM-score` program's iterative superposition search, so the
0.5 threshold is meaningful; Fnat is defined on CA pairs rather than all heavy
atoms as DockQ defines it, because a designed sequence has different side chains
from the reference and a side-chain contact set would not be comparable -- so
these are not DockQ numbers.

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

The columns above give the target for each metric. What they mean for *this*
goal, where they differ:

* **RMSD and TM-score are the success criteria** -- the only metrics that
  directly answer "is this the PHF fold?". TM-score is the more interpretable of
  the two, because 0.5 is a threshold rather than a number.
* **pLDDT and ipTM are necessary companions, not criteria.** A low RMSD with
  pLDDT ~0.3 means AF2 is not actually asserting that fold. ipTM is the more
  telling of the two for a fibril.
* **amyloid, AGGRESCAN and ddG are plausibility checks.** The target is an
  amyloid fibril, so a hit whose amyloidogenic core has been mutated away is
  suspect even if AF2 likes it. Read AGGRESCAN relative to native, never as an
  absolute.
* **LLPS is a hypothesis you choose to encode**, not an established correction:
  droplets are the competing fate of tau, so steering below native is a modeling
  decision.

**Where you start.** The native column is measured on the PHF tau sequence
itself (5O3L, 73 residues) -- the sequence that does form the target filament in
reality.

Read down the **Native PHF tau** column and the shape of the problem appears.
AF2 has no confidence in any fold for this sequence, and **least of all in the
interface** -- ipTM comes out below pLDDT, meaning the stacking that makes it a
fibril is exactly what it misses. The reference-based metrics agree
independently of AF2's opinion: the fold is not there and essentially no native
inter-chain contact is recovered. Meanwhile the sequence itself does hold an
almost maximally amyloidogenic window, VQIVYK at residues 1-6, which is what
makes the failure a structural one rather than a sequence one.

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

# 3. Prepare reference target structure(s). Every option spelled out, so the
#    command says what it targets instead of relying on the defaults.
uv run python prepare_reference.py --pdb-id 5O3L --chains A,C,E,G,I   # PHF tau
uv run python prepare_reference.py --pdb-id 6ELM --chains A           # WNK2 CCT1 monomer
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

`--pdb-id 5O3L --chains A,C,E,G,I` is what you get by passing nothing, but
writing it out means the command records which target it built -- worth doing in
a script or a lab notebook, where a bare `prepare_reference.py` a year later
does not say what it produced.

After setup, you should have:
- `params/` -- AF2 model weights
- `data/5o3l_acegi_ca_coords.npy` -- PHF target coordinates, shape (5, 73, 3)
- `data/5o3l_acegi_sequence.txt` -- PHF native sequence (73 residues)
- `data/6elm_a_ca_coords.npy` -- WNK2 target coordinates, shape (1, 98, 3)
- `data/6elm_a_sequence.txt` -- WNK2 native sequence (98 residues)

## Quick Start

```bash
# PHF tau 5-chain homooligomer (the default target, written out)
uv run python run_search.py --pdb-id 5O3L --chains A,C,E,G,I --n-steps 100

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
--data-dir DIR       AF2 parameters directory (default: params)
--ref-coords FILE    Reference CA coordinates .npy (auto-derived from --pdb-id/--chains)
--initial-seq SEQ    Starting sequence (auto-loaded from prepared reference)

Search parameters:
--n-steps N          Number of optimization steps (default: 1000)
--temperature T      MC temperature for acceptance (default: 1.0)
--n-mutations N      Mutations per step (default: 1)
--num-recycles N     AF2 recycles (default: 3)
--w-plddt W          Weight for pLDDT in fitness (default: 1.0)
--w-rmsd W           Weight for RMSD in fitness (default: 1.0)
--w-ptm W            Weight for pTM (default: 0.0)
--w-iptm W           Weight for interface pTM; higher is better (default: 0.0)
--w-tm-score W             Weight for TM-score; higher is better (default: 0.0)
--w-lddt W           Weight for lDDT; higher is better (default: 0.0)
--w-fnat W           Weight for Fnat; higher is better (default: 0.0)
--w-ipae W           Weight for interface PAE; LOWER is better, so use a
                     NEGATIVE weight (default: 0.0)
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

## How each metric is computed

What the metrics *mean* is above. This section is the provenance, the cost and
the caveats -- what you need in order to trust or cite a number, not to read
one.

TANGO and WALTZ, which this literature usually reports, are deliberately absent:
TANGO ships as a licensed compiled binary and WALTZ is web-server only, with a
score that depends on a licensed FoldX matrix. AGGRESCAN stands in for TANGO's
role and the `6aa` amyloid head for WALTZ's. **Neither reproduces the original
numbers, so results here are not comparable to published TANGO or WALTZ values.**

### AGGRESCAN

`aggrescan.py` is a reimplementation of the published algorithm
([Conchillo-Sole et al. 2007](https://doi.org/10.1186/1471-2105-8-65)), whose
per-residue scale was measured *in vivo* from the intracellular aggregation of
amyloid-beta central-hydrophobic-cluster mutants
([Sanchez de Groot et al. 2005](https://doi.org/10.1186/1472-6807-5-18)). The
scale is its a3v values, the window length follows the sequence length, and
`HST = -0.02` with a 5-residue minimum and no proline defines a hot spot -- all
taken from the paper and its additional files.

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
`--w-mpnn-score` must be **negative** to reward sequences ProteinMPNN likes.

It answers a genuinely different question from the rest. RMSD asks whether AF2
folds the sequence onto the target; the amyloid indices ask whether the sequence
reads like an aggregator; this asks whether a design model would have *proposed*
the sequence for this backbone -- which is the criterion the sequences were
generated under in the first place.

```bash
uv run python run_search.py --n-steps 100 --mpnn-scores on          # record only
uv run python run_search.py --n-steps 100 --w-mpnn-score -1.0             # negative weight!
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

### amyloid and LLPS

Both come from
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

### Which metrics are present when

Whether a key exists in `results.json` depends on the target and the flags, so
analysis code needs to handle absence. The complete set of conditions:

| metric | present when |
|--------|--------------|
| pLDDT, RMSD, pTM, TM-score, lDDT, AGGRESCAN (and the `*_chain` variants) | always |
| ipTM, iPAE, Fnat | the target has more than one chain |
| ddG | a matrix exists in `models/ddg/` for the target |
| ProteinMPNN | `--mpnn-scores on`, or `--w-mpnn-score` is nonzero |
| amyloid, LLPS | `--esm-scores on`, or `--w-amyloid`/`--w-llps` is nonzero |

A metric that does not apply is **absent**, never a placeholder: a monomer has
no `fnat` key rather than `fnat = nan`. `n_native_contacts`, the denominator of
Fnat, is a property of the target and appears once at the top level rather than
in every record.

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
    +-> predict.py     AF2 black-box evaluator (AF2Predictor)
    +-> utils.py       RMSD computation + mutation operator
    +-> esm_scores.py  ESM2-3B amyloid / LLPS scores
    +-> shape.py       TM-score / lDDT / Fnat against the reference
    +-> aggrescan.py   AGGRESCAN aggregation propensity
    +-> ddg.py         ThermoMPNN stability, from a precomputed matrix
    +-> mpnn_score.py  ProteinMPNN inverse-folding score
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

`compute_fitness` is a weighted sum over every metric in
[What the metrics mean](#what-the-metrics-mean) -- thirteen weights, all of them
0 by default except `w_plddt` and `w_rmsd`. Rather than repeat the formula here
(where it went stale four times), the rule is:

* every weight flag is `--w-` plus its record key with `_` replaced by `-`, so
  `tm_score` is driven by `--w-tm-score`
* a metric passed as `None`, or one whose weight is 0, is **skipped** rather
  than multiplied by zero -- so a metric that is merely being reported cannot
  affect the fitness even if it came back non-finite

Add nonlinear terms, thresholds or additional objectives here without touching
anything else.

### 3. Change the mutation operator -- `utils.py`

`mutate_sequence(seq, n_mutations)` does uniform random single-point mutations. For a GA, you would add crossover operators here. The alphabet is defined as:

```python
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 letters
```

### 4. The AF2 wrapper -- `predict.py`

Treat this as a black box for optimization purposes: it is `f(sequence) -> scores
and coordinates`, and the search's job is to explore sequence space, not to
change the evaluator. But it is not off-limits. AF2 populates far more in
`model.aux` than this returns -- contact maps (`cmap`, `i_cmap`), the full PAE
matrix, per-residue confidences -- and pulling another one out is a two-line
change. `ptm`/`iptm`/`ipae` were added exactly that way; they had been computed
and discarded since the first commit.


`AF2Predictor.predict(sequence)` wraps AlphaFold2 via ColabDesign. It takes a sequence string and returns:

| Key | Type | Description |
|-----|------|-------------|
| `plddt` | float (0-1) | Mean prediction confidence |
| `ptm` | float (0-1) | Predicted TM-score of the assembly |
| `iptm` | float (0-1) | Interface pTM -- **only when `copies > 1`** |
| `ipae` | float (0-1) | Interface PAE, divided by 31 A -- **only when `copies > 1`** |
| `ipae_angstrom` | float | The same in angstroms -- **only when `copies > 1`** |
| `ca_coords` | ndarray (copies, length, 3) | Predicted 3D coordinates |
| `plddt_per_residue` | ndarray (length,) | Per-residue confidence |
| `pdb_str` | str | Full atomic structure in PDB format |

Think of this as `f(x) -> score` where `x` is an *L*-dimensional categorical variable. Each call takes a few seconds on GPU.

### 5. Add a new target

The two commands are in [Setup](#setup) and [Quick Start](#quick-start); the
pipeline derives sequence length, chain count and initial sequence from the
prepared reference files, so nothing else needs changing.

Two things a new target does not get for free:

* **ddG** needs its own matrix. Without one the term is simply absent, which is
  why `--w-ddg` errors rather than silently doing nothing -- run
  `precompute_ddg.py` for the target, or leave the weight at 0.
* **Interface metrics** (ipTM, iPAE, Fnat) exist only if the target has more
  than one chain.

## Output

- `results.json` -- full optimization trajectory (sequence, pLDDT, RMSD, fitness per step; plus `amyloid_mean`, `amyloid_max`, `llps_mean`, `llps_max` when the sequence scores are on)
- `structures/step_0000.pdb` -- predicted 3D structure for the initial sequence
- `structures/step_NNNN.pdb` -- structures at subsequent steps

PDB files can be visualized with [PyMOL](https://pymol.org/), [ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or [Mol*](https://molstar.org/).
