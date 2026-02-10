# PHF Search

Sequence optimization for tau protein filament structures using AlphaFold2 as a black-box evaluator. The current implementation uses Metropolis Monte Carlo, but the codebase is structured so you can **replace the search strategy** (e.g., with a genetic algorithm) without touching the prediction or evaluation layers.

## How It Works

The evaluator is [AlphaFold2](https://github.com/google-deepmind/alphafold) (AF2), a deep learning model that predicts a protein's 3D structure from its amino acid sequence. We access AF2 through [ColabDesign](https://github.com/sokrypton/ColabDesign), which provides a differentiable interface to the model. For each candidate sequence, AF2 runs a forward pass and outputs a predicted 3D structure along with per-residue confidence scores. This prediction is the bottleneck (~seconds per evaluation on GPU), making it a classic expensive black-box optimization setting.

## The Optimization Problem

| Aspect | Details |
|--------|---------|
| **Search space** | Strings of length 73 over a 20-letter alphabet (amino acids) |
| **Objective** | Maximize `fitness = w_plddt * pLDDT - w_rmsd * RMSD` |
| **Evaluation cost** | ~seconds per call on GPU (AlphaFold2 forward pass) |

**What the two metrics mean (no biology background needed):**

- **pLDDT** (0 to 1): The predictor's own confidence in its output. Higher = the model is more sure the predicted 3D shape is correct.
- **RMSD** (angstroms, 0 to inf): Distance between the predicted shape and a known experimental target shape. Lower = closer match.

A good sequence has **high confidence** (pLDDT close to 1) and **low deviation** from the target (RMSD close to 0).

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync --extra cpu    # CPU-only (for testing)
uv sync --extra cuda   # GPU (for real runs)

# 2. Download AlphaFold2 model parameters (~3.5 GB)
bash download_params.sh

# 3. Prepare the reference target structure
uv run python prepare_reference.py
```

After setup, you should have:
- `params/` -- AF2 model weights
- `data/reference_ca_coords.npy` -- target 3D coordinates to match

## Quick Start

```bash
uv run python run_search.py --n-steps 100
```

This runs 100 Monte Carlo steps starting from the native tau sequence. Results are saved to `results.json` and predicted structures to `structures/`.

### CLI Options

```
--n-steps N          Number of optimization steps (default: 1000)
--temperature T      MC temperature for acceptance (default: 1.0)
--n-mutations N      Mutations per step (default: 1)
--w-plddt W          Weight for pLDDT in fitness (default: 1.0)
--w-rmsd W           Weight for RMSD in fitness (default: 1.0)
--save-interval N    Save PDB structure every N steps (default: 1)
--structures-dir D   Directory for PDB files (default: structures)
--output FILE        Output JSON path (default: results.json)
```

## Architecture Overview

```
run_search.py          CLI entry point
    |
    v
mc_search.py           Search strategy (THE PART YOU REPLACE)
    |
    v
predict.py             Black-box evaluator (DO NOT MODIFY)
fitness.py             Objective function
utils.py               RMSD computation + mutation operator
```

The key design: **`predict.py` is an expensive black-box function.** You give it a sequence string, it returns scores and 3D coordinates. Your optimizer's job is to explore the sequence space efficiently.

## Where to Modify

### 1. Replace the search strategy -- `mc_search.py`

This is the main file to replace or rewrite. The current `MonteCarloSearch` class follows a simple interface:

```python
# The evaluator -- call this for any candidate sequence
result = self._evaluate(seq)  # returns dict with "plddt", "rmsd", "fitness", "pdb_str"

# The mutation operator -- or write your own
from utils import mutate_sequence
new_seq = mutate_sequence(current_seq, n_mutations=1)
```

To implement a genetic algorithm, you would:
1. Create a new class (e.g., `GeneticAlgorithmSearch`) in a new file or replace `mc_search.py`
2. Use `self.predictor.predict(seq)` + `min_permutation_rmsd()` + `compute_fitness()` to evaluate candidates (or just call `_evaluate()`)
3. Wire it up in `run_search.py`

**Sequence representation:** A sequence is a Python string of length 73 where each character is one of `ACDEFGHIKLMNPQRSTVWY` (the 20 standard amino acids). You can treat this as a categorical optimization problem with 73 positions and 20 choices per position.

### 2. Change the fitness function -- `fitness.py`

Currently a simple weighted sum:

```python
fitness = w_plddt * plddt - w_rmsd * rmsd
```

You can add nonlinear terms, thresholds, or additional objectives here without changing anything else.

### 3. Change the mutation operator -- `utils.py`

`mutate_sequence(seq, n_mutations)` does uniform random single-point mutations. For a GA, you would add crossover operators here. The alphabet is defined as:

```python
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 letters
```

### 4. The evaluator (you probably don't need to touch this) -- `predict.py`

`PHFPredictor.predict(sequence)` wraps AlphaFold2 via ColabDesign. It takes a 73-character string and returns:

| Key | Type | Description |
|-----|------|-------------|
| `plddt` | float (0-1) | Mean prediction confidence |
| `ca_coords` | ndarray (5, 73, 3) | Predicted 3D coordinates |
| `pdb_str` | str | Full atomic structure in PDB format |

Think of this as `f(x) -> score` where `x` is a 73-dimensional categorical variable. Each call takes a few seconds on GPU.

## Output

- `results.json` -- full optimization trajectory (sequence, pLDDT, RMSD, fitness per step)
- `structures/step_0000.pdb` -- predicted 3D structure for the initial sequence
- `structures/step_NNNN.pdb` -- structures at subsequent steps

PDB files can be visualized with [PyMOL](https://pymol.org/), [ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or [Mol*](https://molstar.org/).
