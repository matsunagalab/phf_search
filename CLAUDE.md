# PHF Search

Monte Carlo sequence optimization for PHF (Paired Helical Filament) tau protein structures using AlphaFold2 via ColabDesign.

## Project Structure

```
predict.py           - AF2 prediction wrapper (ColabDesign hallucination protocol, 5-chain homooligomer)
mc_search.py         - Monte Carlo search engine (mutate -> predict -> evaluate -> accept/reject)
fitness.py           - Fitness function (w_plddt * pLDDT - w_rmsd * RMSD)
utils.py             - Kabsch RMSD, chain-permutation RMSD minimization, sequence mutation
prepare_reference.py - Extract reference CA coordinates from PDB 5O3L
run_search.py        - CLI entry point
download_params.sh   - AF2 parameter download
pyproject.toml       - Project definition (managed by uv)
```

## Setup

```bash
bash download_params.sh             # AF2 parameters -> params/
uv run python prepare_reference.py  # Reference coordinates -> data/reference_ca_coords.npy
```

## Usage

```bash
uv run python run_search.py --n-steps 100 --temperature 1.0
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `params` | AF2 parameters directory |
| `--ref-coords` | `data/reference_ca_coords.npy` | Reference CA coordinates |
| `--initial-seq` | Native sequence (73 residues) | Starting sequence |
| `--n-steps` | 1000 | Number of MC steps |
| `--temperature` | 1.0 | MC temperature |
| `--n-mutations` | 1 | Mutations per step |
| `--num-recycles` | 3 | AF2 recycles |
| `--w-plddt` | 1.0 | pLDDT weight |
| `--w-rmsd` | 1.0 | RMSD weight |
| `--log-interval` | 10 | Log every N steps |
| `--save-interval` | 1 | Save PDB every N steps |
| `--structures-dir` | `structures` | PDB output directory |
| `--output` | `results.json` | Output JSON path |

## Output

- `results.json` -- Full trajectory (sequence, pLDDT, RMSD, fitness per step)
- `structures/step_0000.pdb` -- Predicted structure for the initial sequence
- `structures/step_NNNN.pdb` -- Structures at subsequent steps (per `--save-interval`)

## Technical Details

- **Prediction**: ColabDesign hallucination protocol, non-multimer (ptm models), copies=5
- **Structure comparison**: Kabsch RMSD over all 120 chain permutations, taking the minimum
- **Fitness**: `w_plddt * pLDDT - w_rmsd * RMSD` (higher is better)
- **MC acceptance**: Metropolis criterion `exp(delta_fitness / temperature)`
- **PDB saving**: `model.save_pdb(filename=None, get_best=False)` returns PDB string (note: `save_current_pdb` has a missing `return` in ColabDesign)

## Development

- Package management via `uv`
- Python >= 3.10
- `data/`, `params/`, `structures/`, `results.json` are in `.gitignore`
