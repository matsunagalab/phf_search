# PHF Search

Sequence optimization for protein structures using AlphaFold2 via ColabDesign. Supports monomers and homooligomers from any PDB target.

## Project Structure

```
predict.py           - AF2 prediction wrapper (ColabDesign hallucination protocol, configurable length/copies)
mc_search.py         - Monte Carlo search engine (mutate -> predict -> evaluate -> accept/reject)
fitness.py           - Fitness function (w_plddt * pLDDT - w_rmsd * RMSD)
utils.py             - Kabsch RMSD, chain-permutation RMSD minimization, sequence mutation
prepare_reference.py - Extract reference CA coordinates and sequence from any PDB
run_search.py        - CLI entry point (--pdb-id, --chains for target selection)
download_params.sh   - AF2 parameter download
pyproject.toml       - Project definition (managed by uv)
```

## Setup

```bash
bash download_params.sh             # AF2 parameters -> params/
uv run python prepare_reference.py  # Default: 5O3L chains A,C,E,G,I -> data/5o3l_acegi_*
uv run python prepare_reference.py --pdb-id 6ELM --chains A  # -> data/6elm_a_*
```

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
- **Fitness**: `w_plddt * pLDDT - w_rmsd * RMSD` (higher is better)
- **MC acceptance**: Metropolis criterion `exp(delta_fitness / temperature)`
- **PDB saving**: `model.save_pdb(filename=None, get_best=False)` returns PDB string (note: `save_current_pdb` has a missing `return` in ColabDesign)
- **Reference files**: `data/<pdb_id>_<chains>_ca_coords.npy`, `data/<pdb_id>_<chains>_sequence.txt`

## Development

- Package management via `uv`
- Python >= 3.10
- `data/`, `params/`, `structures/`, `results.json` are in `.gitignore`
