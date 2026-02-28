"""Extract reference structure from a PDB file.

Downloads the PDB and extracts CA coordinates and native sequence
from specified chains.

Saves:
  - data/<pdb_id>_<chains>_ca_coords.npy  shape (n_chains, n_residues, 3)
  - data/<pdb_id>_<chains>.pdb            reference PDB with selected chains
  - data/<pdb_id>_<chains>_sequence.txt   native amino acid sequence
"""

import argparse
import os
import sys

import numpy as np
from Bio.PDB import PDBIO, PDBList, PDBParser, Select
from Bio.PDB.Polypeptide import PPBuilder


def extract_reference(
    pdb_id: str,
    chains: list[str],
    data_dir: str = "data",
    expected_residues: int | None = None,
) -> dict:
    """Download PDB and extract CA coords + sequence for given chains.

    Returns dict with keys: ca_coords, sequence, n_residues, coords_path,
    pdb_path, seq_path.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Download PDB
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=data_dir, file_format="pdb")
    if not os.path.exists(pdb_file):
        print(f"ERROR: Failed to download PDB file to {pdb_file}", file=sys.stderr)
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    ppb = PPBuilder()

    # Extract CA coordinates and sequences for each target chain
    all_ca_coords = []
    all_sequences = []
    for chain_id in chains:
        if chain_id not in [c.id for c in model.get_chains()]:
            print(
                f"ERROR: Chain {chain_id} not found in {pdb_id}. "
                f"Available: {[c.id for c in model.get_chains()]}",
                file=sys.stderr,
            )
            sys.exit(1)
        chain = model[chain_id]

        # Extract CA coordinates
        ca_coords = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue  # skip hetero atoms
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_vector().get_array())

        if expected_residues is not None and len(ca_coords) != expected_residues:
            print(
                f"WARNING: Chain {chain_id} has {len(ca_coords)} CA atoms, "
                f"expected {expected_residues}",
                file=sys.stderr,
            )

        # Extract sequence using PPBuilder
        peptides = ppb.build_peptides(chain)
        sequence = "".join(str(pp.get_sequence()) for pp in peptides)

        all_ca_coords.append(ca_coords)
        all_sequences.append(sequence)

    # Validate all chains have the same number of residues
    lengths = [len(c) for c in all_ca_coords]
    if len(set(lengths)) != 1:
        print(f"ERROR: Chains have different lengths: {lengths}", file=sys.stderr)
        sys.exit(1)

    n_res = lengths[0]
    # Use the first chain's sequence as the representative
    sequence = all_sequences[0]

    ca_array = np.array(all_ca_coords, dtype=np.float64)
    chain_label = "".join(chains).lower()
    prefix = f"{pdb_id.lower()}_{chain_label}"

    coords_path = os.path.join(data_dir, f"{prefix}_ca_coords.npy")
    pdb_path = os.path.join(data_dir, f"{prefix}.pdb")
    seq_path = os.path.join(data_dir, f"{prefix}_sequence.txt")

    # Save coordinates
    np.save(coords_path, ca_array)
    print(f"Saved {coords_path} (shape {ca_array.shape})")

    # Save reference PDB with selected chains
    class _ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id in chains

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path, _ChainSelect())
    print(f"Saved {pdb_path}")

    # Save sequence
    with open(seq_path, "w") as f:
        f.write(sequence + "\n")
    print(f"Saved {seq_path}")
    print(f"Sequence ({len(sequence)} residues): {sequence}")

    return {
        "ca_coords": ca_array,
        "sequence": sequence,
        "n_residues": n_res,
        "coords_path": coords_path,
        "pdb_path": pdb_path,
        "seq_path": seq_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference CA coordinates and sequence from a PDB structure"
    )
    parser.add_argument(
        "--pdb-id", default="5O3L", help="PDB ID to download (default: 5O3L)"
    )
    parser.add_argument(
        "--chains",
        default="A,C,E,G,I",
        help="Comma-separated chain IDs (default: A,C,E,G,I)",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Output directory (default: data)"
    )
    parser.add_argument(
        "--expected-residues",
        type=int,
        default=None,
        help="Expected residues per chain (optional, for validation)",
    )
    args = parser.parse_args()

    chains = [c.strip() for c in args.chains.split(",")]
    extract_reference(
        pdb_id=args.pdb_id,
        chains=chains,
        data_dir=args.data_dir,
        expected_residues=args.expected_residues,
    )


if __name__ == "__main__":
    main()
