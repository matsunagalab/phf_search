"""Extract PHF reference structure from PDB 5O3L.

Downloads 5O3L and extracts CA coordinates from chains A, C, E, G, I
(one protofilament of the paired helical filament).
Saves:
  - data/reference_ca_coords.npy  shape (5, 73, 3)
  - data/5o3l_acegi.pdb           reference PDB with selected chains
"""

import os
import sys

import numpy as np
from Bio.PDB import PDBIO, PDBList, PDBParser, Select

TARGET_CHAINS = ["A", "C", "E", "G", "I"]
EXPECTED_RESIDUES = 73  # residues 306-378


class ChainSelect(Select):
    """Select only target chains."""

    def accept_chain(self, chain):
        return chain.id in TARGET_CHAINS


def main():
    os.makedirs("data", exist_ok=True)

    # Download PDB
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file("5O3L", pdir="data", file_format="pdb")
    if not os.path.exists(pdb_file):
        print(f"ERROR: Failed to download PDB file to {pdb_file}", file=sys.stderr)
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("5O3L", pdb_file)
    model = structure[0]

    # Extract CA coordinates for each target chain
    all_ca_coords = []
    for chain_id in TARGET_CHAINS:
        chain = model[chain_id]
        ca_coords = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue  # skip hetero atoms
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_vector().get_array())
        if len(ca_coords) != EXPECTED_RESIDUES:
            print(
                f"WARNING: Chain {chain_id} has {len(ca_coords)} CA atoms, "
                f"expected {EXPECTED_RESIDUES}",
                file=sys.stderr,
            )
        all_ca_coords.append(ca_coords)

    # Validate all chains have the same number of residues
    lengths = [len(c) for c in all_ca_coords]
    if len(set(lengths)) != 1:
        print(f"ERROR: Chains have different lengths: {lengths}", file=sys.stderr)
        sys.exit(1)

    n_res = lengths[0]
    ca_array = np.array(all_ca_coords, dtype=np.float64)  # (5, n_res, 3)
    print(f"Extracted CA coordinates: shape {ca_array.shape}")

    # Save coordinates
    np.save("data/reference_ca_coords.npy", ca_array)
    print("Saved data/reference_ca_coords.npy")

    # Save reference PDB with selected chains
    io = PDBIO()
    io.set_structure(structure)
    io.save("data/5o3l_acegi.pdb", ChainSelect())
    print("Saved data/5o3l_acegi.pdb")


if __name__ == "__main__":
    main()
