"""AF2 prediction wrapper using ColabDesign.

Uses the hallucination protocol with copies=5 for 5-chain homooligomer
structure prediction. Non-multimer mode uses ptm models (model_{1-5}_ptm).
"""

import numpy as np
from colabdesign import mk_af_model


class PHFPredictor:
    def __init__(self, data_dir: str = "params", num_recycles: int = 3):
        self.model = mk_af_model(
            protocol="hallucination",
            num_recycles=num_recycles,
            data_dir=data_dir,
            use_multimer=False,
        )
        self.model.prep_inputs(length=73, copies=5)

    def predict(self, sequence: str) -> dict:
        """Predict 5-chain homooligomer structure from a single sequence.

        Args:
            sequence: amino acid sequence (73 residues)

        Returns:
            dict with keys:
                plddt: mean pLDDT score (float, 0-1)
                ca_coords: CA coordinates, shape (5, 73, 3)
                plddt_per_residue: per-residue pLDDT array
        """
        self.model.set_seq(seq=sequence)
        self.model.predict()

        plddt_values = np.array(self.model.aux["plddt"])
        mean_plddt = float(plddt_values.mean())

        # Extract CA coordinates (atom index 1 = CA in atom37 representation)
        atom_positions = np.array(self.model.aux["atom_positions"])
        ca_coords = atom_positions[:, 1, :]  # (365, 3)
        ca_coords = ca_coords.reshape(5, 73, 3)

        # save_current_pdb has a missing return in ColabDesign; call save_pdb directly
        pdb_str = self.model.save_pdb(filename=None, get_best=False)

        return {
            "plddt": mean_plddt,
            "ca_coords": ca_coords,
            "plddt_per_residue": plddt_values,
            "pdb_str": pdb_str,
        }
