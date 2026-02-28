"""AF2 prediction wrapper using ColabDesign.

Uses the hallucination protocol for structure prediction.
Non-multimer mode uses ptm models (model_{1-5}_ptm).
Supports both homooligomers (copies > 1) and monomers (copies = 1).
"""

import numpy as np
from colabdesign import mk_af_model


class AF2Predictor:
    def __init__(
        self,
        data_dir: str = "params",
        num_recycles: int = 3,
        length: int = 73,
        copies: int = 5,
    ):
        self.length = length
        self.copies = copies
        self.model = mk_af_model(
            protocol="hallucination",
            num_recycles=num_recycles,
            data_dir=data_dir,
            use_multimer=False,
        )
        self.model.prep_inputs(length=length, copies=copies)

    def predict(self, sequence: str) -> dict:
        """Predict structure from a single sequence.

        Args:
            sequence: amino acid sequence

        Returns:
            dict with keys:
                plddt: mean pLDDT score (float, 0-1)
                ca_coords: CA coordinates, shape (copies, length, 3)
                plddt_per_residue: per-residue pLDDT array
                pdb_str: predicted structure in PDB format
        """
        self.model.set_seq(seq=sequence)
        self.model.predict()

        plddt_values = np.array(self.model.aux["plddt"])
        mean_plddt = float(plddt_values.mean())

        # Extract CA coordinates (atom index 1 = CA in atom37 representation)
        atom_positions = np.array(self.model.aux["atom_positions"])
        ca_coords = atom_positions[:, 1, :]
        ca_coords = ca_coords.reshape(self.copies, self.length, 3)

        # save_current_pdb has a missing return in ColabDesign; call save_pdb directly
        pdb_str = self.model.save_pdb(filename=None, get_best=False)

        return {
            "plddt": mean_plddt,
            "ca_coords": ca_coords,
            "plddt_per_residue": plddt_values,
            "pdb_str": pdb_str,
        }


# Backward compatibility alias
PHFPredictor = AF2Predictor
