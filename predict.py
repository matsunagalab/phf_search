"""AF2 prediction wrapper using ColabDesign.

Uses the hallucination protocol for structure prediction.
Non-multimer mode uses ptm models (model_{1-5}_ptm).
Supports both homooligomers (copies > 1) and monomers (copies = 1).
"""

import numpy as np
from colabdesign import mk_af_model

# ColabDesign reports PAE divided by the largest bin of AF2's PAE head, so its
# pae/i_pae are in [0, 1] rather than angstroms (colabdesign/af/loss.py).
PAE_SCALE_ANGSTROM = 31.0


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
                plddt: mean pLDDT score (float, 0-1), higher is better
                ptm: predicted TM-score over the whole assembly (0-1), higher
                    is better
                iptm, ipae, ipae_angstrom: interface confidences, present
                    **only when copies > 1**. (ColabDesign spells these `i_ptm`
                    and `i_pae`; they are renamed here so that every record key
                    maps to its weight flag by replacing `_` with `-`.) iptm is
                    pTM over inter-chain pairs (higher better); ipae is the mean
                    inter-chain
                    predicted aligned error as ColabDesign reports it, divided
                    by 31 A so it lies in [0, 1] (lower better); and
                    ipae_angstrom is that multiplied back, verified equal to
                    the mean of the inter-chain block of aux["pae"] (which is
                    in angstroms) to within 0.001 A.

                    Note these come from a non-multimer AF2 run with chains
                    made by `copies` and a residue-index offset, so iptm is
                    ColabDesign's interface-masked quantity, not AF-Multimer's
                    official ipTM. Interface-PAE thresholds quoted in the
                    binder-design literature were measured with AF-Multimer and
                    do not transfer unexamined.
                ca_coords: CA coordinates, shape (copies, length, 3)
                plddt_per_residue: per-residue pLDDT array
                pdb_str: predicted structure in PDB format

        ptm, i_ptm and i_pae cost nothing extra: AF2 computes them in the same
        forward pass that produces pLDDT, and this wrapper used to discard them.
        """
        self.model.set_seq(seq=sequence)
        self.model.predict()

        plddt_values = np.array(self.model.aux["plddt"])
        mean_plddt = float(plddt_values.mean())

        # Already computed by the same forward pass.
        ptm = float(np.array(self.model.aux["ptm"]))

        # Extract CA coordinates (atom index 1 = CA in atom37 representation)
        atom_positions = np.array(self.model.aux["atom_positions"])
        ca_coords = atom_positions[:, 1, :]
        ca_coords = ca_coords.reshape(self.copies, self.length, 3)

        # save_current_pdb has a missing return in ColabDesign; call save_pdb directly
        pdb_str = self.model.save_pdb(filename=None, get_best=False)

        result = {
            "plddt": mean_plddt,
            "ptm": ptm,
            "ca_coords": ca_coords,
            "plddt_per_residue": plddt_values,
            "pdb_str": pdb_str,
        }

        # Interface metrics only exist when there is an interface. ColabDesign
        # adds i_pae to the losses only for copies > 1 (af/loss.py), and pops
        # i_ptm from its own log for a single chain because it is meaningless
        # there -- it returns 0.0, which would read as a catastrophic interface
        # rather than as "no interface". Reading them from aux unconditionally
        # both crashed the monomer target and bypassed that guard.
        if self.copies > 1:
            i_pae = float(np.array(self.model.aux["log"]["i_pae"]))
            result["iptm"] = float(np.array(self.model.aux["i_ptm"]))
            result["ipae"] = i_pae
            result["ipae_angstrom"] = i_pae * PAE_SCALE_ANGSTROM

        return result


# Backward compatibility alias
PHFPredictor = AF2Predictor
