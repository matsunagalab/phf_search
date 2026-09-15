"""Predicted stability change of a candidate, from a precomputed ddG matrix.

ThermoMPNN (Dieckhaus et al., PNAS 121:e2314851121, 2024) scores the stability
effect of a point mutation from structure. `precompute_ddg.py` runs it once over
every single mutation of the reference structure; this module adds up the
entries for the positions a candidate actually changed. That is a table lookup,
so it costs microseconds and needs neither torch nor the 39 MB checkpoint at
search time.

    ddG > 0  destabilizing      ddG < 0  stabilizing

which is the usual convention in the stability literature, and the opposite of
the Megascale measurements ThermoMPNN was trained on -- ThermoMPNN negates them.
So a search that wants stable designs wants a **negative** `--w-ddg`.

Three limits, all of them real, none of them fixable here:

1. **Additivity.** A candidate differing from the reference at k positions is
   scored as the sum of k independent single-mutant predictions. Epistasis is
   invisible, and the error grows with k. A sequence 30 mutations away from the
   reference is well outside what this can honestly estimate; treat the number
   as a ranking signal among near neighbours, not a free energy.
2. **Out of distribution.** ThermoMPNN was trained on the Megascale set: small
   globular domains measured by proteolysis. An amyloid fibril core is held
   together by inter-chain stacking, and "folding stability" is not the same
   quantity. The predictions are used here because they are cheap and
   structure-aware, not because the model was validated on fibrils.
3. **Fixed reference structure.** Every number describes a mutation *of the
   reference backbone*, not of the structure AF2 predicts for the candidate.

The matrix is computed on the full multi-chain assembly, so inter-chain contacts
are in the model's input: on 5O3L, I3D scores +0.36 against a lone chain and
+1.62 under the default `mean` reduction over the five stacked copies. Those
copies are not equivalent -- per chain the same mutation gives
[1.34, 1.60, 1.76, 1.68, 1.70], lowest at the two ends of the stack, which have
one neighbour instead of two.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_MATRIX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "ddg"
)

# How to combine the chain copies of one position. The copies differ only
# through their neighbourhood in the stack: on 5O3L the two end chains average
# ~0.1 kcal/mol lower than the three interior ones, and the middle chain
# correlates 0.98 with the mean over all five. "sum" is on a different scale
# from the other two -- see the note where it is applied.
CHAIN_REDUCTIONS = ("mean", "middle", "sum")


def matrix_path(pdb_id: str, chains: list[str], matrix_dir: str | None = None) -> str:
    """Where precompute_ddg.py writes the matrix for a target."""
    prefix = f"{pdb_id.lower()}_{''.join(chains).lower()}"
    return os.path.join(matrix_dir or DEFAULT_MATRIX_DIR, f"{prefix}_thermompnn.npz")


class DDGLookup:
    """Candidate -> predicted ddG relative to the reference sequence.

    Args:
        path: .npz written by precompute_ddg.py
        chain_reduction: how to combine the chain copies of a position --
            "mean" over all copies, "middle" for the central chain of the
            stack, or "sum" for the whole assembly
    """

    def __init__(self, path: str, chain_reduction: str = "mean"):
        if chain_reduction not in CHAIN_REDUCTIONS:
            raise ValueError(
                f"chain_reduction must be one of {CHAIN_REDUCTIONS}, "
                f"got {chain_reduction!r}"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ddG matrix not found: {path}. Generate it once with\n"
                "  uv run python precompute_ddg.py --thermompnn-dir <ThermoMPNN checkout>"
            )

        data = np.load(path, allow_pickle=False)
        matrix = data["ddg"]  # (n_chains, n_residues, 20)
        self.reference = str(data["sequence"])
        alphabet = str(data["alphabet"])
        if alphabet != ALPHABET:
            raise ValueError(
                f"{path} uses alphabet {alphabet!r}, expected {ALPHABET!r}"
            )

        if chain_reduction == "mean":
            self.table = matrix.mean(axis=0)
        elif chain_reduction == "sum":
            # NB this scales the whole term by the chain count -- on the
            # 5-chain PHF target roughly 5x the "mean" values -- so a weight
            # tuned under one reduction is wrong under this one.
            self.table = matrix.sum(axis=0)
        else:
            # "middle": for an even chain count there is no middle, and this
            # takes the upper of the two central chains.
            self.table = matrix[matrix.shape[0] // 2]

        self.n_chains = matrix.shape[0]
        self.chain_reduction = chain_reduction
        self.path = path
        self._index = {aa: i for i, aa in enumerate(ALPHABET)}

        if len(self.reference) != self.table.shape[0]:
            raise ValueError(
                f"{path} is inconsistent: {len(self.reference)} reference "
                f"residues but {self.table.shape[0]} matrix rows"
            )
        logger.info(
            "ddG matrix %s: %d residues, %d chain copies reduced by %s",
            os.path.basename(path),
            len(self.reference),
            self.n_chains,
            chain_reduction,
        )

    def score(self, sequence: str) -> dict:
        """ddG of `sequence` relative to the matrix's reference sequence.

        Returns:
            dict with `ddg` (sum over mutated positions), `n_mutations`, and
            `ddg_max` (the single worst mutated position, or 0.0 for the
            reference sequence itself).
        """
        if len(sequence) != len(self.reference):
            raise ValueError(
                f"sequence has {len(sequence)} residues, the ddG matrix was "
                f"built for {len(self.reference)}"
            )

        contributions = []
        for position, (candidate, reference) in enumerate(
            zip(sequence, self.reference)
        ):
            if candidate == reference:
                continue
            try:
                value = self.table[position, self._index[candidate]]
            except KeyError as exc:
                raise ValueError(
                    f"no ddG for residue {candidate!r} at position "
                    f"{position + 1}: the matrix covers {ALPHABET}"
                ) from exc
            # NaN marks a position the reference structure does not model, and
            # it must not reach the fitness: a NaN there makes both branches of
            # the Metropolis criterion false, so every candidate is rejected
            # and the run finishes "successfully" having searched nothing.
            if np.isnan(value):
                raise ValueError(
                    f"the ddG matrix has no value for {candidate} at position "
                    f"{position + 1} (residue {reference} in the reference). "
                    "That position is unmodelled in the structure the matrix "
                    "was built from, so this target cannot use the ddG term "
                    "until the gap is filled or the term is switched off."
                )
            contributions.append(value)

        return {
            "ddg": float(sum(contributions)),
            "n_mutations": len(contributions),
            "ddg_max": float(max(contributions)) if contributions else 0.0,
        }


def load_if_available(
    pdb_id: str, chains: list[str], chain_reduction: str = "mean",
    matrix_dir: str | None = None,
) -> DDGLookup | None:
    """DDGLookup for a target, or None when no matrix has been precomputed.

    Targets other than the ones shipped with a matrix simply do not get the
    term, rather than failing a run that never asked for it.
    """
    path = matrix_path(pdb_id, chains, matrix_dir)
    if not os.path.exists(path):
        logger.info(
            "No ddG matrix at %s; skipping the ddG term. Run precompute_ddg.py "
            "to add one for this target.",
            path,
        )
        return None
    return DDGLookup(path, chain_reduction=chain_reduction)
