"""How close a candidate sequence stays to the target's own sequence.

No model and no structure: this is the candidate read against the reference
sequence, in microseconds. Two measures, because they disagree in a useful way
-- one counts positions, the other weighs what was substituted.

**seq_recovery** in [0, 1] is the fraction of positions that still match. This
is ProteinMPNN's own `seq_recovery`, verified against its implementation
(`protein_mpnn_run.py`: the one-hot inner product of the native and designed
sequences, averaged over designable positions) -- for a fully designable
homo-oligomer that reduces to the plain identity computed here, agreeing to
float32 rounding. It is also the quantity the earlier tau polymorph analysis
reports as `seq_recovery`, where ProteinMPNN designs on 5O3L recovered 20-30%
(median 26.7%) against the 55% quoted for the original ProteinMPNN paper.

**blosum62** is the mean BLOSUM62 substitution score per position. Where
seq_recovery treats every mismatch alike, this one does not: I to V costs little
and I to D costs a lot. Gadhe et al. (2026) characterize their designs with
"blosum62 distances to the WT sequence" -- but they do not give the formula, so
the convention here is ours and their numbers are not reproduced:

    blosum62             mean BLOSUM62 score per aligned position
    blosum62_normalized  the same divided by the reference's self-score, so
                         identity is exactly 1.0 and the scale is comparable
                         across targets

Measured on native PHF tau (5O3L, 73 residues), whose self-score averages 5.25:

    sequence            blosum62    normalized   seq_recovery
    native itself         +5.25        1.000         1.000
    two mutations         +5.07        0.966         0.973
    poly-alanine          -0.73       -0.138         0.000
    native reversed       -1.16       -0.222         0.041

Note the normalized form is not bounded below by 0 -- an actively dissimilar
sequence scores negative. Only `seq_recovery` is a fraction.

Both are measured against the **target's native sequence**, not against
whatever `--initial-seq` a run happened to start from, so a restarted or
continued search reports numbers on the same footing.
"""

import numpy as np

# The rest of the pipeline works over the 20 standard residues, so this does
# too. BLOSUM62 itself also scores B, Z, X and *, but accepting them here would
# let a sequence through that aggrescan.py and ddg.py would reject.
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Cached on first use: loading the matrix costs more than every comparison.
_MATRIX = None


def _matrix() -> dict:
    """BLOSUM62 as a plain nested dict, keyed by residue letter."""
    global _MATRIX
    if _MATRIX is None:
        from Bio.Align import substitution_matrices

        loaded = substitution_matrices.load("BLOSUM62")
        _MATRIX = {
            a: {b: float(loaded[a][b]) for b in loaded.alphabet}
            for a in loaded.alphabet
        }
    return _MATRIX


def _check(sequence: str, reference: str) -> None:
    if not reference:
        raise ValueError("Cannot compare against an empty reference sequence")
    if len(sequence) != len(reference):
        raise ValueError(
            f"sequence has {len(sequence)} residues, the reference has "
            f"{len(reference)}; these metrics assume a 1:1 correspondence"
        )
    unknown = sorted(set(sequence) | set(reference) - set(ALPHABET))
    unknown = [residue for residue in unknown if residue not in ALPHABET]
    if unknown:
        raise ValueError(
            f"non-standard residue(s) {unknown}: these metrics cover "
            f"{ALPHABET}. BLOSUM62 would score X, B and Z, but the rest of the "
            "pipeline does not accept them."
        )


def seq_recovery(sequence: str, reference: str) -> float:
    """Fraction of positions matching the reference, in [0, 1]."""
    _check(sequence, reference)
    return sum(a == b for a, b in zip(sequence, reference)) / len(reference)


def blosum62(sequence: str, reference: str) -> float:
    """Mean BLOSUM62 substitution score per position."""
    _check(sequence, reference)
    matrix = _matrix()
    return float(
        np.mean([matrix[a][b] for a, b in zip(sequence, reference)])
    )


def compare(sequence: str, reference: str) -> dict:
    """Both measures, plus the normalized BLOSUM62 score.

    Returns:
        dict with `seq_recovery`, `blosum62` and `blosum62_normalized`.
    """
    raw = blosum62(sequence, reference)
    self_score = blosum62(reference, reference)
    return {
        "seq_recovery": seq_recovery(sequence, reference),
        "blosum62": raw,
        # The reference's self-score is the ceiling, so this reads as "how much
        # of the reference's own BLOSUM62 mass does the candidate retain".
        "blosum62_normalized": raw / self_score if self_score else float("nan"),
    }
