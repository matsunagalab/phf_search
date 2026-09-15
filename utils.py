import random
from itertools import permutations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Kabsch algorithm: compute RMSD after optimal rotation. P, Q: (N, 3)."""
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rotated = P_centered @ R.T
    diff = P_rotated - Q_centered
    return float(np.sqrt((diff**2).sum() / len(P)))


def min_permutation_rmsd(
    pred_coords: np.ndarray, ref_coords: np.ndarray
) -> tuple[float, tuple[int, ...]]:
    """Try all n_chains! chain permutations and return the minimum RMSD.

    Args:
        pred_coords: predicted CA coordinates, shape (n_chains, n_residues, 3)
        ref_coords: reference CA coordinates, shape (n_chains, n_residues, 3)

    Returns:
        (best_rmsd, best_permutation)
    """
    n_chains = pred_coords.shape[0]
    ref_flat = ref_coords.reshape(-1, 3)
    best_rmsd = float("inf")
    best_perm: tuple[int, ...] = tuple(range(n_chains))
    for perm in permutations(range(n_chains)):
        pred_perm = pred_coords[list(perm)]
        pred_flat = pred_perm.reshape(-1, 3)
        rmsd = kabsch_rmsd(pred_flat, ref_flat)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
    return best_rmsd, best_perm


def parse_positions(spec: str) -> set[int]:
    """Parse ColabDesign's position syntax: "306,310-315,320" -> {306, 310..315, 320}.

    Same shape as `colabdesign.shared.prep.prep_pos`: comma-separated segments,
    ranges written with a hyphen, and the numbers are **PDB residue numbers**,
    not indices into the sequence. On the PHF target that means VQIVYK is
    `306-311`, not `1-6`.

    ColabDesign also accepts a chain prefix (`A1-10`) and a bare chain letter.
    Neither is accepted here, and the error says why: this pipeline designs one
    sequence that every chain shares, so a position cannot be fixed in one copy
    and free in another.
    """
    positions: set[int] = set()
    for segment in spec.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if any(character.isalpha() for character in segment):
            raise ValueError(
                f"{segment!r} looks like it names a chain. Positions here carry "
                "no chain: the chains share one designed sequence, so fixing a "
                "position fixes it in every copy."
            )
        low, _, high = segment.partition("-")
        try:
            start = int(low)
            end = int(high) if high else start
        except ValueError:
            raise ValueError(
                f"{segment!r} is not a residue number or a range like 306-311"
            ) from None
        if end < start:
            raise ValueError(f"{segment!r} counts backwards")
        positions.update(range(start, end + 1))
    if not positions:
        raise ValueError(f"{spec!r} names no positions")
    return positions


def designable_indices(
    residue_numbers: list[int], fixed: set[int], inverse: bool = False
) -> list[int]:
    """Sequence indices the search may mutate.

    Args:
        residue_numbers: PDB residue number of each sequence position, in order
        fixed: residue numbers to hold at the reference residue
        inverse: treat `fixed` as the only *designable* positions instead, which
            is what ColabDesign's `inverse=True` does -- useful for "design this
            segment and nothing else"

    Raises:
        ValueError: if a requested residue number is not in the structure, or
            if nothing is left to design.
    """
    available = set(residue_numbers)
    unknown = sorted(fixed - available)
    if unknown:
        low, high = min(available), max(available)
        raise ValueError(
            f"residue number(s) {unknown} are not in the target, which covers "
            f"{low}-{high}. These are PDB residue numbers, not sequence "
            f"indices -- position 1 of the sequence is residue {low}."
        )

    if inverse:
        indices = [i for i, number in enumerate(residue_numbers) if number in fixed]
    else:
        indices = [
            i for i, number in enumerate(residue_numbers) if number not in fixed
        ]
    if not indices:
        raise ValueError(
            "every position is fixed; there is nothing left for the search to "
            "change."
        )
    return indices


def mutate_sequence(
    seq: str, n_mutations: int = 1, designable: list[int] | None = None
) -> str:
    """Mutate random positions in the sequence to random amino acids.

    `designable` restricts which indices may change; None means all of them.
    """
    seq_list = list(seq)
    choices = range(len(seq)) if designable is None else designable
    positions = random.sample(list(choices), min(n_mutations, len(choices)))
    for pos in positions:
        current = seq_list[pos]
        candidates = [aa for aa in AMINO_ACIDS if aa != current]
        seq_list[pos] = random.choice(candidates)
    return "".join(seq_list)
