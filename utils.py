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
    """Try all 5! = 120 chain permutations and return the minimum RMSD.

    Args:
        pred_coords: predicted CA coordinates, shape (5, 73, 3)
        ref_coords: reference CA coordinates, shape (5, 73, 3)

    Returns:
        (best_rmsd, best_permutation)
    """
    ref_flat = ref_coords.reshape(-1, 3)
    best_rmsd = float("inf")
    best_perm: tuple[int, ...] = (0, 1, 2, 3, 4)
    for perm in permutations(range(5)):
        pred_perm = pred_coords[list(perm)]
        pred_flat = pred_perm.reshape(-1, 3)
        rmsd = kabsch_rmsd(pred_flat, ref_flat)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
    return best_rmsd, best_perm


def mutate_sequence(seq: str, n_mutations: int = 1) -> str:
    """Mutate random positions in the sequence to random amino acids."""
    seq_list = list(seq)
    positions = random.sample(range(len(seq)), min(n_mutations, len(seq)))
    for pos in positions:
        current = seq_list[pos]
        candidates = [aa for aa in AMINO_ACIDS if aa != current]
        seq_list[pos] = random.choice(candidates)
    return "".join(seq_list)
