def compute_fitness(
    plddt: float, rmsd: float, w_plddt: float = 1.0, w_rmsd: float = 1.0
) -> float:
    """Fitness score = w_plddt * pLDDT - w_rmsd * RMSD.

    Higher is better. pLDDT is in [0, 1] (higher = more confident),
    RMSD is in angstroms (lower = closer to reference).
    """
    return w_plddt * plddt - w_rmsd * rmsd
