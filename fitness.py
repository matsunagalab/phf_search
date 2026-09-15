def compute_fitness(
    plddt: float,
    rmsd: float,
    w_plddt: float = 1.0,
    w_rmsd: float = 1.0,
    *,
    amyloid: float | None = None,
    llps: float | None = None,
    aggrescan: float | None = None,
    w_amyloid: float = 0.0,
    w_llps: float = 0.0,
    w_aggrescan: float = 0.0,
) -> float:
    """Fitness score, higher is better.

        w_plddt * pLDDT - w_rmsd * RMSD
            + w_amyloid * amyloid + w_llps * LLPS + w_aggrescan * AGGRESCAN

    Structural terms (always present):
        plddt: AF2 confidence in [0, 1], higher = more confident
        rmsd: deviation from the reference structure in angstroms, lower = closer

    Sequence terms (optional):
        amyloid: amyloidogenicity probability in [0, 1] (esm_scores.ESMScorer)
        llps: LLPS propensity probability in [0, 1] (esm_scores.ESMScorer)
        aggrescan: an AGGRESCAN summary scalar (aggrescan.score). Unlike the
            two probabilities this is unbounded and can be negative, so its
            weight is not on the same scale as theirs.

    A term with weight 0 does not move the fitness, which is how a metric is
    reported without being optimized. Weights may be negative: `w_llps = -1.0`
    pushes the search away from phase-separating sequences. A None metric is
    dropped from the sum -- pass one only when it was actually computed.

    Everything past `w_rmsd` is keyword-only on purpose. Each new metric used to
    be inserted into the middle of the signature, which silently reassigned any
    caller's positional arguments -- a wrong objective with no error. Adding the
    next metric now cannot do that.
    """
    fitness = w_plddt * plddt - w_rmsd * rmsd
    if amyloid is not None:
        fitness += w_amyloid * amyloid
    if llps is not None:
        fitness += w_llps * llps
    if aggrescan is not None:
        fitness += w_aggrescan * aggrescan
    return fitness
