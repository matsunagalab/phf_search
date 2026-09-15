def compute_fitness(
    plddt: float,
    rmsd: float,
    w_plddt: float = 1.0,
    w_rmsd: float = 1.0,
    *,
    ptm: float | None = None,
    i_ptm: float | None = None,
    i_pae: float | None = None,
    tm_score: float | None = None,
    lddt: float | None = None,
    fnat: float | None = None,
    w_ptm: float = 0.0,
    w_iptm: float = 0.0,
    w_ipae: float = 0.0,
    w_tm: float = 0.0,
    w_lddt: float = 0.0,
    w_fnat: float = 0.0,
    amyloid: float | None = None,
    llps: float | None = None,
    aggrescan: float | None = None,
    ddg: float | None = None,
    mpnn: float | None = None,
    w_amyloid: float = 0.0,
    w_llps: float = 0.0,
    w_aggrescan: float = 0.0,
    w_ddg: float = 0.0,
    w_mpnn: float = 0.0,
) -> float:
    """Fitness score, higher is better.

        w_plddt * pLDDT - w_rmsd * RMSD + w_amyloid * amyloid
            + w_llps * LLPS + w_aggrescan * AGGRESCAN + w_ddg * ddG
            + w_mpnn * mpnn_score

    Structural terms:
        plddt: AF2 confidence in [0, 1], higher = more confident
        rmsd: deviation from the reference structure in angstroms, lower = closer
        ptm: predicted TM-score of the assembly in [0, 1], higher = better
        i_ptm: the same over inter-chain pairs -- the interface confidence,
            which for a fibril is what its defining contacts rest on
        i_pae: mean inter-chain predicted aligned error, normalized by 31 A, so
            in [0, 1] and **lower is better** -- a search that wants a confident
            interface needs a negative `w_ipae`

    Shape-fidelity terms (shape.compare, all higher = better, all in [0, 1]):
        tm_score: TM-score of the assembly against the reference; >0.5 is the
            conventional same-fold threshold
        lddt: local distance difference test, superposition-free, and the
            quantity pLDDT claims to predict
        fnat: fraction of the reference's inter-chain contacts recovered -- the
            interface analogue of a fraction-of-native-contacts Q, and `nan`
            for a single chain, where there is no interface

    Sequence terms (optional):
        amyloid: amyloidogenicity probability in [0, 1] (esm_scores.ESMScorer)
        llps: LLPS propensity probability in [0, 1] (esm_scores.ESMScorer)
        aggrescan: an AGGRESCAN summary scalar (aggrescan.score). Unlike the
            two probabilities this is unbounded and can be negative, so its
            weight is not on the same scale as theirs.
        ddg: predicted stability change in kcal/mol (ddg.DDGLookup), where
            **positive is destabilizing**. A search after stable designs
            therefore wants a negative `w_ddg`.
        mpnn: ProteinMPNN score (mpnn_score.MPNNScorer) -- mean negative log
            probability of the sequence given the backbone, so **lower is
            better** and a search that wants sequences ProteinMPNN likes needs
            a negative `w_mpnn`.

    A term with weight 0 is skipped outright, not multiplied by zero, so a
    metric that is merely being reported cannot affect the fitness even if it
    came back non-finite. That is the point of the `!= 0.0` guards below: `0.0 *
    nan` is `nan`, which would poison a run over a metric nobody asked to
    optimize. Weights may be negative: `w_llps = -1.0`
    pushes the search away from phase-separating sequences. A None metric is
    dropped from the sum -- pass one only when it was actually computed.

    Everything past `w_rmsd` is keyword-only on purpose. Each new metric used to
    be inserted into the middle of the signature, which silently reassigned any
    caller's positional arguments -- a wrong objective with no error. Adding the
    next metric now cannot do that.
    """
    fitness = w_plddt * plddt - w_rmsd * rmsd
    if ptm is not None and w_ptm != 0.0:
        fitness += w_ptm * ptm
    if i_ptm is not None and w_iptm != 0.0:
        fitness += w_iptm * i_ptm
    if i_pae is not None and w_ipae != 0.0:
        fitness += w_ipae * i_pae
    if tm_score is not None and w_tm != 0.0:
        fitness += w_tm * tm_score
    if lddt is not None and w_lddt != 0.0:
        fitness += w_lddt * lddt
    if fnat is not None and w_fnat != 0.0:
        fitness += w_fnat * fnat
    if amyloid is not None and w_amyloid != 0.0:
        fitness += w_amyloid * amyloid
    if llps is not None and w_llps != 0.0:
        fitness += w_llps * llps
    if aggrescan is not None and w_aggrescan != 0.0:
        fitness += w_aggrescan * aggrescan
    if ddg is not None and w_ddg != 0.0:
        fitness += w_ddg * ddg
    if mpnn is not None and w_mpnn != 0.0:
        fitness += w_mpnn * mpnn
    return fitness
