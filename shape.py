"""Shape fidelity: how much the predicted assembly looks like the target.

pLDDT, pTM, ipTM and iPAE are AF2's opinion of its own prediction -- they need
no reference and can be high for a confidently wrong shape. These metrics
compare against the experimental coordinates instead, and they exist because one
global RMSD hides where a failure is.

On native PHF tau the decomposition says something a single number cannot:

    global RMSD            34.97 A     "not similar", and nothing more
    per-chain RMSD         ~20 A       the single-chain C-shape is also wrong,
                                       so this is not only a stacking failure
    per-chain lDDT         0.44        local geometry is partly plausible
    global lDDT            0.13        the assembly is not
    Fnat                   0.001       of 2346 native inter-chain CA contacts,
                                       0.1% are recovered -- no stacking at all

What each one is for:

* **per-chain RMSD / TM-score** -- is the monomer fold right? Computed against
  one reference chain, so it is independent of how the chains are stacked.
* **lDDT** -- superposition-free. It compares intra-structure distances, so one
  badly placed chain cannot contaminate the whole number the way it does for a
  global Kabsch fit. It is also the quantity pLDDT predicts, which makes
  `plddt` vs `lddt` a direct check of whether AF2's confidence was earned.
* **Fnat** -- the fraction of the reference's inter-chain contacts that the
  prediction recovers: the interface analogue of the fraction-of-native-contacts
  Q used in folding. For a fibril this is the stacking, i.e. the thing that
  makes it a fibril.

Everything here works on the CA coordinates the pipeline already has, in
microseconds, with no new dependency.

Caveats worth keeping in mind:

* These use the 1:1 residue correspondence the design problem gives us, so no
  alignment is searched. For TM-score that matches the `TM-score` program's
  premise (fixed correspondence) and the iterative superposition search below
  follows it, but this is still an independent implementation, not TM-align.
* Fnat is defined on CA pairs here, not all heavy atoms as DockQ defines it.
  That is deliberate: a designed sequence has different side chains from the
  reference, so a side-chain contact set is not comparable between them. Do not
  report these as DockQ numbers.
"""

import numpy as np

# Fnat contact cutoff between CA atoms. DockQ uses 5 A over all heavy atoms;
# CA-CA contacts need a longer cutoff to describe the same neighbourhoods.
CONTACT_CUTOFF = 8.0

# lDDT inclusion radius and tolerance thresholds, as defined by Mariani et al.
LDDT_CUTOFF = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation taking centred `mobile` onto centred `target`."""
    h = mobile.T @ target
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def tm_score(pred: np.ndarray, ref: np.ndarray) -> float:
    """TM-score for a 1:1 correspondence, length-normalized, in [0, 1].

    Follows the TM-score program's approach: try superpositions seeded on
    fragments of decreasing length, refine each by re-selecting the residues
    that fit, and keep the best. Above ~0.5 two structures share a fold.

    A single global superposition would only give a lower bound, and the 0.5
    threshold would not be safe to quote against it.
    """
    n_res = len(ref)
    if n_res != len(pred):
        raise ValueError(f"{len(pred)} predicted vs {n_res} reference residues")
    if n_res < 3:
        return 0.0

    d0 = max(1.24 * (n_res - 15) ** (1 / 3) - 1.8, 0.5) if n_res > 15 else 0.5
    d0_search = float(np.clip(d0, 4.5, 8.0))

    best = 0.0
    seed_length = n_res
    while seed_length >= 4:
        for start in range(0, n_res - seed_length + 1, max(seed_length // 2, 1)):
            selected = np.arange(start, start + seed_length)
            for _ in range(20):
                if len(selected) < 3:
                    break
                # Score the whole chain under this superposition.
                all_dev = _fit_and_deviate(pred, ref, selected)
                best = max(best, float((1.0 / (1.0 + (all_dev / d0) ** 2)).mean()))
                nearby = np.where(all_dev < d0_search)[0]
                if len(nearby) < 3 or np.array_equal(nearby, selected):
                    break
                selected = nearby
        seed_length //= 2
    return best


def _fit_and_deviate(pred, ref, selected) -> np.ndarray:
    """Superimpose on `selected`, then measure every residue's deviation."""
    pred_centre = pred[selected].mean(0)
    ref_centre = ref[selected].mean(0)
    rotation = _kabsch_rotation(pred[selected] - pred_centre, ref[selected] - ref_centre)
    moved = (pred - pred_centre) @ rotation.T + ref_centre
    return np.linalg.norm(moved - ref, axis=1)


def lddt(pred: np.ndarray, ref: np.ndarray) -> float:
    """Local distance difference test in [0, 1]. No superposition involved."""
    if len(pred) != len(ref):
        raise ValueError(f"{len(pred)} predicted vs {len(ref)} reference residues")
    ref_distances = np.linalg.norm(ref[:, None] - ref[None, :], axis=-1)
    pred_distances = np.linalg.norm(pred[:, None] - pred[None, :], axis=-1)
    included = (ref_distances < LDDT_CUTOFF) & ~np.eye(len(ref), dtype=bool)
    if not included.any():
        return 0.0
    error = np.abs(pred_distances - included * ref_distances)[included]
    return float(np.mean([(error < t).mean() for t in LDDT_THRESHOLDS]))


def fnat(pred: np.ndarray, ref: np.ndarray) -> tuple[float, int] | None:
    """Fraction of the reference's inter-chain CA contacts that are recovered.

    Args:
        pred, ref: coordinates shaped (n_chains, n_residues, 3)

    Returns:
        (fnat, number of native inter-chain contacts), or None when there is no
        interface to score. Returning None rather than nan keeps the metric
        *absent* for a monomer, the way i_ptm and i_pae are: a nan would travel
        into the fitness, into results.json as a bare `NaN` that strict JSON
        parsers reject, and into plots as a gap that looks like a failure
        rather than an inapplicable question.
    """
    n_chains, n_res = ref.shape[0], ref.shape[1]
    if n_chains < 2:
        return None

    chain_of = np.repeat(np.arange(n_chains), n_res)
    inter_chain = chain_of[:, None] != chain_of[None, :]

    flat_ref = ref.reshape(-1, 3)
    flat_pred = pred.reshape(-1, 3)
    native = (
        np.linalg.norm(flat_ref[:, None] - flat_ref[None, :], axis=-1) < CONTACT_CUTOFF
    ) & inter_chain
    n_native = int(native.sum())
    if n_native == 0:
        return None

    recovered = (
        np.linalg.norm(flat_pred[:, None] - flat_pred[None, :], axis=-1) < CONTACT_CUTOFF
    ) & native
    return float(recovered.sum() / n_native), n_native


def compare(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Every shape-fidelity metric for one predicted assembly.

    Args:
        pred, ref: coordinates shaped (n_chains, n_residues, 3)

    Returns:
        dict with `tm_score` and `lddt` for the assembly as a whole, the same
        two plus `rmsd_chain` for the best single chain against one reference
        chain, and `fnat` -- which is **omitted entirely** when there is only
        one chain and therefore no interface.

        The per-chain values answer "is the monomer fold right"; the global ones
        answer "is the assembly right". They come apart, and the gap is the
        useful part.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"shapes differ: {pred.shape} vs {ref.shape}")

    flat_pred, flat_ref = pred.reshape(-1, 3), ref.reshape(-1, 3)
    reference_chain = ref[0]

    # Best-matching single chain: the reference chains are copies, so comparing
    # every predicted chain against one of them and keeping the best answers
    # "did any chain come out right" without depending on the stacking order.
    chain_scores = [
        (
            tm_score(chain, reference_chain),
            lddt(chain, reference_chain),
            float(np.sqrt((_fit_and_deviate(chain, reference_chain, np.arange(len(chain))) ** 2).mean())),
        )
        for chain in pred
    ]
    best_tm, best_lddt, best_rmsd = max(chain_scores, key=lambda s: s[0])

    metrics = {
        "tm_score": tm_score(flat_pred, flat_ref),
        "lddt": lddt(flat_pred, flat_ref),
        "tm_score_chain": best_tm,
        "lddt_chain": best_lddt,
        "rmsd_chain": best_rmsd,
    }
    # Absent, not nan, when there is no interface. `n_native_contacts` depends
    # only on the reference, so it is a property of the run and is reported by
    # native_contact_count() instead of being repeated in every record.
    interface = fnat(pred, ref)
    if interface is not None:
        metrics["fnat"] = interface[0]
    return metrics


def native_contact_count(ref: np.ndarray) -> int:
    """How many inter-chain CA contacts the reference has -- the Fnat denominator.

    A constant for a given target, so it belongs in a run summary rather than
    in every history record.
    """
    interface = fnat(ref, ref)
    return 0 if interface is None else interface[1]
