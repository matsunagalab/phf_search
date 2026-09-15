"""AGGRESCAN aggregation propensity, computed from sequence alone.

Reimplementation of the published algorithm (Conchillo-Sole et al., BMC
Bioinformatics 8:65, 2007). Its per-residue scale was measured in vivo, from
the intracellular aggregation of mutants at the central position of the central
hydrophobic cluster of the amyloid-beta peptide (Sanchez de Groot et al., BMC
Struct Biol 5:18, 2005).

This is the cheapest of the repository's amyloid indices, and deliberately the
most transparent one:

    esm_scores.py   a protein language model's learned 2560-dim representation
    aggrescan.py    twenty numbers and a sliding window

Both claim to score "aggregation propensity", and they do not have to agree.
Keeping a maximally simple index next to a learned one is what makes the
disagreement legible -- see the README.

NOT the official AGGRESCAN server, and two details cannot be reproduced from the
publication at all:

* **Terminal residues.** Additional file 2 says "the 2 extreme residues are
  given specific a4v to account for charge effects" without giving the value.
  Here they inherit the nearest window centre like the other off-centre
  terminal residues, so profiles can differ at the first and last residue.
* **The areas.** The same file specifies "trapezoidal integration (midpoint
  rule)" -- two different methods. `aat`, `ta` and `thsa` below are the
  trapezoid of the threshold-clipped profile, which is one reading of that
  sentence; where the profile crosses the threshold inside an interval this
  slightly overestimates the area above it. The scale, window rule, threshold,
  hot-spot rule and every other quantity follow the paper directly, and the
  a3v/a4v/hot-spot parts are checked against hand computation -- the areas are
  not, because there is nothing unambiguous to check them against.

Absolute values from this module should not be reported as server output.
"""

import numpy as np

# Additional file 1 of Conchillo-Sole et al. (2007): relative experimental
# aggregation propensities of the 20 natural amino acids. Higher = more
# aggregation-prone.
A3V = {
    "I": 1.822,
    "F": 1.754,
    "V": 1.594,
    "L": 1.380,
    "Y": 1.159,
    "W": 1.037,
    "M": 0.910,
    "C": 0.604,
    "A": -0.036,
    "T": -0.159,
    "S": -0.294,
    "P": -0.334,
    "G": -0.535,
    "K": -0.931,
    "H": -1.033,
    "Q": -1.231,
    "R": -1.240,
    "N": -1.302,
    "E": -1.412,
    "D": -1.836,
}

# Hot-spot threshold: the a3v of each residue type weighted by its frequency in
# SwissProt, averaged. Additional file 2 gives its value directly.
HST = -0.02

# A hot spot needs this many consecutive residues above the threshold.
MIN_HOT_SPOT = 5

# Proline breaks a hot spot: it is an aggregation breaker.
BREAKER = "P"

# Which scalars may steer the search. na4vss is the default because it is the
# closest analogue of the "TANGO total" that the amyloid design literature
# reports. nhs/nnhs are deliberately NOT here: counting hot spots takes only a
# handful of distinct values over a fixed-length target (three, across all 1387
# single mutants of the 73-residue PHF core), so as an objective it is a step
# function whose steps are threshold noise. They are still recorded.
METRICS = ("na4vss", "a3vsa", "thsar")

# Scalars worth keeping for every candidate of a run.
SUMMARY_KEYS = (
    "a3vsa",
    "na4vss",
    "nhs",
    "nnhs",
    "aatr",
    "thsar",
    "ta",
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def window_size(n_res: int) -> int:
    """a4v window length, which the paper ponderates by sequence length."""
    if n_res <= 75:
        return 5
    if n_res <= 175:
        return 7
    if n_res <= 300:
        return 9
    return 11


def _a4v(a3v: np.ndarray) -> np.ndarray:
    """Sliding-window average of a3v, assigned to the central residue.

    Residues too close to a terminus to centre a window inherit the first or
    last window centre (see the module docstring on the missing terminal charge
    correction).
    """
    n_res = len(a3v)
    window = min(window_size(n_res), n_res)
    if window % 2 == 0:  # only possible when the sequence is shorter than 5
        window -= 1

    centres = np.convolve(a3v, np.ones(window) / window, mode="valid")
    half = window // 2
    return np.concatenate(
        [np.full(half, centres[0]), centres, np.full(half, centres[-1])]
    )


def _hot_spots(a4v: np.ndarray, sequence: str) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of the hot spots."""
    above = a4v > HST
    spots = []
    start = None
    for i, (hot, residue) in enumerate(zip(above, sequence)):
        if hot and residue != BREAKER:
            start = i if start is None else start
            continue
        if start is not None:
            if i - start >= MIN_HOT_SPOT:
                spots.append((start, i))
            start = None
    if start is not None and len(a4v) - start >= MIN_HOT_SPOT:
        spots.append((start, len(a4v)))
    return spots


def score(sequence: str) -> dict:
    """AGGRESCAN quantities for one sequence.

    Returns:
        dict with the summary scalars named as in the paper --

            a3vsa   mean a3v over the sequence
            a4vss   sum of a4v;  na4vss  the same per 100 residues
            nhs     number of hot spots;  nnhs  the same per 100 residues
            aat     area of the profile above the threshold;  aatr  per residue
            thsa    total hot-spot area;  thsar  per residue
            ta      total profile area, with the threshold as the zero axis

        plus `a4v_profile` (the per-residue profile) and `hot_spots`, a list of
        one dict per hot spot carrying its 1-based inclusive span, sequence,
        `hsa` (hot-spot area), `nhsa` (that area per residue), `a4vahs` (mean
        a4v inside it) and `margin`.

        `margin` is not from the paper: it is the smallest amount by which the
        profile clears the threshold inside the hot spot, and it says whether
        the hot spot is a real feature or an artifact of where HST falls. On
        native PHF tau the VQIVYK hot spot clears by 0.50 while the second one
        clears by 0.0026, and a threshold of -0.015 would erase the latter.

    Raises:
        ValueError: on an empty sequence, or a character outside the 20
            standard residues.
    """
    if not sequence:
        raise ValueError("Cannot score an empty sequence")

    unknown = sorted({residue for residue in sequence if residue not in A3V})
    if unknown:
        raise ValueError(
            f"No a3v value for {unknown}: the AGGRESCAN scale covers only the "
            "20 standard residues. prepare_reference.py emits 'X' for "
            "non-standard residues, so a target containing modified residues "
            "needs its data/<pdb>_sequence.txt cleaned before searching."
        )

    a3v = np.array([A3V[residue] for residue in sequence], dtype=float)
    a4v = _a4v(a3v)
    n_res = len(sequence)

    above_threshold = np.clip(a4v - HST, 0.0, None)

    hot_spots = []
    for start, end in _hot_spots(a4v, sequence):
        hsa = float(_trapezoid(above_threshold[start:end]))
        hot_spots.append(
            {
                "start": start + 1,  # 1-based inclusive
                "end": end,
                "sequence": sequence[start:end],
                "hsa": hsa,
                "nhsa": hsa / n_res,
                "a4vahs": float(a4v[start:end].mean()),
                "margin": float((a4v[start:end] - HST).min()),
            }
        )
    thsa = float(sum(spot["hsa"] for spot in hot_spots))

    return {
        "a3vsa": float(a3v.mean()),
        "a4vss": float(a4v.sum()),
        "na4vss": float(a4v.sum() / n_res * 100.0),
        "nhs": len(hot_spots),
        "nnhs": float(len(hot_spots) / n_res * 100.0),
        "aat": float(_trapezoid(above_threshold)),
        "aatr": float(_trapezoid(above_threshold) / n_res),
        "thsa": thsa,
        "thsar": thsa / n_res,
        "ta": float(_trapezoid(a4v - HST)),
        "a4v_profile": a4v,
        "hot_spots": hot_spots,
    }
