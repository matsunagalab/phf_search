"""Evaluation layer: one sequence in, all metrics and the fitness out.

Search strategies should call `SequenceEvaluator.evaluate()` rather than talking
to the predictor directly. Swapping Monte Carlo for a genetic algorithm then
means writing a new search loop only -- the metrics, their weights and the
bookkeeping stay identical, and runs from different strategies stay comparable.
"""

import logging

import numpy as np

import aggrescan
from fitness import compute_fitness
from utils import min_permutation_rmsd

logger = logging.getLogger(__name__)

# Which number from the per-residue profile enters the fitness.
AGGREGATIONS = ("mean", "max")


class SequenceEvaluator:
    """AF2 prediction + structural RMSD + optional sequence-level scores.

    Args:
        predictor: AF2Predictor (or anything with the same `predict()`)
        ref_coords: reference CA coordinates, shape (n_chains, n_residues, 3)
        w_plddt, w_rmsd: weights of the structural terms
        w_amyloid, w_llps, w_aggrescan: weights of the sequence terms; 0.0 keeps
            a metric out of the fitness while still reporting it. Negative
            weights penalize.
        scorer: ESMScorer, or None to skip the amyloid/LLPS metrics entirely.
            AGGRESCAN needs no scorer -- it is a table lookup and a sliding
            window, too cheap to be worth switching off, so it is always
            computed and recorded.
        amyloid_agg: "mean" or "max" of the per-residue amyloid profile. The
            LLPS term always uses the mean, which for the default whole-sequence
            LLPS scoring is simply the score itself.
        aggrescan_metric: which AGGRESCAN scalar enters the fitness; all of them
            are recorded regardless.
    """

    def __init__(
        self,
        predictor,
        ref_coords: np.ndarray,
        w_plddt: float = 1.0,
        w_rmsd: float = 1.0,
        w_amyloid: float = 0.0,
        w_llps: float = 0.0,
        w_aggrescan: float = 0.0,
        scorer=None,
        amyloid_agg: str = "mean",
        aggrescan_metric: str = "na4vss",
    ):
        if amyloid_agg not in AGGREGATIONS:
            raise ValueError(
                f"amyloid_agg must be one of {AGGREGATIONS}, got {amyloid_agg!r}"
            )
        if aggrescan_metric not in aggrescan.METRICS:
            raise ValueError(
                f"aggrescan_metric must be one of {aggrescan.METRICS}, "
                f"got {aggrescan_metric!r}"
            )

        self.predictor = predictor
        self.ref_coords = ref_coords
        self.w_plddt = w_plddt
        self.w_rmsd = w_rmsd
        self.w_amyloid = w_amyloid
        self.w_llps = w_llps
        self.w_aggrescan = w_aggrescan
        self.scorer = scorer
        self.amyloid_agg = amyloid_agg
        self.aggrescan_metric = aggrescan_metric

        if scorer is None and (w_amyloid != 0.0 or w_llps != 0.0):
            raise ValueError(
                "w_amyloid/w_llps are nonzero but no scorer was given; "
                "the fitness would silently drop those terms."
            )

    def evaluate(self, seq: str) -> dict:
        """Evaluate one sequence.

        Returns:
            dict with seq, plddt, rmsd, perm, pdb_str, fitness; the AGGRESCAN
            scalars (`aggrescan`, holding whichever one the fitness used, plus
            one `aggrescan_<name>` per quantity in aggrescan.SUMMARY_KEYS); and
            -- when a scorer is configured -- amyloid, amyloid_mean,
            amyloid_max, llps, llps_mean, llps_max.
        """
        result = self.predictor.predict(seq)
        rmsd, perm = min_permutation_rmsd(result["ca_coords"], self.ref_coords)

        record = {
            "seq": seq,
            "plddt": result["plddt"],
            "rmsd": rmsd,
            "perm": perm,
            "pdb_str": result["pdb_str"],
        }

        # Always on: a table lookup and a sliding window over ~73 numbers.
        agg = aggrescan.score(seq)
        record.update(
            {f"aggrescan_{key}": agg[key] for key in aggrescan.SUMMARY_KEYS}
        )
        record["aggrescan"] = agg[self.aggrescan_metric]

        amyloid = llps = None
        if self.scorer is not None:
            scores = self.scorer.score(seq)
            # Both the mean and the peak of each profile are always recorded, so
            # that choosing one for the objective still leaves the other
            # available for later analysis. `amyloid`/`llps` hold the aggregate
            # the fitness actually used.
            amyloid = scores[f"amyloid_{self.amyloid_agg}"]
            llps = scores["llps_mean"]
            record.update(
                amyloid=amyloid,
                amyloid_mean=scores["amyloid_mean"],
                amyloid_max=scores["amyloid_max"],
                llps=llps,
                llps_mean=scores["llps_mean"],
                llps_max=scores["llps_max"],
            )

        record["fitness"] = compute_fitness(
            plddt=result["plddt"],
            rmsd=rmsd,
            w_plddt=self.w_plddt,
            w_rmsd=self.w_rmsd,
            amyloid=amyloid,
            llps=llps,
            aggrescan=record["aggrescan"],
            w_amyloid=self.w_amyloid,
            w_llps=self.w_llps,
            w_aggrescan=self.w_aggrescan,
        )
        return record

    def format_metrics(self, record: dict) -> str:
        """Compact 'pLDDT=... RMSD=... amyloid(mean)=...' string for log lines.

        The aggregate is named so that a mean-mode log cannot be mistaken for a
        max-mode one.
        """
        parts = [f"pLDDT={record['plddt']:.4f}", f"RMSD={record['rmsd']:.2f}"]
        parts.append(f"aggrescan({self.aggrescan_metric})={record['aggrescan']:.3f}")
        if self.scorer is not None:
            parts.append(f"amyloid({self.amyloid_agg})={record['amyloid']:.3f}")
            parts.append(f"llps={record['llps']:.3f}")
        parts.append(f"fitness={record['fitness']:.4f}")
        return " ".join(parts)
