"""Evaluation layer: one sequence in, all metrics and the fitness out.

Search strategies should call `SequenceEvaluator.evaluate()` rather than talking
to the predictor directly. Swapping Monte Carlo for a genetic algorithm then
means writing a new search loop only -- the metrics, their weights and the
bookkeeping stay identical, and runs from different strategies stay comparable.
"""

import logging

import numpy as np

import aggrescan
import homology
import shape as shape_metrics
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
        w_plddt, w_rmsd, w_ptm, w_iptm, w_ipae: weights of the structural
            terms. ptm/i_ptm/i_pae come free from the same AF2 pass and are
            always recorded; i_pae is lower-is-better, so `w_ipae` should be
            negative to reward a confident interface
        w_seq_recovery, w_blosum62: weights of the sequence-homology terms,
            measured against `reference_sequence`
        w_tm, w_lddt, w_fnat: weights of the shape-fidelity terms
        w_amyloid, w_llps, w_aggrescan, w_ddg, w_mpnn: weights of the
            sequence terms.
            0.0 keeps a metric out of the fitness while still reporting it.
            Negative weights penalize -- and ddG is positive-is-destabilizing,
            so a search after stable designs wants `w_ddg` negative.
        scorer: ESMScorer, or None to skip the amyloid/LLPS metrics entirely.
            AGGRESCAN needs no scorer -- it is a table lookup and a sliding
            window, too cheap to be worth switching off, so it is always
            computed and recorded.
        amyloid_agg: "mean" or "max" of the per-residue amyloid profile. The
            LLPS term always uses the mean, which for the default whole-sequence
            LLPS scoring is simply the score itself.
        mpnn_scorer: mpnn_score.MPNNScorer, or None to skip the inverse-folding
            term. Its score is lower-is-better, so `w_mpnn` should be negative
            to reward sequences ProteinMPNN finds plausible.
        ddg_lookup: ddg.DDGLookup, or None to skip the stability term. Its
            numbers are a sum of independent single-mutant predictions against
            a fixed reference backbone -- see ddg.py for what that does and
            does not support.
        reference_sequence: the target's own sequence, enabling seq_recovery
            and blosum62. None skips them.
        aggrescan_metric: which AGGRESCAN scalar enters the fitness; all of them
            are recorded regardless.
    """

    def __init__(
        self,
        predictor,
        ref_coords: np.ndarray,
        w_plddt: float = 1.0,
        w_rmsd: float = 1.0,
        w_ptm: float = 0.0,
        w_iptm: float = 0.0,
        w_ipae: float = 0.0,
        w_seq_recovery: float = 0.0,
        w_blosum62: float = 0.0,
        w_tm: float = 0.0,
        w_lddt: float = 0.0,
        w_fnat: float = 0.0,
        w_amyloid: float = 0.0,
        w_llps: float = 0.0,
        w_aggrescan: float = 0.0,
        w_ddg: float = 0.0,
        w_mpnn: float = 0.0,
        scorer=None,
        ddg_lookup=None,
        mpnn_scorer=None,
        reference_sequence: str | None = None,
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
        self.w_ptm = w_ptm
        self.w_iptm = w_iptm
        self.w_ipae = w_ipae
        self.w_seq_recovery = w_seq_recovery
        self.w_blosum62 = w_blosum62
        self.w_tm = w_tm
        self.w_lddt = w_lddt
        self.w_fnat = w_fnat
        self.w_amyloid = w_amyloid
        self.w_llps = w_llps
        self.w_aggrescan = w_aggrescan
        self.w_ddg = w_ddg
        self.w_mpnn = w_mpnn
        self.scorer = scorer
        self.ddg_lookup = ddg_lookup
        self.mpnn_scorer = mpnn_scorer
        self.reference_sequence = reference_sequence
        self.amyloid_agg = amyloid_agg
        self.aggrescan_metric = aggrescan_metric

        if scorer is None and (w_amyloid != 0.0 or w_llps != 0.0):
            raise ValueError(
                "w_amyloid/w_llps are nonzero but no scorer was given; "
                "the fitness would silently drop those terms."
            )
        if reference_sequence is None and (w_seq_recovery != 0.0 or w_blosum62 != 0.0):
            raise ValueError(
                "w_seq_recovery/w_blosum62 are nonzero but no reference "
                "sequence was given; the fitness would silently drop them."
            )
        if mpnn_scorer is None and w_mpnn != 0.0:
            raise ValueError(
                "w_mpnn is nonzero but no ProteinMPNN scorer was given; the "
                "fitness would silently drop that term."
            )
        if ddg_lookup is None and w_ddg != 0.0:
            raise ValueError(
                "w_ddg is nonzero but no ddG matrix was given; the fitness "
                "would silently drop that term. Run precompute_ddg.py for this "
                "target."
            )

    def evaluate(self, seq: str) -> dict:
        """Evaluate one sequence.

        Returns:
            dict with seq, plddt, rmsd, perm, pdb_str, fitness; the AGGRESCAN
            scalars (`aggrescan`, holding whichever one the fitness used, plus
            one `aggrescan_<name>` per quantity in aggrescan.SUMMARY_KEYS);
            when a ddG matrix is configured -- ddg, ddg_max, ddg_n_mutations;
            when the native sequence is known -- seq_recovery, blosum62,
            blosum62_normalized; the shape-fidelity metrics from
            shape.compare; when a ProteinMPNN scorer is configured --
            mpnn_score; and when an ESM scorer is configured -- amyloid, amyloid_mean,
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
        # Free: the same AF2 forward pass produced them. i_ptm and i_pae speak
        # to the inter-chain interface, which is what a fibril is made of, and
        # are absent for a single chain.
        for key in ("ptm", "iptm", "ipae", "ipae_angstrom"):
            if key in result:
                record[key] = result[key]

        # Sequence homology to the target's own sequence: no model, no
        # structure, ~37 us. Recorded whenever the native sequence is known.
        if self.reference_sequence is not None:
            record.update(homology.compare(seq, self.reference_sequence))

        # Shape fidelity against the reference, as opposed to AF2's confidence
        # in itself. One global RMSD cannot say whether a failure is in the
        # monomer fold or in the stacking; these separate the two.
        record.update(shape_metrics.compare(result["ca_coords"], self.ref_coords))

        # Always on: a table lookup and a sliding window over ~73 numbers.
        agg = aggrescan.score(seq)
        record.update(
            {f"aggrescan_{key}": agg[key] for key in aggrescan.SUMMARY_KEYS}
        )
        record["aggrescan"] = agg[self.aggrescan_metric]

        ddg_value = None
        if self.ddg_lookup is not None:
            stability = self.ddg_lookup.score(seq)
            ddg_value = stability["ddg"]
            record.update(
                ddg=ddg_value,
                ddg_max=stability["ddg_max"],
                ddg_n_mutations=stability["n_mutations"],
            )

        mpnn_value = None
        if self.mpnn_scorer is not None:
            inverse_folding = self.mpnn_scorer.score(seq)
            mpnn_value = inverse_folding["mpnn_score"]
            record["mpnn_score"] = mpnn_value

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
            ptm=record.get("ptm"),
            iptm=record.get("iptm"),
            ipae=record.get("ipae"),
            seq_recovery=record.get("seq_recovery"),
            blosum62=record.get("blosum62"),
            tm_score=record.get("tm_score"),
            lddt=record.get("lddt"),
            fnat=record.get("fnat"),
            w_ptm=self.w_ptm,
            w_iptm=self.w_iptm,
            w_ipae=self.w_ipae,
            w_seq_recovery=self.w_seq_recovery,
            w_blosum62=self.w_blosum62,
            w_tm=self.w_tm,
            w_lddt=self.w_lddt,
            w_fnat=self.w_fnat,
            amyloid=amyloid,
            llps=llps,
            aggrescan=record["aggrescan"],
            ddg=ddg_value,
            mpnn=mpnn_value,
            w_amyloid=self.w_amyloid,
            w_llps=self.w_llps,
            w_aggrescan=self.w_aggrescan,
            w_ddg=self.w_ddg,
            w_mpnn=self.w_mpnn,
        )
        # A non-finite fitness makes both branches of the Metropolis criterion
        # false, so the search would reject every candidate and still report
        # success. Whatever produced it, stop here rather than run blind.
        if not np.isfinite(record["fitness"]):
            raise ValueError(
                f"non-finite fitness {record['fitness']} for {seq}; metrics: "
                + ", ".join(
                    f"{k}={record[k]}"
                    for k in ("plddt", "rmsd", "ptm", "iptm", "ipae",
                              "tm_score", "lddt", "fnat", "seq_recovery",
                              "blosum62", "aggrescan", "ddg", "mpnn_score")
                    if k in record
                )
            )
        return record

    def format_metrics(self, record: dict) -> str:
        """Compact 'pLDDT=... RMSD=... amyloid(mean)=...' string for log lines.

        The aggregate is named so that a mean-mode log cannot be mistaken for a
        max-mode one.
        """
        parts = [f"pLDDT={record['plddt']:.4f}", f"RMSD={record['rmsd']:.2f}"]
        if "iptm" in record:
            parts.append(f"ipTM={record['iptm']:.3f}")
        if "ipae_angstrom" in record:
            parts.append(f"ipae={record['ipae_angstrom']:.1f}A")
        # Each key is checked on its own: a metric that does not apply to the
        # target is absent from the record, not present-and-nan, so testing one
        # key and reading another is how this breaks.
        if "seq_recovery" in record:
            parts.append(f"recov={record['seq_recovery']:.3f}")
        if "tm_score" in record:
            parts.append(f"TM={record['tm_score']:.3f}")
        if "lddt" in record:
            parts.append(f"lDDT={record['lddt']:.3f}")
        if "fnat" in record:
            parts.append(f"Fnat={record['fnat']:.3f}")
        parts.append(f"aggrescan({self.aggrescan_metric})={record['aggrescan']:.3f}")
        if self.ddg_lookup is not None:
            parts.append(f"ddG={record['ddg']:+.2f}")
        if self.mpnn_scorer is not None:
            parts.append(f"mpnn={record['mpnn_score']:.3f}")
        if self.scorer is not None:
            parts.append(f"amyloid({self.amyloid_agg})={record['amyloid']:.3f}")
            parts.append(f"llps={record['llps']:.3f}")
        parts.append(f"fitness={record['fitness']:.4f}")
        return " ".join(parts)
