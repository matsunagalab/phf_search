"""Main entry point for Monte Carlo sequence search."""

import argparse
import json
import logging
import os

import numpy as np

from aggrescan import METRICS as AGGRESCAN_METRICS

# Default: PHF tau (5O3L) for backward compatibility
DEFAULT_PDB_ID = "5O3L"
DEFAULT_CHAINS = "A,C,E,G,I"
NATIVE_SEQ = (
    "VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo search for protein sequences matching a target structure"
    )
    # Target specification
    parser.add_argument(
        "--pdb-id",
        default=DEFAULT_PDB_ID,
        help="Target PDB ID (default: 5O3L)",
    )
    parser.add_argument(
        "--chains",
        default=DEFAULT_CHAINS,
        help="Comma-separated chain IDs (default: A,C,E,G,I)",
    )

    # Paths and model
    parser.add_argument(
        "--data-dir", default="params", help="AF2 parameters directory"
    )
    parser.add_argument(
        "--ref-coords",
        default=None,
        help="Reference CA coordinates (.npy). Auto-derived from --pdb-id/--chains if not set.",
    )
    parser.add_argument(
        "--initial-seq",
        default=None,
        help="Starting sequence. Auto-loaded from prepared reference if not set.",
    )

    # Search parameters
    parser.add_argument(
        "--n-steps", type=int, default=1000, help="Number of MC steps"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="MC temperature"
    )
    parser.add_argument(
        "--n-mutations", type=int, default=1, help="Mutations per step"
    )
    parser.add_argument(
        "--num-recycles", type=int, default=3, help="AF2 recycles"
    )
    parser.add_argument(
        "--w-plddt", type=float, default=1.0, help="pLDDT weight in fitness"
    )
    parser.add_argument(
        "--w-rmsd", type=float, default=1.0, help="RMSD weight in fitness"
    )
    parser.add_argument(
        "--w-aggrescan",
        type=float,
        default=0.0,
        help="AGGRESCAN weight in fitness; negative penalizes. Always computed "
        "and recorded regardless (default: 0.0, report only)",
    )
    parser.add_argument(
        "--aggrescan-metric",
        choices=list(AGGRESCAN_METRICS),
        default="na4vss",
        help="Which AGGRESCAN scalar enters the fitness; all are recorded "
        "(default: na4vss)",
    )

    # Sequence-level scores (amyloid-predict / LLPS-predict, Lobo et al. 2026)
    esm = parser.add_argument_group("ESM2-based sequence scores")
    esm.add_argument(
        "--esm-scores",
        choices=["auto", "on", "off"],
        default="auto",
        help="Compute amyloid/LLPS scores. auto = on when a weight is nonzero "
        "(default: auto)",
    )
    esm.add_argument(
        "--w-amyloid",
        type=float,
        default=0.0,
        help="Amyloidogenicity weight in fitness; negative penalizes (default: 0.0, report only)",
    )
    esm.add_argument(
        "--w-llps",
        type=float,
        default=0.0,
        help="LLPS propensity weight in fitness; negative penalizes (default: 0.0, report only)",
    )
    esm.add_argument(
        "--amyloid-probe-lengths",
        default="6,15",
        help="Sliding-window lengths for the amyloid profile, or 'none' for the "
        "whole sequence (default: 6,15)",
    )
    esm.add_argument(
        "--amyloid-agg",
        choices=["mean", "max"],
        default="mean",
        help="Which number from the per-residue amyloid profile enters the "
        "fitness; both are always recorded (default: mean)",
    )
    esm.add_argument(
        "--esm-cpu", action="store_true", help="Run ESM2 on CPU instead of GPU"
    )
    esm.add_argument(
        "--esm-toks-per-batch",
        type=int,
        default=4096,
        help="ESM2 batch size in tokens; lower it on GPU OOM (default: 4096)",
    )
    # Remaining knobs (LLPS windowing, the amyloid classifier policy, the
    # checkpoint directory) are constructor arguments of ESMScorer rather than
    # flags: their defaults are the ones the published classifiers were trained
    # for, and changing them is a deliberate, code-level decision.

    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Save PDB every N steps"
    )
    parser.add_argument(
        "--structures-dir", default="structures", help="Directory for PDB structures"
    )
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    return parser


def esm_scores_enabled(args) -> bool:
    if args.esm_scores == "auto":
        return args.w_amyloid != 0.0 or args.w_llps != 0.0
    return args.esm_scores == "on"


def build_scorer(parser, args):
    """Construct the ESMScorer, or None when the scores are switched off.

    Called before AF2 is even imported: the constructor checks its settings but
    loads no weights, so a bad setting fails in a fraction of a second instead
    of surfacing at the first scored candidate -- a minute later, past AF2's
    parameter load, one AF2 prediction and the ESM2-3B load.
    """
    if not esm_scores_enabled(args):
        if args.w_amyloid != 0.0 or args.w_llps != 0.0:
            parser.error(
                "--esm-scores off leaves the amyloid/LLPS terms uncomputed, but "
                "--w-amyloid/--w-llps are nonzero."
            )
        return None

    from esm_scores import ESMScorer, parse_probe_lengths

    try:
        return ESMScorer(
            amyloid_probes=parse_probe_lengths(args.amyloid_probe_lengths),
            use_gpu=not args.esm_cpu,
            toks_per_batch=args.esm_toks_per_batch,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(f"bad ESM scoring settings: {exc}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Built first, while it is still cheap to fail.
    scorer = build_scorer(parser, args)

    # JAX grabs most of the GPU on import. When ESM2 shares the device it needs
    # room, so switch preallocation off before AF2 is imported. Set the variable
    # yourself to override.
    if scorer is not None and not args.esm_cpu:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from evaluate import SequenceEvaluator
    from mc_search import MonteCarloSearch
    from predict import AF2Predictor

    # Derive target configuration
    chains = [c.strip() for c in args.chains.split(",")]
    n_chains = len(chains)
    chain_label = "".join(chains).lower()
    prefix = f"{args.pdb_id.lower()}_{chain_label}"

    # Resolve reference coordinates path
    if args.ref_coords is not None:
        ref_coords_path = args.ref_coords
    else:
        ref_coords_path = os.path.join("data", f"{prefix}_ca_coords.npy")

    # Load reference coordinates
    logger.info("Loading reference coordinates from %s", ref_coords_path)
    ref_coords = np.load(ref_coords_path)
    logger.info("Reference shape: %s", ref_coords.shape)

    n_residues = ref_coords.shape[1]

    # Resolve initial sequence
    if args.initial_seq is not None:
        initial_seq = args.initial_seq
    else:
        seq_path = os.path.join("data", f"{prefix}_sequence.txt")
        if os.path.exists(seq_path):
            with open(seq_path) as f:
                initial_seq = f.read().strip()
            logger.info(
                "Loaded initial sequence from %s (%d residues)",
                seq_path,
                len(initial_seq),
            )
        elif args.pdb_id == DEFAULT_PDB_ID and args.chains == DEFAULT_CHAINS:
            initial_seq = NATIVE_SEQ
            logger.info(
                "Using default PHF native sequence (%d residues)", len(initial_seq)
            )
        else:
            raise ValueError(
                f"No initial sequence found. Run prepare_reference.py for "
                f"{args.pdb_id} chains {args.chains}, or provide --initial-seq."
            )

    # prepare_reference.py writes 'X' where a chain holds a non-standard
    # residue, and neither the mutation operator nor AGGRESCAN can act on one.
    # Caught here, before AF2 spends time loading its parameters.
    from utils import AMINO_ACIDS

    non_standard = sorted(set(initial_seq) - set(AMINO_ACIDS))
    if non_standard:
        parser.error(
            f"initial sequence contains non-standard residue(s) {non_standard}, "
            "which the pipeline cannot score. Edit the sequence file "
            "(prepare_reference.py writes 'X' for modified residues) or pass a "
            "cleaned --initial-seq."
        )

    # Setup predictor
    logger.info(
        "Initializing AF2 predictor (length=%d, copies=%d, recycles=%d)",
        n_residues,
        n_chains,
        args.num_recycles,
    )
    predictor = AF2Predictor(
        data_dir=args.data_dir,
        num_recycles=args.num_recycles,
        length=n_residues,
        copies=n_chains,
    )

    evaluator = SequenceEvaluator(
        predictor=predictor,
        ref_coords=ref_coords,
        w_plddt=args.w_plddt,
        w_rmsd=args.w_rmsd,
        w_amyloid=args.w_amyloid,
        w_llps=args.w_llps,
        w_aggrescan=args.w_aggrescan,
        scorer=scorer,
        amyloid_agg=args.amyloid_agg,
        aggrescan_metric=args.aggrescan_metric,
    )
    logger.info(
        "Fitness terms: pLDDT, RMSD, AGGRESCAN%s",
        ", amyloid, LLPS" if scorer is not None else "",
    )

    # Run MC search
    mc = MonteCarloSearch(
        initial_seq=initial_seq,
        temperature=args.temperature,
        n_mutations=args.n_mutations,
        save_interval=args.save_interval,
        structures_dir=args.structures_dir,
        evaluator=evaluator,
    )

    logger.info(
        "Starting MC search: %d steps, T=%.2f, %d mutations/step",
        args.n_steps,
        args.temperature,
        args.n_mutations,
    )
    summary = mc.run(n_steps=args.n_steps, log_interval=args.log_interval)

    # Save results
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    output = {
        "args": vars(args),
        "initial_seq": initial_seq,
        "best_seq": summary["best_seq"],
        "best_fitness": summary["best_fitness"],
        "best_plddt": summary["best_plddt"],
        "best_rmsd": summary["best_rmsd"],
        "best_metrics": {k: _convert(v) for k, v in summary["best_metrics"].items()},
        "final_seq": summary["final_seq"],
        "final_fitness": summary["final_fitness"],
        "final_metrics": {k: _convert(v) for k, v in summary["final_metrics"].items()},
        "accept_rate": summary["accept_rate"],
        "history": [
            {k: _convert(v) for k, v in record.items()}
            for record in summary["history"]
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=_convert)

    logger.info("Results saved to %s", args.output)
    logger.info("Best sequence: %s", summary["best_seq"])
    logger.info(
        "Best fitness: %.4f (%s)",
        summary["best_fitness"],
        " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in summary["best_metrics"].items()
        ),
    )


if __name__ == "__main__":
    main()
