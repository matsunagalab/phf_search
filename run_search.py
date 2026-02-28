"""Main entry point for Monte Carlo sequence search."""

import argparse
import json
import logging
import os

import numpy as np

from mc_search import MonteCarloSearch
from predict import AF2Predictor

# Default: PHF tau (5O3L) for backward compatibility
DEFAULT_PDB_ID = "5O3L"
DEFAULT_CHAINS = "A,C,E,G,I"
NATIVE_SEQ = (
    "VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF"
)


def main():
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
        "--log-interval", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="Save PDB every N steps"
    )
    parser.add_argument(
        "--structures-dir", default="structures", help="Directory for PDB structures"
    )
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

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

    # Run MC search
    mc = MonteCarloSearch(
        predictor=predictor,
        ref_coords=ref_coords,
        initial_seq=initial_seq,
        temperature=args.temperature,
        n_mutations=args.n_mutations,
        w_plddt=args.w_plddt,
        w_rmsd=args.w_rmsd,
        save_interval=args.save_interval,
        structures_dir=args.structures_dir,
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
        "final_seq": summary["final_seq"],
        "final_fitness": summary["final_fitness"],
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
        "Best fitness: %.4f (pLDDT=%.4f, RMSD=%.2f)",
        summary["best_fitness"],
        summary["best_plddt"],
        summary["best_rmsd"],
    )


if __name__ == "__main__":
    main()
