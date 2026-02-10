"""Main entry point for PHF Monte Carlo sequence search."""

import argparse
import json
import logging

import numpy as np

from mc_search import MonteCarloSearch
from predict import PHFPredictor

NATIVE_SEQ = "VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF"


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo search for PHF-forming sequences"
    )
    parser.add_argument(
        "--data-dir", default="params", help="AF2 parameters directory"
    )
    parser.add_argument(
        "--ref-coords",
        default="data/reference_ca_coords.npy",
        help="Reference CA coordinates (.npy)",
    )
    parser.add_argument("--initial-seq", default=NATIVE_SEQ, help="Starting sequence")
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

    # Load reference coordinates
    logger.info("Loading reference coordinates from %s", args.ref_coords)
    ref_coords = np.load(args.ref_coords)
    logger.info("Reference shape: %s", ref_coords.shape)

    # Setup predictor
    logger.info("Initializing AF2 predictor (data_dir=%s, recycles=%d)",
                args.data_dir, args.num_recycles)
    predictor = PHFPredictor(data_dir=args.data_dir, num_recycles=args.num_recycles)

    # Run MC search
    mc = MonteCarloSearch(
        predictor=predictor,
        ref_coords=ref_coords,
        initial_seq=args.initial_seq,
        temperature=args.temperature,
        n_mutations=args.n_mutations,
        w_plddt=args.w_plddt,
        w_rmsd=args.w_rmsd,
        save_interval=args.save_interval,
        structures_dir=args.structures_dir,
    )

    logger.info(
        "Starting MC search: %d steps, T=%.2f, %d mutations/step",
        args.n_steps, args.temperature, args.n_mutations,
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
        "initial_seq": args.initial_seq,
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
    logger.info("Best fitness: %.4f (pLDDT=%.4f, RMSD=%.2f)",
                summary["best_fitness"], summary["best_plddt"], summary["best_rmsd"])


if __name__ == "__main__":
    main()
