"""Monte Carlo search engine for PHF sequence optimization."""

import logging
import math
import os
import random

import numpy as np

from fitness import compute_fitness
from utils import min_permutation_rmsd, mutate_sequence

logger = logging.getLogger(__name__)


class MonteCarloSearch:
    def __init__(
        self,
        predictor,
        ref_coords: np.ndarray,
        initial_seq: str,
        temperature: float = 1.0,
        n_mutations: int = 1,
        w_plddt: float = 1.0,
        w_rmsd: float = 1.0,
        save_interval: int = 1,
        structures_dir: str | None = None,
    ):
        self.predictor = predictor
        self.ref_coords = ref_coords
        self.current_seq = initial_seq
        self.temperature = temperature
        self.n_mutations = n_mutations
        self.w_plddt = w_plddt
        self.w_rmsd = w_rmsd
        self.save_interval = save_interval
        self.structures_dir = structures_dir

        if self.structures_dir is not None:
            os.makedirs(self.structures_dir, exist_ok=True)

        # Will be set during initial evaluation
        self.current_fitness: float = float("-inf")
        self.current_plddt: float = 0.0
        self.current_rmsd: float = float("inf")

        # Tracking
        self.best_seq = initial_seq
        self.best_fitness: float = float("-inf")
        self.best_plddt: float = 0.0
        self.best_rmsd: float = float("inf")
        self.history: list[dict] = []
        self.n_accepted = 0
        self.n_total = 0

    def _save_pdb(self, step: int, pdb_str: str) -> None:
        """Save PDB string to file in structures_dir."""
        if self.structures_dir is None:
            return
        filename = os.path.join(self.structures_dir, f"step_{step:04d}.pdb")
        with open(filename, "w") as f:
            f.write(pdb_str)
        logger.debug("Saved %s", filename)

    def _evaluate(self, seq: str) -> dict:
        """Run AF2 prediction and compute fitness."""
        result = self.predictor.predict(seq)
        rmsd, perm = min_permutation_rmsd(result["ca_coords"], self.ref_coords)
        fitness = compute_fitness(result["plddt"], rmsd, self.w_plddt, self.w_rmsd)
        return {
            "seq": seq,
            "plddt": result["plddt"],
            "rmsd": rmsd,
            "fitness": fitness,
            "perm": perm,
            "pdb_str": result["pdb_str"],
        }

    def step(self) -> dict:
        """One MC step: mutate -> predict -> evaluate -> accept/reject."""
        new_seq = mutate_sequence(self.current_seq, self.n_mutations)
        result = self._evaluate(new_seq)

        delta = result["fitness"] - self.current_fitness
        self.n_total += 1

        if delta > 0 or random.random() < math.exp(delta / self.temperature):
            # Accept
            self.current_seq = new_seq
            self.current_fitness = result["fitness"]
            self.current_plddt = result["plddt"]
            self.current_rmsd = result["rmsd"]
            self.n_accepted += 1
            accepted = True

            if result["fitness"] > self.best_fitness:
                self.best_seq = new_seq
                self.best_fitness = result["fitness"]
                self.best_plddt = result["plddt"]
                self.best_rmsd = result["rmsd"]
        else:
            accepted = False

        if self.n_total % self.save_interval == 0:
            self._save_pdb(self.n_total, result["pdb_str"])

        record = {
            "step": self.n_total,
            "seq": new_seq,
            "plddt": result["plddt"],
            "rmsd": result["rmsd"],
            "fitness": result["fitness"],
            "accepted": accepted,
            "current_seq": self.current_seq,
            "current_fitness": self.current_fitness,
        }
        self.history.append(record)
        return record

    def run(self, n_steps: int, log_interval: int = 10) -> dict:
        """Run MC search for n_steps.

        Returns summary dict with best results and full history.
        """
        # Initial evaluation
        logger.info("Evaluating initial sequence...")
        init_result = self._evaluate(self.current_seq)
        self.current_fitness = init_result["fitness"]
        self.current_plddt = init_result["plddt"]
        self.current_rmsd = init_result["rmsd"]
        self.best_seq = self.current_seq
        self.best_fitness = self.current_fitness
        self.best_plddt = self.current_plddt
        self.best_rmsd = self.current_rmsd

        self._save_pdb(0, init_result["pdb_str"])

        logger.info(
            "Initial: pLDDT=%.4f RMSD=%.2f fitness=%.4f",
            self.current_plddt,
            self.current_rmsd,
            self.current_fitness,
        )

        for i in range(1, n_steps + 1):
            record = self.step()

            if i % log_interval == 0 or i == n_steps:
                accept_rate = self.n_accepted / self.n_total if self.n_total > 0 else 0
                logger.info(
                    "Step %d/%d: pLDDT=%.4f RMSD=%.2f fitness=%.4f "
                    "accepted=%s rate=%.2f best_fitness=%.4f",
                    i,
                    n_steps,
                    record["plddt"],
                    record["rmsd"],
                    record["fitness"],
                    record["accepted"],
                    accept_rate,
                    self.best_fitness,
                )

        return {
            "best_seq": self.best_seq,
            "best_fitness": self.best_fitness,
            "best_plddt": self.best_plddt,
            "best_rmsd": self.best_rmsd,
            "final_seq": self.current_seq,
            "final_fitness": self.current_fitness,
            "n_accepted": self.n_accepted,
            "n_total": self.n_total,
            "accept_rate": self.n_accepted / self.n_total if self.n_total > 0 else 0,
            "history": self.history,
        }
