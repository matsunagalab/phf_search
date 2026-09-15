"""Monte Carlo search engine for PHF sequence optimization."""

import logging
import math
import os
import random

import numpy as np

from evaluate import SequenceEvaluator
from utils import mutate_sequence

logger = logging.getLogger(__name__)

# An evaluator record minus these is exactly what belongs in the history: every
# metric it reports, whatever they are, without the bulky structure fields. A
# new metric therefore needs no change here.
NOT_METRICS = ("seq", "perm", "pdb_str")


def metrics_of(result: dict) -> dict:
    """The metric fields of an evaluator record."""
    return {k: v for k, v in result.items() if k not in NOT_METRICS}


class MonteCarloSearch:
    """Metropolis Monte Carlo over sequence space.

    Pass an `evaluator` to control which metrics are computed and how they are
    weighted; without one, an evaluator is built from `predictor`, `ref_coords`
    and the pLDDT/RMSD weights, i.e. structure terms only.

    `designable` restricts which sequence indices may be mutated; None means
    all of them. A replacement search strategy has to honour it the same way,
    or `--fix-pos` silently stops working.

    `w_plddt` and `w_rmsd` belong to the evaluator. Passing them *alongside* an
    evaluator is refused rather than silently ignored -- a search that quietly
    optimizes a different objective than the caller asked for is worse than a
    failure at startup.
    """

    def __init__(
        self,
        predictor=None,
        ref_coords: np.ndarray | None = None,
        initial_seq: str = "",
        temperature: float = 1.0,
        n_mutations: int = 1,
        w_plddt: float | None = None,
        w_rmsd: float | None = None,
        save_interval: int = 1,
        structures_dir: str | None = None,
        evaluator: SequenceEvaluator | None = None,
        designable: list[int] | None = None,
    ):
        if evaluator is None:
            if predictor is None or ref_coords is None:
                raise ValueError(
                    "Provide either an evaluator or both predictor and ref_coords."
                )
            evaluator = SequenceEvaluator(
                predictor=predictor,
                ref_coords=ref_coords,
                w_plddt=1.0 if w_plddt is None else w_plddt,
                w_rmsd=1.0 if w_rmsd is None else w_rmsd,
            )
        elif w_plddt is not None or w_rmsd is not None:
            raise ValueError(
                "w_plddt/w_rmsd were passed together with an evaluator, which "
                "already carries its own weights. Set them on the evaluator."
            )

        self.evaluator = evaluator
        self.predictor = evaluator.predictor
        self.ref_coords = evaluator.ref_coords
        self.current_seq = initial_seq
        self.temperature = temperature
        self.n_mutations = n_mutations
        self.w_plddt = evaluator.w_plddt
        self.w_rmsd = evaluator.w_rmsd
        self.save_interval = save_interval
        self.structures_dir = structures_dir
        self.designable = designable

        if self.structures_dir is not None:
            os.makedirs(self.structures_dir, exist_ok=True)

        # Will be set during initial evaluation
        self.current_fitness: float = float("-inf")
        self.current_plddt: float = 0.0
        self.current_rmsd: float = float("inf")
        self.current_metrics: dict = {}

        # Tracking
        self.best_seq = initial_seq
        self.best_fitness: float = float("-inf")
        self.best_plddt: float = 0.0
        self.best_rmsd: float = float("inf")
        self.best_metrics: dict = {}
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
        """Alias for `self.evaluator.evaluate(seq)`."""
        return self.evaluator.evaluate(seq)

    def _adopt(self, result: dict) -> None:
        """Make the candidate in `result` the current state."""
        self.current_seq = result["seq"]
        self.current_fitness = result["fitness"]
        self.current_plddt = result["plddt"]
        self.current_rmsd = result["rmsd"]
        self.current_metrics = metrics_of(result)

    def _record_best(self, result: dict) -> None:
        """Record `result` as the best candidate so far."""
        self.best_seq = result["seq"]
        self.best_fitness = result["fitness"]
        self.best_plddt = result["plddt"]
        self.best_rmsd = result["rmsd"]
        self.best_metrics = metrics_of(result)

    def step(self) -> dict:
        """One MC step: mutate -> predict -> evaluate -> accept/reject."""
        new_seq = mutate_sequence(
            self.current_seq, self.n_mutations, designable=self.designable
        )
        result = self._evaluate(new_seq)

        delta = result["fitness"] - self.current_fitness
        self.n_total += 1

        if delta > 0 or random.random() < math.exp(delta / self.temperature):
            # Accept
            self._adopt(result)
            self.n_accepted += 1
            accepted = True

            if result["fitness"] > self.best_fitness:
                self._record_best(result)
        else:
            accepted = False

        if self.n_total % self.save_interval == 0:
            self._save_pdb(self.n_total, result["pdb_str"])

        record = {
            "step": self.n_total,
            "seq": new_seq,
            **metrics_of(result),
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
        initial_metrics = metrics_of(init_result)
        self._adopt(init_result)
        self._record_best(init_result)

        self._save_pdb(0, init_result["pdb_str"])

        logger.info("Initial: %s", self.evaluator.format_metrics(init_result))

        for i in range(1, n_steps + 1):
            record = self.step()

            if i % log_interval == 0 or i == n_steps:
                accept_rate = self.n_accepted / self.n_total if self.n_total > 0 else 0
                logger.info(
                    "Step %d/%d: %s accepted=%s rate=%.2f best_fitness=%.4f",
                    i,
                    n_steps,
                    self.evaluator.format_metrics(record),
                    record["accepted"],
                    accept_rate,
                    self.best_fitness,
                )

        return {
            "initial_metrics": initial_metrics,
            "best_seq": self.best_seq,
            "best_fitness": self.best_fitness,
            "best_plddt": self.best_plddt,
            "best_rmsd": self.best_rmsd,
            "best_metrics": self.best_metrics,
            "final_seq": self.current_seq,
            "final_fitness": self.current_fitness,
            "final_metrics": self.current_metrics,
            "n_accepted": self.n_accepted,
            "n_total": self.n_total,
            "accept_rate": self.n_accepted / self.n_total if self.n_total > 0 else 0,
            "history": self.history,
        }
