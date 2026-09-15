"""ProteinMPNN score: how likely the candidate sequence is, given the backbone.

This is the inverse-folding model's own opinion of a sequence -- the mean
negative log probability it assigns to the residues actually present, given the
target coordinates:

    score = mean over positions of -log p(residue | backbone)

**Lower is better**: ProteinMPNN finds the sequence more plausible for that
fold. It says nothing about whether the sequence *folds* that way, only whether
an inverse-folding model would have proposed it, so it is a different question
from both RMSD and the amyloid indices.

`identity` comes along as the fraction of positions still matching the
reference sequence -- cheap, and useful for seeing how far a search has drifted.

**The score is stochastic.** ProteinMPNN decodes in a random order and the score
depends on that order, so `n_eval` orders are averaged. Measured on the 5-chain
5O3L reference:

    n_eval    spread (SD) of the returned value    cost
    1         0.019                                ~18 ms
    3         0.013                                ~53 ms
    10        0.006                                ~180 ms

Single-point mutations move this score by a comparable amount, so `n_eval=1`
would hand the search mostly noise. The default is 10. (The earlier tau
polymorph analysis used 3 for new calculations and 32 for its reference table.)

For scale, measured on the same reference at `n_eval=5`: native 0.588,
poly-alanine 0.733, poly-glycine 0.791, poly-aspartate 0.836, the native
sequence reversed 0.949, poly-proline 1.136, poly-tryptophan 1.246. So the
usable span is roughly 0.6, not the ~0.15 that poly-alanine alone suggests.

Two implementation notes, both about ColabDesign, which is where this gets its
ProteinMPNN -- already a dependency, already carrying the `v_48_020` weights, so
no new package and no new checkpoint:

1. It is a JAX reimplementation and **measurably not equivalent** to the
   original PyTorch tool. On the 5O3L reference with the native sequence it
   averages 0.5815 over 30 decoding orders where the PyTorch code gives 0.5731.
   That 0.0084 gap is not noise: the spread of a *single* evaluation is ~0.021,
   but the quantity being compared is a 30-order average, whose standard error
   is ~0.004 (measured: five independent 30-order averages spread by 0.003).
   The gap is therefore about **2x the standard error -- a resolvable, apparently
   systematic difference**, and its cause was not identified. It is *not* the
   20-vs-21 letter normalization: ColabDesign log-softmaxes over all 21 and then
   slices, and rescoring from its raw logits over 21 classes reproduces its own
   number exactly. Do not mix numbers from the two implementations.

2. Its residue ordering is AlphaFold's (`ARNDCQEGHILKMFPSTWYV`), *not*
   ProteinMPNN's own (`ACDEFGHIKLMNPQRSTVWYX`), so the integer sequence in
   `_inputs["S"]` must be decoded with ColabDesign's own table.

3. `mk_mpnn_model.score()` **overwrites the model's stored reference sequence in
   place** (`copy_dict` is `tree_map(lambda y: y, ...)`, which shares the leaf
   arrays). Its own `seqid` output is therefore always 1.0 -- it compares the
   scored sequence against itself -- and the model object cannot be shared with
   anything that reads `_inputs["S"]` afterwards. This module snapshots the
   reference sequence before the first call and computes `identity` itself.
"""

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "v_48_020"
DEFAULT_N_EVAL = 10


class MPNNScorer:
    """ProteinMPNN score of a candidate sequence on one fixed backbone.

    Args:
        pdb_filename: backbone to score against
        chains: comma-separated chain IDs, e.g. "A,C,E,G,I"
        homooligomer: tie the chains so one sequence is scored in every copy.
            The rest of this pipeline is homo-oligomer only, so this is True.
        n_eval: decoding orders to average over; see the module docstring for
            the noise it buys
        model_name: ColabDesign ProteinMPNN weights
        seed: seeds the model's RNG, so a whole run is reproducible
        cache_size: how many sequences to remember (0 disables)
    """

    def __init__(
        self,
        pdb_filename: str,
        chains: str,
        homooligomer: bool = True,
        n_eval: int = DEFAULT_N_EVAL,
        model_name: str = DEFAULT_MODEL,
        seed: int = 37,
        cache_size: int = 4096,
    ):
        if n_eval < 1:
            raise ValueError(f"n_eval must be at least 1, got {n_eval}")

        self.pdb_filename = pdb_filename
        self.chains = chains
        self.homooligomer = homooligomer
        self.n_eval = n_eval
        self.model_name = model_name
        self.seed = seed
        self.cache_size = cache_size

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._model = None
        self.reference: str | None = None

    def _load(self):
        """Build the model, featurize the backbone, and snapshot its sequence."""
        if self._model is not None:
            return self._model

        # Imported here so that --help and runs with this switched off do not
        # pay for it; predict.py imports ColabDesign anyway.
        from colabdesign.mpnn import mk_mpnn_model

        model = mk_mpnn_model(model_name=self.model_name, seed=self.seed)
        model.prep_inputs(
            pdb_filename=self.pdb_filename,
            chain=self.chains,
            homooligomer=self.homooligomer,
        )

        # Taken before the first score(), which would overwrite it (see the
        # module docstring). One chain's worth is enough: the chains are tied.
        # The ordering comes from ColabDesign rather than being hardcoded: it
        # uses AlphaFold's "ARNDCQEGHILKMFPSTWYV", not ProteinMPNN's own
        # "ACDEFGHIKLMNPQRSTVWYX", and getting that wrong silently produces a
        # plausible-looking but wrong reference sequence.
        from colabdesign.mpnn.model import order_aa

        native = "".join(order_aa.get(int(i), "X") for i in model._inputs["S"])
        n_chains = len(self.chains.split(","))
        self.reference = native[: len(native) // n_chains]

        logger.info(
            "ProteinMPNN %s ready on %s chains %s (n_eval=%d, %d residues)",
            self.model_name,
            self.pdb_filename,
            self.chains,
            self.n_eval,
            len(self.reference),
        )
        self._model = model
        return model

    def score(self, sequence: str) -> dict:
        """Score one sequence.

        Returns:
            dict with `mpnn_score` (mean negative log probability over
            `n_eval` decoding orders, lower is better) and `mpnn_identity`
            (fraction of positions matching the reference sequence).
        """
        cached = self._cache.get(sequence)
        if cached is not None:
            self._cache.move_to_end(sequence)
            return dict(cached)

        model = self._load()
        if len(sequence) != len(self.reference):
            raise ValueError(
                f"sequence has {len(sequence)} residues but the backbone "
                f"{self.pdb_filename} has {len(self.reference)} per chain"
            )

        total = 0.0
        for _ in range(self.n_eval):
            total += float(model.score(seq=sequence)["score"])

        matches = sum(a == b for a, b in zip(sequence, self.reference))
        result = {
            "mpnn_score": total / self.n_eval,
            "mpnn_identity": matches / len(self.reference),
        }

        if self.cache_size > 0:
            self._cache[sequence] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return dict(result)
