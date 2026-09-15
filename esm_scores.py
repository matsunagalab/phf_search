"""Sequence-level amyloid and LLPS scores from Lobo et al., PNAS 123:e2531932123 (2026).

Both metrics work the same way: embed a peptide with ESM2-3B (layer 36,
mean-pooled over residues) and push the 2560-dim vector through a logistic
regression head, giving a probability in [0, 1].

  amyloid  how amyloidogenic the peptide looks
  LLPS     how likely the sequence is to drive liquid-liquid phase separation

The upstream CLIs score one sequence per process. Here the search loop calls
this for every candidate, so ESM2-3B is loaded once and shared between the two
heads, and repeated sequences are cached. The scoring itself calls the upstream
functions, so the numbers match `amyloid-predict` / `llps-predict`.

Sequences longer than a probe length are handled the way the upstream
per-residue CLIs do it: slide a window of each probe length along the sequence,
score every window, average the window scores back onto the residues they
cover, then reduce that per-residue profile to one number per sequence.
"""

import logging
import os
from collections import OrderedDict
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HEAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "esm_heads"
)

# Amyloid classifier name -> vendored checkpoint file (see models/esm_heads/README.md)
AMYLOID_HEADS = {
    "general": "general_model_latest.pt",
    "6aa": "6aa_model_latest.pt",
    "6aa-FETA": "6aa_FETA_model_latest.pt",
    "10aa": "10aa_model_latest.pt",
    "15aa": "15aa_model_latest.pt",
}
LLPS_HEAD = "LLPS_model_latest.pt"

# 6 aa and 15 aa match the two amyloid heads trained on hexapeptides and on
# 15 aa tau fragments -- the sizes the published classifiers know best.
DEFAULT_AMYLOID_PROBES = (6, 15)
# Empty means "score the whole sequence in one shot": the LLPS head was trained
# on entire IDRs, so a 73-residue construct is already in its training regime.
DEFAULT_LLPS_PROBES = ()

ESM_MODEL_SIZE = "3B"


class ESMScorer:
    """amyloid and LLPS probabilities for a candidate sequence.

    Args:
        head_dir: directory holding the vendored logistic-regression checkpoints
        amyloid_probes: window lengths for the amyloid profile; () scores the
            whole sequence with a single head
        llps_probes: window lengths for the LLPS profile; () scores the whole
            sequence (the default -- the LLPS head was trained on whole IDRs)
        amyloid_policy: which amyloid head scores a window of a given length.
            "matched" uses 6aa/10aa/15aa where the length matches, else
            `general_classifier`; "general" always uses `general_classifier`;
            "matched-feta6" is "matched" with the FETA variant for hexapeptides.
        stride: window step in residues
        use_gpu: run ESM2 on CUDA (the model is ~11 GB in float32)
        esm_weights_dir: torch hub directory for the ESM2 checkpoint; None uses
            the default cache, downloading ~5.7 GB on first use
        toks_per_batch: ESM2 batching; lower it if the GPU runs out of memory
        cache_size: how many sequences to remember (0 disables caching)
    """

    def __init__(
        self,
        head_dir: str = DEFAULT_HEAD_DIR,
        amyloid_probes: tuple[int, ...] = DEFAULT_AMYLOID_PROBES,
        llps_probes: tuple[int, ...] = DEFAULT_LLPS_PROBES,
        amyloid_policy: str = "matched",
        general_classifier: str = "general",
        stride: int = 1,
        use_gpu: bool = True,
        esm_weights_dir: str | None = None,
        toks_per_batch: int = 4096,
        truncation_seq_length: int = 1022,
        cache_size: int = 2048,
    ):
        self.head_dir = head_dir
        self.amyloid_probes = tuple(amyloid_probes)
        self.llps_probes = tuple(llps_probes)
        self.amyloid_policy = amyloid_policy
        self.general_classifier = general_classifier
        self.stride = stride
        self.use_gpu = use_gpu
        self.esm_weights_dir = esm_weights_dir
        self.toks_per_batch = toks_per_batch
        self.truncation_seq_length = truncation_seq_length
        self.cache_size = cache_size

        # Validated here rather than at the first score() call, which happens
        # only after AF2 has loaded its parameters and run a prediction.
        for name, probes in (
            ("amyloid_probes", self.amyloid_probes),
            ("llps_probes", self.llps_probes),
        ):
            bad = [p for p in probes if p <= 0]
            if bad:
                raise ValueError(f"{name} must be positive integers, got {bad}")
        for name, value in (
            ("stride", stride),
            ("toks_per_batch", toks_per_batch),
            ("truncation_seq_length", truncation_seq_length),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        if amyloid_policy not in ("matched", "general", "matched-feta6"):
            raise ValueError(f"Unsupported amyloid_policy: {amyloid_policy!r}")
        if not os.path.isdir(head_dir):
            raise FileNotFoundError(
                f"Classifier directory not found: {head_dir}. "
                "It ships with the repository as models/esm_heads/."
            )

        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._loaded = False
        self._esm_model = None
        self._alphabet = None
        self._layer = None
        self._amyloid_heads: dict = {}
        self._llps_head = None
        self._warned_probes: set[tuple] = set()

    # -- loading ---------------------------------------------------------

    def _load(self) -> None:
        """Load ESM2-3B and the logistic-regression heads (first call only)."""
        if self._loaded:
            return

        # Imported lazily: torch + a 5.7 GB checkpoint should not be touched by
        # runs that leave these metrics switched off.
        import torch

        from amyloid_predict.inference import (
            configure_torch_hub_dir,
            load_esm_model,
            load_torch_lr_from_pt,
        )

        # Each upstream package carries its own copy of the same embedding code
        # (the two `inference.py` modules differ only in import placement and a
        # temporary variable). One shared ESM2-3B therefore serves both metrics;
        # the LLPS head is still loaded through its own package so that the
        # dependency declaring it is the one that reads it.
        from llps_predict.inference import (
            load_torch_lr_from_pt as load_llps_lr_from_pt,
        )

        if self.use_gpu and not torch.cuda.is_available():
            logger.warning("No CUDA device visible to torch; running ESM2 on CPU")
            self.use_gpu = False

        configure_torch_hub_dir(self.esm_weights_dir)

        logger.info(
            "Loading ESM2-%s (this downloads ~5.7 GB on first use)", ESM_MODEL_SIZE
        )
        with redirect_stdout(StringIO()):
            self._esm_model, self._alphabet, self._layer = load_esm_model(
                ESM_MODEL_SIZE
            )

        for name, filename in AMYLOID_HEADS.items():
            path = os.path.join(self.head_dir, filename)
            if os.path.exists(path):
                self._amyloid_heads[name] = load_torch_lr_from_pt(path)

        llps_path = os.path.join(self.head_dir, LLPS_HEAD)
        if not os.path.exists(llps_path):
            raise FileNotFoundError(f"LLPS classifier not found: {llps_path}")
        self._llps_head = load_llps_lr_from_pt(llps_path)

        device = "GPU" if self.use_gpu else "CPU"
        logger.info(
            "ESM scorers ready on %s: amyloid probes=%s, LLPS probes=%s",
            device,
            self.amyloid_probes or "whole sequence",
            self.llps_probes or "whole sequence",
        )
        self._loaded = True

    # -- fragments -------------------------------------------------------

    def _usable_probes(self, probes: tuple[int, ...], n_res: int) -> tuple[int, ...]:
        """Drop probe lengths that do not fit in the sequence."""
        usable = tuple(p for p in probes if p <= n_res)
        if usable != probes and (probes, n_res) not in self._warned_probes:
            self._warned_probes.add((probes, n_res))
            logger.warning(
                "Probe lengths %s exceed the %d-residue sequence; using %s",
                [p for p in probes if p > n_res],
                n_res,
                usable or "the whole sequence",
            )
        return usable

    def _build_fragments(self, tag: str, sequence: str, probes: tuple[int, ...]):
        """Windows of each probe length; an empty `probes` gives one whole-sequence window."""
        from amyloid_predict.per_res import FragmentRecord, build_fragments

        probes = self._usable_probes(probes, len(sequence))
        if not probes:
            return [
                FragmentRecord(
                    name=f"{tag}_full",
                    sequence=sequence,
                    probe_length=len(sequence),
                    start_idx=0,
                    end_idx=len(sequence),
                )
            ], (len(sequence),)
        return build_fragments(tag, sequence, list(probes), self.stride), probes

    # -- scoring ---------------------------------------------------------

    def score(self, sequence: str) -> dict:
        """Score one sequence.

        Returns:
            dict with keys:
                amyloid_mean, amyloid_max: mean / peak of the per-residue amyloid profile
                llps_mean, llps_max: same for LLPS
                amyloid_per_residue, llps_per_residue: the profiles themselves

            Mean and peak are both always present, under names that say which
            is which -- picking one for the fitness must not cost the other.
        """
        cached = self._cache.get(sequence)
        if cached is not None:
            self._cache.move_to_end(sequence)
            # Shallow copy: a caller mutating the record must not corrupt the cache.
            return dict(cached)

        self._load()

        from amyloid_predict.inference import embed_sequences, predict_with_heads
        from amyloid_predict.per_res import (
            aggregate_per_residue_scores,
            pick_classifier_for_length,
        )

        amyloid_frags, amyloid_probes = self._build_fragments(
            "amyloid", sequence, self.amyloid_probes
        )
        llps_frags, llps_probes = self._build_fragments(
            "llps", sequence, self.llps_probes
        )

        # Both metrics go through ESM2 together: their windows are batched by
        # length, so this is a small number of forward passes (two for the
        # default config) rather than one per window.
        fragments = amyloid_frags + llps_frags

        # embed_sequences prints its own truncation warning, which the stdout
        # redirect below would swallow, so warn here instead.
        too_long = [f.name for f in fragments if len(f.sequence) > self.truncation_seq_length]
        if too_long:
            logger.warning(
                "%d fragment(s) exceed truncation_seq_length=%d and will be "
                "truncated for ESM2 (e.g. %s)",
                len(too_long),
                self.truncation_seq_length,
                too_long[0],
            )

        with redirect_stdout(StringIO()):
            embeddings = embed_sequences(
                names=[f.name for f in fragments],
                sequences=[f.sequence for f in fragments],
                esm_model=self._esm_model,
                alphabet=self._alphabet,
                layer=self._layer,
                use_gpu=self.use_gpu,
                toks_per_batch=self.toks_per_batch,
                truncation_seq_length=self.truncation_seq_length,
            )

        n_amyloid = len(amyloid_frags)
        amyloid_scores = self._score_amyloid(
            embeddings[:n_amyloid],
            amyloid_frags,
            predict_with_heads,
            pick_classifier_for_length,
        )
        llps_scores = predict_with_heads(
            embeddings[n_amyloid:], {"llps": self._llps_head}
        )["llps"]

        result = {}
        for key, frags, probes, scores in (
            ("amyloid", amyloid_frags, amyloid_probes, amyloid_scores),
            ("llps", llps_frags, llps_probes, llps_scores),
        ):
            _, _, per_residue = aggregate_per_residue_scores(
                len(sequence), list(probes), frags, scores
            )
            result[f"{key}_mean"] = float(np.nanmean(per_residue))
            result[f"{key}_max"] = float(np.nanmax(per_residue))
            result[f"{key}_per_residue"] = per_residue

        self._remember(sequence, result)
        return dict(result)

    def _score_amyloid(
        self, embeddings, fragments, predict_with_heads, pick_classifier_for_length
    ) -> np.ndarray:
        """Apply the head that matches each window length (see `amyloid_policy`)."""
        chosen = [
            pick_classifier_for_length(
                f.probe_length, self.amyloid_policy, self.general_classifier
            )
            for f in fragments
        ]
        missing = sorted(set(chosen) - set(self._amyloid_heads))
        if missing:
            raise FileNotFoundError(
                f"Missing amyloid classifier checkpoint(s) in {self.head_dir}: {missing}"
            )
        by_head = predict_with_heads(
            embeddings, {name: self._amyloid_heads[name] for name in set(chosen)}
        )
        return np.array([by_head[name][i] for i, name in enumerate(chosen)])

    def _remember(self, sequence: str, result: dict) -> None:
        if self.cache_size <= 0:
            return
        self._cache[sequence] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)


def parse_probe_lengths(value: str) -> tuple[int, ...]:
    """Parse a CLI probe-length list: "6,15" -> (6, 15); "" or "none" -> ()."""
    value = value.strip()
    if not value or value.lower() == "none":
        return ()
    return tuple(int(part) for part in value.split(",") if part.strip())
