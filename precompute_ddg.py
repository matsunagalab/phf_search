"""Precompute a ThermoMPNN ddG matrix for a reference structure. Run once.

ThermoMPNN (Dieckhaus et al., PNAS 121:e2314851121, 2024; MIT licensed) predicts
the stability change of a point mutation from structure. Its selling point here
is that it scores every single mutation of a structure in one pass: for the
5-chain PHF reference that is 7300 mutations in ~4 s on a GPU.

So it never has to run inside the search. This script writes the whole
(n_chains, n_residues, 20) matrix to models/ddg/, and `ddg.py` turns a candidate
sequence into a number by adding up the entries for the positions it mutated --
microseconds, no torch, no weights. The matrix is ~30 kB and is committed, so a
run against the default target needs nothing from this script.

You only need to run it to add a new target. ThermoMPNN is not on PyPI, so clone
it once and point this script at the checkout:

    git clone https://github.com/Kuhlman-Lab/ThermoMPNN
    uv run python precompute_ddg.py --thermompnn-dir ThermoMPNN

Nothing else is installed: this loads ThermoMPNN's `TransferModel` directly
rather than through its PyTorch Lightning wrapper, which avoids pulling in
pytorch-lightning, wandb, omegaconf and tqdm for what is a forward pass.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from provenance import git_identity, sha256_file

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "ddg"
)

# The hyperparameters of the released checkpoint, copied from ThermoMPNN's
# analysis/custom_inference.py. They have to match or the state dict will not
# load, which is why load_state_dict below is strict.
MODEL_CFG = {
    "hidden_dims": [64, 32],
    "subtract_mut": True,
    "num_final_layers": 2,
    "freeze_weights": True,
    "load_pretrained": True,
    "lightattn": True,
}


class _Cfg(dict):
    """Attribute access over a dict, standing in for ThermoMPNN's OmegaConf cfg.

    `TransferModel` reads seven fields (`platform.thermompnn_dir` and six under
    `model`) and may set `cfg.decoding_order`; that is the whole contract, so
    this replaces the dependency.

    A missing field raises AttributeError rather than returning None -- except
    for names that `dict` already defines. A future upstream `cfg.update` or
    `cfg.keys` would silently resolve to the dict method instead of raising, so
    this is not a general-purpose config object.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@dataclass
class _Mutation:
    """ThermoMPNN's Mutation dataclass, redefined to avoid importing its
    datasets module (which pulls in pandas and tqdm)."""

    position: int
    wildtype: str
    mutation: str
    ddG: float | None = None
    pdb: str | None = ""


def load_model(thermompnn_dir: str, device: torch.device):
    """ThermoMPNN's TransferModel with the released weights, sans Lightning."""
    thermompnn_dir = os.path.abspath(thermompnn_dir)
    checkpoint = os.path.join(thermompnn_dir, "models", "thermoMPNN_default.pt")
    for path in (thermompnn_dir, checkpoint):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Clone ThermoMPNN and pass --thermompnn-dir:\n"
                "  git clone https://github.com/Kuhlman-Lab/ThermoMPNN"
            )

    sys.path.insert(0, thermompnn_dir)
    from transfer_model import TransferModel  # noqa: E402

    cfg = _Cfg(
        platform=_Cfg(thermompnn_dir=thermompnn_dir), model=_Cfg(**MODEL_CFG)
    )
    model = TransferModel(cfg)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key[len("model.") :]: value
        for key, value in state["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def site_saturation(model, pdb_path: str, chains: list[str]) -> tuple:
    """ddG for every (position, amino acid) of every chain, in one pass.

    Returns (matrix, sequence) where matrix has shape
    (n_chains, n_residues, 20) indexed by ALPHABET, and sequence is one chain's
    worth of residues.
    """
    from protein_mpnn_utils import alt_parse_PDB  # noqa: E402

    parsed = alt_parse_PDB(pdb_path, chains)
    concat_seq = parsed[0]["seq"]
    n_chains = len(chains)
    found = [c for c in chains if f"seq_chain_{c}" in parsed[0]]
    if len(found) != n_chains:
        raise ValueError(
            f"{pdb_path} does not contain chain(s) "
            f"{sorted(set(chains) - set(found))}; it has {found}"
        )
    if len(concat_seq) % n_chains:
        raise ValueError(
            f"{len(concat_seq)} residues do not divide into {n_chains} chains; "
            "this script assumes a homo-oligomer with equal-length chains."
        )
    n_res = len(concat_seq) // n_chains

    sequences = {concat_seq[i * n_res : (i + 1) * n_res] for i in range(n_chains)}
    if len(sequences) != 1:
        raise ValueError(
            f"Chains {chains} do not share one sequence: {sorted(sequences)}"
        )

    mutations = [
        _Mutation(position=i, wildtype=concat_seq[i], mutation=aa, pdb=parsed[0]["name"])
        for i in range(len(concat_seq))
        if concat_seq[i] != "-"
        for aa in ALPHABET
    ]
    print(f"Scoring {len(mutations)} mutations over {len(concat_seq)} residues...")
    with torch.no_grad():
        predictions, _ = model(parsed, mutations)

    index = {aa: i for i, aa in enumerate(ALPHABET)}
    matrix = np.full((n_chains, n_res, len(ALPHABET)), np.nan, dtype=np.float32)
    for mutation, prediction in zip(mutations, predictions):
        chain, position = divmod(mutation.position, n_res)
        matrix[chain, position, index[mutation.mutation]] = prediction["ddG"].item()

    return matrix, concat_seq[:n_res]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdb-id", default="5O3L", help="Target PDB ID")
    parser.add_argument("--chains", default="A,C,E,G,I", help="Comma-separated chains")
    parser.add_argument(
        "--reference-pdb",
        default=None,
        help="Structure to score (default: data/<pdb_id>_<chains>.pdb from "
        "prepare_reference.py)",
    )
    parser.add_argument(
        "--thermompnn-dir",
        default=os.environ.get("THERMOMPNN_DIR", "ThermoMPNN"),
        help="ThermoMPNN checkout (default: $THERMOMPNN_DIR or ./ThermoMPNN)",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

    chains = [c.strip() for c in args.chains.split(",")]
    prefix = f"{args.pdb_id.lower()}_{''.join(chains).lower()}"
    pdb_path = args.reference_pdb or os.path.join("data", f"{prefix}.pdb")
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(
            f"{pdb_path} not found. Run prepare_reference.py for "
            f"{args.pdb_id} chains {args.chains} first."
        )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    provenance = {
        "schema_version": 1,
        "source_pdb_sha256": sha256_file(pdb_path),
        "checkpoint_sha256": sha256_file(os.path.join(
            args.thermompnn_dir, "models", "thermoMPNN_default.pt")),
        "thermompnn": git_identity(args.thermompnn_dir),
        "generator": git_identity(os.path.dirname(os.path.abspath(__file__))),
        "conditions": {"pdb_id": args.pdb_id, "chains": chains,
                       "device": str(device), "model_config": MODEL_CFG},
    }
    model = load_model(args.thermompnn_dir, device)
    matrix, sequence = site_saturation(model, pdb_path, chains)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{prefix}_thermompnn.npz")
    np.savez_compressed(
        out_path,
        ddg=matrix,
        sequence=sequence,
        alphabet=ALPHABET,
        chains=",".join(chains),
        pdb_id=args.pdb_id,
        source_pdb=os.path.basename(pdb_path),
        provenance_json=json.dumps(provenance, allow_nan=False),
    )

    n_missing = int(np.isnan(matrix).sum())
    per_chain = np.nanmean(matrix, axis=(1, 2))
    print(f"\nSaved {out_path}  shape={matrix.shape}")
    if n_missing:
        print(
            f"  WARNING: {n_missing} of {matrix.size} entries are NaN, from "
            "positions the structure does not model. ddg.py refuses to score a "
            "candidate that mutates one of them, so this target cannot use the "
            "ddG term until the gaps are filled."
        )
    print(f"  sequence ({len(sequence)} residues): {sequence}")
    print(f"  mean ddG per chain copy: {np.round(per_chain, 3).tolist()}")
    print(
        "  positive ddG = destabilizing (ThermoMPNN negates the Megascale "
        "convention); self-mutations are 0 by construction"
    )


if __name__ == "__main__":
    main()
