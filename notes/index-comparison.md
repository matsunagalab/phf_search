# What the amyloid indices actually measure

Kept here because it is a finding about the *indices*, not usage documentation.
The README says how to run them; this says what came out when three of them
were pointed at the same sequence.

Measured on the PHF tau core (5O3L chains A,C,E,G,I, 73 residues,
`VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF`)
with this repository at commit `fd01377`, on an RTX A6000.

## Two indices agree where it matters

AGGRESCAN (a 2007 table of twenty in-vivo-derived numbers plus a sliding
window) and the `6aa`/`15aa` language-model heads (ESM2-3B embeddings plus
logistic regression, 2026) share no method, no training data and no decade.
Pointed at this sequence they independently pick out the same window:

| | AGGRESCAN | language model |
|---|---|---|
| peak position | residue 1, inside VQIVYK | residue 1, inside VQIVYK |
| strongest region | hot spot residues 1-6, `a4vahs` 0.7873, clears `HST` by 0.5026 | per-residue peak 0.9957 |
| VQIVYK region (res 1-6) | -- | mean 0.7712 |
| C-terminal (res 68-73) | below threshold | mean 0.1211 |

VQIVYK is the known aggregation-driving motif of tau R3. Where two indices this
different agree, the signal is in the sequence rather than in the index. That
is the useful case.

## The same index's other output is a threshold artifact

AGGRESCAN reports **two** hot spots on this sequence, and they are not
comparable findings:

| hot spot | residues | margin above `HST = -0.02` | `hsa` |
|---|---|---|---|
| 1 | 1-6 `VQIVYK` | **0.5026** | 4.0218 |
| 2 | 9-13 `DLSKV` | **0.0026** | 0.4058 |

A factor of ~190. Moving the threshold from -0.02 to -0.015 deletes the second
one outright (`nhs` 2 -> 1); -0.01 likewise. So "this sequence has two
aggregation hot spots" is a statement about where `HST` happens to fall, not
about the sequence.

This is why `aggrescan.score()` returns a `margin` per hot spot, and why
`nhs`/`nnhs` are recorded but excluded from `METRICS`. Over all 1387 single
mutants of this 73-residue target:

| metric | distinct values |
|---|---|
| `na4vss` | 455 |
| `thsar` | 328 |
| `a3vsa` | 322 |
| `nnhs` | **3** (1.3699 / 2.7397 / 4.1096) |

Counting hot spots is a step function whose steps are threshold noise. As a
search objective it would present a flat landscape to five of every six
single-point moves.

Reproduce with:

```python
from aggrescan import score
from utils import AMINO_ACIDS
tau = "VQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTF"
muts = [tau[:i] + a + tau[i+1:] for i in range(len(tau)) for a in AMINO_ACIDS if a != tau[i]]
print(len({round(score(m)["nnhs"], 9) for m in muts}))   # 3
print(len({round(score(m)["na4vss"], 9) for m in muts}))  # 455
```

## The literature says the same thing more strongly

Gadhe et al. (bioRxiv 2026, doi 10.64898/2026.05.08.723915) ran ProteinMPNN
inverse folding on three alpha-synuclein fibril templates and characterized
4000 designs per template with TANGO and WALTZ. As sampling temperature rises,
**TANGO falls while WALTZ rises** -- a strong anti-correlation. The same
sequences become *less* aggregation-prone by one index and *more*
amyloidogenic by the other, because TANGO scores generic hydrophobic
beta-aggregation and WALTZ scores sequence-specific amyloid motifs.

So "aggregation propensity" is not one quantity, and two tools that both claim
to measure it can move in opposite directions on the same data. Their paper
never uses a structure predictor: no AlphaFold, no pLDDT, no RMSD anywhere in
it. They assume the fold and ask only whether a sequence is energetically
compatible with it (PyRosetta `ref2015_cart` relax, FoldX per-residue ddG).

## Reference points on this target

For calibration. Native PHF tau, the sequence that does form the target
filament in reality:

| metric | value |
|---|---|
| pLDDT | 0.2349 |
| RMSD to 5O3L | 34.97 A |
| amyloid mean / max | 0.2335 / 0.9957 |
| LLPS | 0.5043 |
| AGGRESCAN `na4vss` | -13.4989 |
| AGGRESCAN `a3vsa` | -0.1230 |
| AGGRESCAN `nhs` | 2 (see above) |

Note the first two rows: **AF2 fails on the correct answer.** Any search here
is looking for a sequence AF2 *will* fold onto the PHF coordinates, which is
not the same thing as looking for a sequence that forms PHF.

## Status

The Dementia Japan 41(1) review that prompted this work was submitted on
2026-09-15, so these observations did not feed into it. Kept for whatever comes
next -- the GA search, or a later write-up.
