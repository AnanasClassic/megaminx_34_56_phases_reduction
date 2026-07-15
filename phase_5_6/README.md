# Phases 5+6: exhaustive certificate for `D_(5+6) <= 25`

This directory contains the distributed proof certificate, the two Q-MLP
checkpoints that produced it, a portable reference replay report, and the search
statistics for the combined phase-5/phase-6 claim.

## Coordinate and exhaustive set

The metric is FTM: every non-identity power of a face turn costs one move.
The subgroup chain and left-coset convention are fixed in
[`docs/phase-definitions.md`](../docs/phase-definitions.md).

| Phase | Transition | Allowed source faces | States | Diameter | Maximal states |
|---:|---|---|---:|---:|---:|
| 5 | `G7 -> G8` | `U R F L BR` | 64,157,184 | 13 | 3,484 |
| 6 | `G8 -> G9` | `U R F L` | 25,945,920 | 13 | 117 |

Their maximal-layer Cartesian product has `3,484 * 117 = 407,628` physical
states. Exact boundary-window rewriting over all 48 FTM actions certifies 8,461
states without a model. The SQLite database contains bounded words for all
399,167 remaining states. The maximum independently replayed word length is 25.

## Coverage by discovery method

![Log-scale certificate coverage by exact reduction and beam width](figures/beam-coverage.svg)

| First successful method | States |
|---|---:|
| Exact reduction, no model | 8,461 |
| Beam 64 | 235,840 |
| Beam 128 | 139,544 |
| Beam 256 | 19,689 |
| Beam 1024 | 4,066 |
| Beam 4096 | 28 |

The beam width records search provenance only. It is not trusted by the proof.

## Certificate format

`certificates/beam-cascade.sqlite3.xz` expands to a SQLite database with two
tables. `metadata(key TEXT PRIMARY KEY, value TEXT)` binds the database to the
pair, FTM metric, combined problem hash, physical composition hash, remaining
ID hash, target subgroup `G9`, and limit 25.

The `certificates` table has one row per remaining representative:

| Column | Meaning |
|---|---|
| `state_id` | ID in `reductions/pair56/remaining_ids.bin` |
| `state_sha256` | hash of the 108-byte `FullStateV1` state |
| `solution` | byte-coded sequence of the 20 combined-problem FTM actions |
| `solution_length` | exact number of FTM moves, constrained to 0--25 |
| `beam_width` | first beam stage that found this word |
| `checkpoint_sha256`, `checkpoint_epoch` | untrusted search provenance |
| `verification_sha256` | hash binding the state, word, and proof problem |
| `created_utc` | provenance timestamp |

The checker requires the exact ID set, recomputes state hashes, and replays
every word in both the combined quotient and the independent full-state
simulator. It does not load a neural checkpoint.

## Fast integrity audit

From the repository root:

```bash
make verify-package
./phase_5_6/scripts/prepare
```

The first command streams the XZ payload, validates the uncompressed database,
queries its complete beam/checkpoint distribution, and hashes both models. The
second command installs the checked database at the ignored runtime path
`certificates/pair56/beam-cascade.sqlite3`.

## Full deterministic reproduction and replay

```bash
./phase_5_6/scripts/verify
```

This builds phases 5 and 6 from the pinned upstream source, extracts all 3,484
and 117 maximal records, composes all 407,628 physical states, regenerates the
8,461 exact rewrite witnesses, and replays every direct certificate. Typical
persistent generated data is below 0.5 GiB; at most ten workers are used.

Successful output ends with:

```text
pair56 FULL REPRODUCTION COMPLETE
```

## Reproducing model-guided discovery

The 802,260-parameter sparse-Q MLP uses the combined `G7/G9` quotient, 20 FTM
actions, and fixed `K_max=26`. Training writes only below ignored `artifacts/`:

```bash
./phase_5_6/scripts/train \
  --device cuda:0 --amp bf16 \
  --epochs 8192 --steps-per-epoch 256 --batch-size 1024 \
  --K-min 2 --val-size 16384 --val-batch-size 2048 \
  --val-every 32 --save-every 100 --run-id reproduction
```

`models/pair56-qmlp-epoch864.pt` produced the 235,840 beam-64 certificates;
`models/pair56-qmlp-epoch3744.pt` produced the other 163,327 certificates.
The distributed files have tensor-identical weights and optimizer state, but
their four producing-machine pathname fields were replaced by repository-relative
paths. The manifest retains both original and distributed SHA-256 values; the
SQLite provenance fields refer to the distributed files.
Run `scripts/run_publication_cascade pair56` after regenerating the deterministic
phase artifacts to repeat the staged search. Any newly produced database must
still pass the independent certificate checker.

Paper: forthcoming.
