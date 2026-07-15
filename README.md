# Megaminx phase 3+4 and 5+6 reductions

This repository is the computational companion for a forthcoming paper on two
exhaustive reductions in the face-turn metric (FTM):

- every maximal-layer composition of phases 3 and 4 has a verified solution
  of length at most 21;
- every maximal-layer composition of phases 5 and 6 has a verified solution
  of length at most 25.

The complete coverage is:

| Pair | Raw compositions | Exact rewrites | Direct certificates | Maximum length |
|---|---:|---:|---:|---:|
| 3+4 | 536,572 | 203 | 536,369 | 21 |
| 5+6 | 407,628 | 8,461 | 399,167 | 25 |

The certificates, not the neural networks, are the proof objects. Every stored
word is independently replayed in both the combined quotient and a full
Megaminx simulator. Missing, malformed, or over-length records make the
checker fail closed.

## Claims and dependencies

The local claims are `D_(3+4) <= 21` and `D_(5+6) <= 25`. Relative to the
published 114-move bound, they give the conditional improvement `114 -> 112`.
Together with the separate color-neutral phase-1 result, they give the
conditional value 111.

This repository does **not** reproduce the large Tomas Rokicki reduction from
116 to 114. The value 114 and the compatibility of the surrounding published
phase chain are explicit external dependencies. See [docs/proof.md](docs/proof.md).

Paper: forthcoming.

## Repository map

- [phase_3_4](phase_3_4/README.md): certificate, models, search provenance,
  plot, and reproduction guide for phases 3+4.
- [phase_5_6](phase_5_6/README.md): the corresponding package for phases 5+6.
- `src/mdr`, `builder`, and `verifier`: shared proof implementation and the
  independent full-state verifier.
- `environment`: pinned acquisition and build scripts for the Alexander Botz
  source snapshot.

## Fast package audit

Python 3.10+ and XZ support from the Python standard library are sufficient
for the archive audit. The full unit suite additionally uses the packages in
`requirements-training.txt`:

```bash
make verify-package
python3 -m pip install -r requirements-training.txt
make test
```

This verifies both compressed and uncompressed certificate hashes, the SQLite
schemas and coverage distributions, and all four checkpoint hashes. It does
not replay the solutions because the regenerated phase tables and compositions
are intentionally not stored in Git.

## Full reproduction

Requirements: Docker, Git, 7-Zip, Python 3.10+, roughly 2 GiB of working disk,
and up to ten CPU workers.

```bash
make verify-pair34
make verify-pair56
# or both:
make verify-all
```

Each command downloads and verifies the pinned upstream archive, builds the Go
table generator and both verifier paths, regenerates the phase tables, maximal
layers, physical Cartesian compositions, and exact reductions, unpacks the
distributed SQLite certificate, and independently replays 100% of solutions.

## License and provenance

The code and computational data are distributed under CC BY-NC 4.0, matching
the license of the Alexander Botz software dataset from which the phase
coordinates and move conventions were independently reimplemented. See
[LICENSE-CODE.txt](LICENSE-CODE.txt) and [NOTICE.md](NOTICE.md).
