# Proof boundary and external dependencies

## What is certified here

Let `Omega` denote all 48 FTM moves and `Sigma5` the 28 moves on the seven
faces generating `G5`. The two local computational statements are the stronger
target-solving bounds

```text
R_Sigma5(G5,G7) <= 21
R_Sigma5(G7,G9) <= 25
```

The ambient `R_Omega` bounds follow immediately. All pair34 words use
`Sigma5`. Pair56 direct certificates use the 20 moves of `Sigma7`, while its
exact boundary rewriting uses `Sigma5`. Because those rewrites may leave
`G7`, the second result is not a source-alphabet `Sigma7` radius; it is a
`Sigma5` target bound. Neither result is an ordinary graph diameter. The
machine-readable manifests use these same target-bound statements.

For each statement the checker regenerates both complete phase tables, extracts
the full maximal layers, constructs their physical Cartesian product, verifies
the exact local-rewrite witnesses, and independently replays one bounded word
for every remaining state. Neural scores and checkpoints are never read by the
certificate checker.

| Pair | Raw states | Exact rewrites | Direct words | Missing | Limit |
|---|---:|---:|---:|---:|---:|
| 3+4 | 536,572 | 203 | 536,369 | 0 | 21 |
| 5+6 | 407,628 | 8,461 | 399,167 | 0 | 25 |

## What is not certified here

This package does not reconstruct the complete previously published global
bound. In particular it does not reproduce the large `116 -> 114` reduction
associated with Tomas Rokicki, nor the other externally published large phase
tables outside phases 3--6.

Consequently, `114 -> 112` is a conditional corollary: it requires the
published 114-move chain to use the same FTM metric, move convention, subgroup
chain, and phase composition semantics. Combining the two reductions with the
separate color-neutral phase-1 result gives the likewise conditional value 111.

## What full reproduction checks

For one pair, `scripts/reproduce_pair` fails unless the chained audit and
certificate checker can:

1. recompute the three subgroup orders, both individual quotient factors, and
   the combined target pointwise stabilizer from the committed action arrays,
   after checking those arrays against `FullStateV1` through the stored
   conjugacy;
2. validate the regenerated table payload checksums, histograms, generators,
   target radii (stored under the legacy metadata key `diameter`), and
   maximal-layer counts;
3. verify every hard-state record against its complete depth table;
4. enumerate the complete Cartesian product and round-trip both phase
   coordinates in a full physical state;
5. replay every exact-reduction witness and require an exact partition into
   reduced and remaining IDs;
6. require the SQLite ID set to equal the remaining set exactly;
7. replay every SQLite word in the quotient and the separately implemented
   Python `FullStateV1` transition, enforcing the pair-specific length limit;
8. in the full reproduction commands, stream the same exact record set through
   one persistent Go verifier and require its count and maximum length to agree
   with the Python pass.

The full reproduction entry points are `make verify-pair34`,
`make verify-pair56`, and `make verify-all`.

The dependency-light command `scripts/certify` does not import PyTorch. A
direct invocation uses the Python quotient/full-state checker unless
`--go-verifier PATH` is supplied. The full reproduction entry points build the
Go verifier and always supply that flag; their JSON reports record the Go
binary hash and exact number of Go-replayed certificates.
