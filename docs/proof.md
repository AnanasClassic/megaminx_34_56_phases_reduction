# Proof boundary and external dependencies

## What is certified here

The two local computational statements are:

```text
D_(3+4) <= 21
D_(5+6) <= 25
```

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

## What `certify` checks

For one pair, `scripts/certify` fails unless it can:

1. validate the regenerated table payload checksums, histograms, generators,
   diameters, and maximal-layer counts;
2. verify every hard-state record against its complete depth table;
3. enumerate the complete Cartesian product and round-trip both phase
   coordinates in a full physical state;
4. replay every exact-reduction witness and require an exact partition into
   reduced and remaining IDs;
5. require the SQLite ID set to equal the remaining set exactly;
6. independently replay every SQLite word in both the quotient and
   `FullStateV1`, enforcing the pair-specific length limit.

The full reproduction entry points are `make verify-pair34`,
`make verify-pair56`, and `make verify-all`.
