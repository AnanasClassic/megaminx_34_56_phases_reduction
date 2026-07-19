# Verifier

The primary verifier is `build/mdr-verify`, compiled from the pinned upstream
move/hash implementation plus the strict adapter in `verifier/go/main.go`. The
independent verifier is implemented in `src/mdr/state.py` and
`src/mdr/dual_verify.py`; its move cycles and state action are separately
written and do not call the Go transition code.

`scripts/verify` accepts a solution only when both paths accept the same strict
`FullStateV1` input, canonical move word, target and length bound. Unknown
moves, malformed records, invalid physical invariants, target disagreement and
over-bound words fail closed.

```bash
make build-verifier
scripts/verify --state state.bin --solution solution.txt \
  --target solved --max-length 21
```

The differential suite checks all 48 legal moves, generator order and inverse,
random words, serialization, subgroup predicates and the left-coset phase
convention. The exhaustive pre-training gate additionally uses the independent
path to replay every hard state, every composition and every emitted reduction
witness.

## Persistent bulk replay

`verify-batch` applies the same strict Go state decoder, move parser, FTM bound,
and subgroup predicate to a stream of direct certificates. One persistent
process avoids launching the verifier roughly one million times. Each input
line has three tab-separated fields:

```text
decimal-state-id<TAB>FullStateV1-hex<TAB>canonical move word
```

IDs must be strictly increasing, and every record must end in one LF byte. The
command reports the exact record count, first and last IDs, maximum observed
length, and SHA-256 of the complete canonical input transcript as JSON. The
Python bridge computes the same digest over the bytes that it sends and
compares every field with its independently enumerated SQLite pass. Dropping,
duplicating, reordering, or changing a streamed row therefore fails closed.
`scripts/reproduce_pair` enables this path automatically with
`--go-verifier build/mdr-verify`.

The Go bulk command deliberately does not parse SQLite or the neural quotient.
Python remains responsible for database metadata, ID-set equality, record
hashes, and quotient replay; Go independently checks the physical
`FullStateV1` transition and target-subgroup claim for every direct word.
