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
