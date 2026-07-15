# Metric and moves

The project uses the face-turn metric (FTM) from Definition 1 of the pinned
thesis.

The twelve faces, in the subgroup-chain order, are:

```text
U R F L BR BL FR FL DR DL B D
```

For each face `x`, the legal non-identity turns are `x`, `x2`, `x2'`, and
`x'`, corresponding to powers 1, 2, 3, and 4 of an order-five generator. Each
token costs exactly one move. Whole-puzzle rotations and slice turns are not
legal moves and are not free.

Words are replayed left-to-right. The strict canonical on-disk spelling will
use a face plus an integer power (`U1`, `U2`, `U3`, `U4`) to avoid apostrophe
ambiguity. Human-readable spellings may be imported only through a strict,
tested converter.

The production verifier must reject, rather than ignore:

- unknown face names;
- powers outside 1 through 4;
- empty interior tokens;
- implicit zero/identity moves;
- rotations, wide moves, or slice moves;
- trailing garbage.

Move count is the number of valid tokens after parsing, never the number of
characters or quarter turns.
