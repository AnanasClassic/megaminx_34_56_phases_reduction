# FullStateV1 and executable action convention

`FullStateV1` is the stable physical-state record used by both verifiers. It is
exactly 108 bytes; integers are unsigned single bytes and no padding is present.

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 8 | ASCII magic `MDRFSV1` followed by NUL |
| 8 | 30 | edge piece at each edge position 0..29 |
| 38 | 30 | edge orientation at each position, 0 or 1 |
| 68 | 20 | corner piece at each corner position 0..19 |
| 88 | 20 | corner orientation at each position, 0..2 |

Both piece arrays must be permutations. Both permutation parities are even,
the edge-orientation sum is zero modulo 2, and the corner-orientation sum is
zero modulo 3. Decoders reject every invalid record and never repair or
canonicalize input.

The token sequence `a b c` means start with the input state, apply `a`, then
`b`, then `c`. A canonical token is one of the twelve configured face names
followed by one power digit 1..4. Tokens are separated by exactly one ASCII
space. Only an optional single terminal LF is accepted.

The upstream phase coordinate for a state word `s` is invariant when a word
`h` in the next subgroup is prepended: `index(h s) = index(s)`. Appending `h`
does not generally preserve it. Thus the executable action convention stores
the left coset `Hs`. Randomized tests establish both directions for phases 3
through 6. Composition code must account for this explicitly.

| Phase | Coordinate | Source generators | Target |
|---:|---|---|---|
| 3 | `hash7Gen` | `U..FR` | G6 (`U..BL`) |
| 4 | `hash6Gen` | `U..BL` | G7 (`U..BR`) |
| 5 | `hash5Gen` | `U..BR` | G8 (`U..L`) |
| 6 | `hash4Gen` | `U..L` | G9 (`U R F`) |

The primary Go adapter is compiled with the pinned upstream move and hash
functions. The independent Python verifier has separately written move cycles.
`scripts/verify` accepts a result only when both implementations agree.

```bash
make build-verifier
printf 'U1 R2 F4\n' > /tmp/word.txt
scripts/make_state --moves /tmp/word.txt --out /tmp/state.bin
printf 'F1 R3 U4\n' > /tmp/inverse.txt
scripts/verify --state /tmp/state.bin --solution /tmp/inverse.txt \
  --target solved --max-length 3
```
