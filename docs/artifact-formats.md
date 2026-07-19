# Artifact contracts

All integer fields are little-endian. Payloads are published through a partial
file plus fsync/rename and are covered by SHA-256 manifests. Byte value 255 is
the unreachable/unset sentinel and is distinct from solved depth zero.

## Phase tables

Each `tables/phaseN/` contains `metadata.json`, `histogram.csv`, `depths.bin`,
`predecessors.bin`, and `antipodes.bin`. The predecessor byte is a deterministic
lowering move code (`face_index * 4 + power - 1`). Dense phases derive the
predecessor index by applying that move. Sparse phase 4 additionally stores
`predecessor_indices.bin`. A physical representative is reconstructed by
reversing/inverting the lowering path, so no 108-byte state is duplicated for
every coset.

## Hard states

`MDRHSV1` has a 16-byte header followed by fixed 172-byte records:

- phase index, depth, and optimal-word length;
- one 108-byte `FullStateV1` representative;
- up to 16 lowering move codes, padded with 255;
- 32-bit masks of every possible first and last optimal move;
- SHA-256 of the preceding 140 record bytes.

The companion `.metadata.json` covers the full payload. The independent Python
checker replays every record and recomputes both masks from the depth table.

## Compositions

Raw order is hard-state A major, hard-state B minor. `states.bin` stores one
108-byte physical state per raw pair; `pair_indices.bin` stores two 32-bit phase
indices. `raw_to_unique.bin` maps raw IDs to `unique_states.bin`. No injectivity
is assumed; the regenerated pair34 and pair56 maps happened to have no
duplicates.

## Reductions

`mapping.bin` has one 16-byte record per raw pair: unique ID, status, witness
length, representative ID, and witness offset. Status 0 maps to an unresolved
canonical representative; statuses 1 and 2 refer to bounded boundary/local
witnesses in `witnesses.bin`. `remaining_ids.bin` is the ordered unresolved
list. `equivalences.json` records all 60 rotations, the admissible stabilizer,
the exhaustive inversion scan, and all-optimal boundary-mask compatibility.

The current reduction maps are exhaustive: every raw ID occurs exactly once.
Status 2 contains 203 pair34 and 8,461 pair56 replayed local-rewrite witnesses;
status 0 contains the remaining 536,369 and 399,167 verified unique states.
No status-1 boundary-only witness exists because the exhaustive optimal-move
mask test found no compatible same-face merge.

Local rewrites first normalize commuting/cancellable moves, then use exact
meet-in-the-middle joins around the phase boundary. The published maps cover
windows through 4+4 moves. Both published maps use the first seven faces, the
28-move `G5` alphabet. Pair56 therefore allows exact rewrites to leave its
20-move `G7` source alphabet, but does not claim an all-48-move preprocessing
search. A hash match is only a candidate: full 108-byte states are compared
before a witness is emitted, and every emitted word is independently replayed
afterward.
