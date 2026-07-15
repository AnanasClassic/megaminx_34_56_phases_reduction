# Exhaustive-proof threat model

The checker and pipeline must explicitly defend against:

- using published counts to manufacture or truncate generated layers;
- confusing an allocated hash slot with a reachable coset;
- treating two cosets as a freely composable physical-state product;
- applying moves in the opposite composition order;
- accepting unknown tokens because the baseline parser ignores them;
- measuring powers 2 or 3 as multiple moves instead of one FTM move;
- canonicalizing with rotations that do not preserve both phase targets;
- using inversion without transforming the target and solution correctly;
- data leakage between symmetry orbits;
- missing raw pairs hidden by duplicate elimination;
- partial files appearing complete after interruption;
- stale artifacts from a different commit or configuration;
- integer overflow in ranks, counts, offsets, or Cartesian products;
- nondeterministic representative selection;
- accepting a model score or claimed depth without replay;
- verifying only a sample of solutions;
- sharing buggy transition code between the primary and independent verifier.

Intentional-corruption tests must cover truncated payloads, changed bytes,
duplicate/missing IDs, wrong state keys, wrong transforms, illegal moves, and
solutions one move above the bound.
