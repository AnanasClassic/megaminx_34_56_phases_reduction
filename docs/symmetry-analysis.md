# Symmetry analysis

Status: established for both pairs.

Candidate automorphisms are admitted per pair only after checking that they:

1. permute the 48 legal FTM generators with unchanged cost;
2. preserve both relevant subgroup target predicates;
3. preserve the coordinate/lifting contracts;
4. have a stored inverse;
5. transform a verified solution back to a verified source solution.

The 60 orientation-preserving rotations were reconstructed from the cyclic
face-neighbor orders induced by the pinned move cycles. A rotation is fixed by
the image of one oriented adjacent-face pair, giving exactly `12 * 5 = 60`
checked maps. Filtering by the nested generator sets gives:

| Pair | Required nested sets | Admissible rotations |
|---|---|---:|
| pair34 | first 7, first 6, first 5 faces | identity only |
| pair56 | first 5, first 4, first 3 faces | identity only |

Consequently symmetry orbit size is one for every raw state. The complete face
maps and generator conjugations are stored in each
`reductions/pair*/equivalences.json`.

Inversion was scanned over 100% of both raw composition sets. No inverted state
belonged to the same raw set (`0 / 536572` and `0 / 407628`), and inversion
exchanges the left-coset convention with a right-coset convention. It is
therefore not an admissible reduction.
