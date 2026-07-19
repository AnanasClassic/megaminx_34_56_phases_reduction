# Related-work citation map

This file records what each bibliography entry can safely support.  It is a
claim-control note for drafting, not part of the manuscript.

## Megaminx-specific sources and provenance

| Key | Safe use in the paper | Important limitation |
|---|---|---|
| `botz2026dataset` | Primary source for Botz's published Megaminx solver dataset, the availability of table-generation code, and the provenance of the phase coordinates and move conventions used by this artifact. | TUdatalib classifies the item as **Software / Other** and only lists “Thesis” as a keyword.  Do not describe it bibliographically as a published thesis unless a separately catalogued thesis is located. |
| `botzsource` | Immutable pointer to the exact public source snapshot pinned by `environment/fetch_upstream.sh`; appropriate when discussing implementation provenance or independent reimplementation. | This is source code, not a refereed mathematical publication.  No publication year is asserted because the commit date was not available from the repository metadata checked here. |
| `botz2025bound116` | The public report of the merged-phase 116-move total. | Forum report only; not an independently reproduced theorem in this paper. |
| `botz2025correction` | The correction of the displayed depth-14 table to depth 13.  In context this confirms the already reported 116-move total; it is **not** a (116\to115) improvement. | Historical clarification only.  The surrounding exchange first questions whether the total should be 117 and then accepts 116 after the table correction. |
| `rokicki2025bound115` | The report that all 16 depth-26 two-generator antipodes have 24-move solutions after adjoining (F).  All other positions were already at two-generator depth at most 25, so this proves a last-phase upper bound of 25 and changes the external total (116\to115). | Forum report only; the present artifact does not reproduce this computation.  Do not describe this post as the (115\to114) step. |
| `rokicki2025bound114` | The subsequent report that all 7,595 depth-25 positions (1,199 representatives after symmetry and inversion) also have solutions of length at most 24 in the enlarged generator set.  This proves the next last-phase reduction (25\to24), hence the external-total step (115\to114) (two moves below the original 116). | Forum report only; use solely to explain why all 112/111 global consequences are conditional. |
| `botz2026bound114` | Botz's later identification of 114 as the improved upper bound. | Corroborates the public history but does not replace a reproducible certificate. |
| `kuznetsov2026phase1` | The separate exact phase-1 color-neutral result and its reproducibility package; safe support for the local reduction from 10 to 9 used in the conditional 111 arithmetic. | The repository describes a forthcoming manuscript and has no DOI or archival release in the checked metadata.  Cite the pinned commit and do not call it peer reviewed. |
| `kuznetsov2026artifact` | Availability/provenance citation for the certificate databases, models, scripts, and verification code accompanying the present paper. | Cite an immutable release DOI later if one is created; until then retain the pinned commit. |

The exact post URLs above were cross-checked against the already curated
bibliography in the phase-1 manuscript.  They should remain confined to the
historical paragraph about the external 114-move dependency.

The claim-critical arithmetic supported by the primary posts is therefore

\[
116 \xrightarrow[\text{all depth-26 antipodes handled}]{26\to25} 115
    \xrightarrow[\text{all depth-25 positions handled}]{25\to24} 114.
\]

The September correction precedes this chain but does not supply either
arrow: it resolves the apparent (117/116) discrepancy in the displayed
phase table and leaves the reported total at 116.

## Certified computational puzzle bounds

| Key | Safe use in the paper | Important limitation |
|---|---|---|
| `korf1997` | Introduced the Rubik's Cube application of IDA* with large pattern databases and reported the first optimal solutions of random instances. | It did not establish the cube's worst-case diameter. |
| `culberson1998` | General pattern-database method and its role as an exact abstract-distance lower bound for heuristic search. | The main experiment is the 15-puzzle, not Megaminx. |
| `kunkle2007` | A large certified computation proving that 26 face turns suffice for Rubik's Cube. | This is an upper bound, not the later exact diameter. |
| `rokicki2013` | The certified proof that the ordinary Rubik's Cube group has diameter 20 in the face-turn/half-turn metric; also a strong precedent for coset partitioning, symmetry reduction, and large computation plus verification. | Its graph is a Cayley graph of the full cube group.  Do not transfer “diameter” terminology automatically to non-normal Megaminx phase quotients. |

## Neural-guided and batched search

| Key | Safe use in the paper | Important limitation |
|---|---|---|
| `agostinelli2019` | DeepCubeA learns a cost-to-go function from reverse/generated states and combines it with batched weighted search; it solves random test instances of Rubik's Cube and other puzzles. | Empirical test-set solving is not exhaustive certification of a fixed frontier. |
| `takano2023` | EfficientCube shows that goal-rooted random scrambles can provide simple self-supervision for a neural action-ranking/value model used with beam search. | The method and beam search predate this paper.  Do not claim that neural-guided beam search itself is new here. |
| `chervov2025` | Neural diffusion-distance estimation and beam search for 3x3x3, 4x4x4, and 5x5x5 Rubik's Cubes; relevant precedent for massively parallel neural puzzle solving. | As cited, this is arXiv v1 (2025), not a refereed venue.  Its performance claims concern benchmark instances, not exhaustive proof obligations. |
| `cohenbeck2021` | Complete-anytime beam search with restarts, heavy-tail analysis, and randomized restarting in goal-oriented neural sequence decoding.  It supports discussion of progressively enlarged/restarted beam portfolios. | It does not introduce the survivor-only cascade or model-independent certificate coverage used here. |
| `greco2022` | K-Focal Search batches expensive learned-heuristic evaluations on a GPU and evaluates the idea on Rubik's Cube while retaining bounded-suboptimal guarantees. | It is a focal-search algorithm, not beam search and not a certificate cascade. |
| `futuhisturtevant2026` | Recent CPU--GPU batching of neural heuristic evaluations for depth-first heuristic search, including Rubik's Cube. | It addresses IDA*/BTS-style depth-first search, not beam cascades; cite it for the broader batching precedent only. |

Recommended novelty sentence supported by this group of citations:

> Learned puzzle heuristics, batched GPU evaluation, and restarted beam search
> are established techniques.  The distinctive role here is to use them as an
> untrusted generator of replayable bounded-word certificates for every member
> of a predetermined exhaustive boundary layer.

## Mathematical background

| Key | Safe use in the paper | Important limitation |
|---|---|---|
| `joyner2008` | Textbook background on permutation puzzles, group actions, Cayley graphs, symmetries, subgroups, and solution strategies. | Use for general mathematical background, not for this paper's numerical Megaminx bounds. |
| `seress2003` | Standard reference for base-and-strong-generating-set and Schreier--Sims style permutation-group algorithms used in subgroup/stabilizer audits. | It supports the algorithmic framework, not the correctness of this implementation or its generated numerical data. |

## Metadata sources checked

Bibliographic details were taken from the official TUdatalib record, AAAI
proceedings PDFs/pages, ACM DOI record, SIAM article page, Nature article page,
OpenReview/TMLR record, arXiv record, Springer chapter page, Hopkins Press book
page, Cambridge DOI metadata, and the pinned GitHub repositories.  The entries
avoid inferred DOIs and omit metadata that could not be established from those
sources.
