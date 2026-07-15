# Attribution and provenance

The phase coordinates, face cycles, cubie-orientation conventions, and ranking
formulae are independent Go/Python reimplementations and adapters for:

- Alexander Botz, *A Megaminx Solver*, TUdatalib, 2026.
  https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5015
- Public source history:
  https://git.rwth-aachen.de/alexander.botz/megaminx-solver-v2.0

The pinned TUdatalib archive is acquired by `environment/fetch_upstream.sh`
and is not committed to this repository. Its software item is licensed
CC BY-NC 4.0; this repository uses the same license for code and generated
computational data unless a file states otherwise.

The publication tree was curated from the clean proof-code snapshot
`2a579b6` of the development repository. Experimental phase-1 and unfinished
Tomas Rokicki audit files from later laboratory work are intentionally absent.

The published 114-move bound is attributed to the work of Alexander Botz,
Ben Whitmore, and Tomas Rokicki. This repository independently certifies only
the two phase-pair reductions stated in its README. No endorsement by those
authors is implied.
