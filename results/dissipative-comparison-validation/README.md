# Pinned dissipative comparison evidence

This compact Git record is a subset of complete run
`qcenn-2beba723e4d54f86a69a19a1183b6da0`, using exact source
`c2c18d71ac24b4082048826c9b0080212efcfa6d`. All 23 full-profile software
stages and 213 unit tests passed locally. Synthetic noisy quality remains false.
No physical QPU execution is claimed.

`result.json.gz` decompresses to the unchanged raw comparison JSON. CSVs derive
from that JSON. `full-study-MANIFEST.json` describes the **complete study**, not
just this compact subset. All 39 actual dense matrix inputs and circuit evidence
are in the complete downloadable validation/manuscript supplement; they are not
repeated in Git. The full bundle hash is recorded in `SUMMARY.json`.

Reproduce from the exact source using `python docker/reproduce.py --profile full`.
See `docs/dissipative-comparison.md` for matched criteria, rate conventions,
finite-budget/stationary distinctions, and the logical-versus-gate-noise scope.
The independently checked complete source and evidence are included with the
revised article, which remains outside this repository.
