# Recurrent-channel scaling study: 20 October 2026

This is the compact, frozen summary of the order-10, 100 and 1000 study.
The exact source commit is `807c9534c8f9119f427555c6dc3eef2ab52aceb2`.
See [REPORT.md](REPORT.md) for results and all noise/simulation qualifications.

- 54/54 main ideal trajectories met the 1e-6 residual criterion.
- 42/72 logical-noise trajectories met the 1e-2 criterion.
- All three condition-100 stress probes failed the finite 4096-update budget.
- Nine dynamical trajectories retain the quantum data state through drift and a jump.
- Six small complete-circuit comparisons agreed to within 3.75e-15 in density norm.
- Large cases simulate the exact reduced quantum channel, with dense classical SVD preprocessing. No physical hardware or native-gate execution at orders 100/1000 is claimed.

`result.json.gz` contains every recorded trajectory and the original provenance.
`static_trajectories.csv.gz` is compressed without changing its uncompressed bytes.
The other CSVs and the PNG/SVG figures are directly readable. `MANIFEST.json`
checks this compact subset; `FULL_RUN_MANIFEST.json` is the original full-run
index and also lists input NPZ files that are omitted here to avoid large binary
matrices in Git. Those actual matrices and RHS vectors are retained in the
complete validation/source ZIP supplied with the separate report.

Verify this subset and the CSVs against the raw results from the repository root:

```python
import gzip, hashlib, json
from pathlib import Path
from qcenn_mi.recurrent_scaling_report import tables
p = Path("results/recurrent-scaling-20261020")
m = json.loads((p / "MANIFEST.json").read_text())
for name, sha in m["files"].items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == sha
raw = gzip.decompress((p / "result.json.gz").read_bytes())
original = json.loads((p / "FULL_RUN_MANIFEST.json").read_text())
assert hashlib.sha256(raw).hexdigest() == original["files"]["result.json"]
for name, expected in tables(json.loads(raw)).items():
    path = p / name
    actual = path.read_bytes() if path.exists() else gzip.decompress((p / (name + ".gz")).read_bytes())
    assert actual.decode() == expected
```

Reproduce the study in a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m qcenn_mi.recurrent_scaling_benchmark \
  --config configs/full.json --output-dir results/new-recurrent-run
```

The complete Docker validation now includes this stage. This separate local run
executed the full scaling stage and final report; it did not rerun every other
Docker experiment. The unit-test suite passed 198 tests separately.
