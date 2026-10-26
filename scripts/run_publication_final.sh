#!/usr/bin/env bash
# Pinned full scientific reproduction, temporal comparison, and publication preflight.
# A software PASS never certifies calibrated QPU speedup or IEEE acceptance.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PROFILE=full
OUTPUT=""
PAPER_DIR=""
COMPILE_PAPER=0
REQUIRE_PAPER=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_publication_final.sh
  [--profile full|smoke] [--output-dir PATH]
  [--paper-dir EXTRACTED_QCeNN_MI_Revision_DIR] [--compile-paper] [--require-paper]

Default: full Docker research campaign, 96-case paired temporal benchmark,
real-Julia QSP phase/noise audit, previous-vs-new evidence reconciliation,
exact manuscript table checks, SHA-256 manifest, and compact final archive.

Paper inputs are optional unless --require-paper is set. Rebuilding the PDF
needs YOUR OWN legally licensed IEEE Access font files; none are redistributed.
EOF
}
die() { echo "ERROR: $*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --profile) (( $# >= 2 )) || die "Missing profile"; PROFILE="$2"; shift 2 ;;
    --output-dir) (( $# >= 2 )) || die "Missing output dir"; OUTPUT="$2"; shift 2 ;;
    --paper-dir) (( $# >= 2 )) || die "Missing paper dir"; PAPER_DIR="$2"; shift 2 ;;
    --compile-paper) COMPILE_PAPER=1; shift ;;
    --require-paper) REQUIRE_PAPER=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done
[[ "$PROFILE" == "smoke" || "$PROFILE" == "full" ]] || die "Unsupported profile"
command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"
command -v docker >/dev/null || die "Docker Compose v2 is required"
docker compose version >/dev/null || die "docker compose v2 unavailable"
[[ -n "$PAPER_DIR" || "$COMPILE_PAPER" -eq 0 ]] || die "--compile-paper needs --paper-dir"
[[ -n "$PAPER_DIR" || "$REQUIRE_PAPER" -eq 0 ]] || die "--require-paper needs --paper-dir"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] ||
  die "Commit repository changes before a pinned publication run"
REVISION="$(git rev-parse HEAD)"
SHORT_REVISION="$(git rev-parse --short=12 HEAD)"
if [[ -z "$OUTPUT" ]]; then OUTPUT="$ROOT/artifacts/publication-final-$SHORT_REVISION"; fi
OUTPUT="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT")"
[[ ! -e "$OUTPUT" ]] || die "Output exists: $OUTPUT; choose a fresh directory"
mkdir -p "$OUTPUT/suite"
if [[ -n "$PAPER_DIR" ]]; then
  PAPER_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$PAPER_DIR")"
  [[ -d "$PAPER_DIR" ]] || die "Missing paper directory"
fi
trap 'echo "FAILED at line $LINENO; partial results: $OUTPUT" >&2' ERR
printf 'Git: %s\nProfile: %s\nOutput: %s\n' "$REVISION" "$PROFILE" "$OUTPUT" \
  | tee "$OUTPUT/RUN.txt"

echo "[1/7] Run the clean full/smoke research campaign"
QCENN_ARTIFACTS_DIR="$OUTPUT/suite" python3 docker/reproduce.py --profile "$PROFILE" \
  2>&1 | tee "$OUTPUT/suite-run.log"

echo "[2/7] Independently verify the validation archive"
QCENN_ARTIFACTS_DIR="$OUTPUT/suite" docker compose run --rm validation \
  python -m qcenn_mi.evidence_report --results /artifacts --verify-only
python3 - "$OUTPUT/suite/validation.json" "$PROFILE" "$REVISION" <<'PY'
import json, sys
path, profile, revision = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    state = json.load(stream)
assert state["passed"] and state["evidence_integrity_passed"]
assert state["profile"] == profile
assert state["provenance"]["source_commit"] == revision
print("Pinned source and validation integrity verified.")
PY

echo "[3/7] Run the complete 96-case matched temporal comparator"
QCENN_ARTIFACTS_DIR="$OUTPUT" docker compose run --rm validation \
  python -m qcenn_mi.temporal_chebyshev_publication \
  --output-dir /artifacts/temporal-full

echo "[4/7] Audit actual small QSP circuits and logical-noise bias"
QCENN_ARTIFACTS_DIR="$OUTPUT" docker compose run --rm validation \
  python -m qcenn_mi.phase_bias_audit --conditions 2 10 \
  --output-dir /artifacts/phase-bias

echo "[5/7] Compare with earlier results and verify all manuscript table entries"
PYTHONPATH="$ROOT/src" python3 -m qcenn_mi.publication_comparison \
  --previous "$ROOT/results/polynomial-access-validation/polynomial-summary.json" \
  --temporal "$OUTPUT/temporal-full/result.json" \
  --phase "$OUTPUT/phase-bias/result.json" \
  --manuscript "$ROOT/manuscript/QCeNN_MI.tex" \
  --output-dir "$OUTPUT/comparison"

echo "[6/7] Check editable manuscript and PDFs (if supplied)"
if [[ -n "$PAPER_DIR" ]]; then
  for file in QCeNN_MI.tex QCeNN_MI.pdf QCeNN_MI_Supplement.tex QCeNN_MI_Supplement.pdf; do
    [[ -f "$PAPER_DIR/$file" ]] || die "Required paper file absent: $file"
  done
  cmp -s "$ROOT/manuscript/QCeNN_MI.tex" "$PAPER_DIR/QCeNN_MI.tex" ||
    die "Manuscript source differs from GitHub revision; reconcile it first"
  if (( COMPILE_PAPER )); then
    command -v pdflatex >/dev/null || die "pdflatex required for --compile-paper"
    mkdir -p "$OUTPUT/paper-build"
    cp -a "$PAPER_DIR"/. "$OUTPUT/paper-build/"
    (
      cd "$OUTPUT/paper-build"
      for doc in QCeNN_MI QCeNN_MI_Supplement; do
        pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null
        pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null
      done
    ) || die "PDF build failed; licensed IEEE template fonts may be missing"
    PAPER_DIR="$OUTPUT/paper-build"
  fi
  command -v pdfinfo >/dev/null || die "pdfinfo required for PDF validation"
  pages="$(pdfinfo "$PAPER_DIR/QCeNN_MI.pdf" | awk '/^Pages:/ {print $2}')"
  [[ -n "$pages" ]] || die "Unreadable PDF page count"
  (( pages <= 20 )) || die "Main manuscript exceeds 20-page target: $pages"
  printf 'Main manuscript pages: %s\n' "$pages" | tee -a "$OUTPUT/RUN.txt"
  if (( REQUIRE_PAPER )); then
    PYTHONPATH="$ROOT/src" python3 -m qcenn_mi.publication_preflight \
      --repository "$ROOT" --evidence "$OUTPUT/temporal-full" \
      --manuscript-source "$PAPER_DIR/QCeNN_MI.tex" \
      --manuscript-pdf "$PAPER_DIR/QCeNN_MI.pdf" \
      --output "$OUTPUT/submission-preflight.json" --strict
  else
    PYTHONPATH="$ROOT/src" python3 -m qcenn_mi.publication_preflight \
      --repository "$ROOT" --evidence "$OUTPUT/temporal-full" \
      --manuscript-source "$PAPER_DIR/QCeNN_MI.tex" \
      --manuscript-pdf "$PAPER_DIR/QCeNN_MI.pdf" \
      --output "$OUTPUT/submission-preflight.json"
  fi
else
  PYTHONPATH="$ROOT/src" python3 -m qcenn_mi.publication_preflight \
    --repository "$ROOT" --evidence "$OUTPUT/temporal-full" \
    --output "$OUTPUT/submission-preflight.json"
fi

echo "[7/7] Produce compact cryptographically indexed scientific evidence archive"
python3 - "$OUTPUT" "$REVISION" <<'PY'
import hashlib, json, pathlib, sys, tarfile
root, revision = pathlib.Path(sys.argv[1]), sys.argv[2]
names = [
    "RUN.txt", "suite/validation.json", "suite/final-report.json",
    "suite/final-report.md", "suite/source-files.json",
    "temporal-full/result.json", "temporal-full/REPORT.md",
    "temporal-full/MANIFEST.json", "phase-bias/result.json",
    "phase-bias/SHA256SUMS.json", "comparison/comparison.json",
    "comparison/COMPARISON.md", "comparison/MANIFEST.json",
    "submission-preflight.json",
]
missing = [p for p in names if not (root / p).is_file()]
if missing:
    raise SystemExit(f"Incomplete final evidence: {missing}")
files = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in names}
manifest = {
    "source_commit": revision, "files_sha256": files,
    "software_evidence_integrity_verified": True,
    "hardware_speedup_established": False,
    "journal_submission_approved": False,
}
(root / "FINAL_SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
with tarfile.open(root / "FINAL_EVIDENCE.tar.gz", "w:gz") as tar:
    for p in names + ["FINAL_SHA256SUMS.json"]:
        tar.add(root / p, arcname=p)
print(f"Archive: {root / 'FINAL_EVIDENCE.tar.gz'}")
PY
printf 'RESULTS:\n  %s/comparison/COMPARISON.md\n  %s/submission-preflight.json\n  %s/FINAL_EVIDENCE.tar.gz\n' \
  "$OUTPUT" "$OUTPUT" "$OUTPUT"
printf 'Scientific claims remain conditional; author and journal checks are not automatically approved.\n'
