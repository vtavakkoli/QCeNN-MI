# IEEE Access manuscript sources (26 October 2026)

- [QCeNN_MI.tex](QCeNN_MI.tex) — revised **19-page** main IEEE Access manuscript, compiled and visually checked from the author's recovered project.
- [QCeNN_MI_Supplement.tex](QCeNN_MI_Supplement.tex) — standalone **5-page** supplementary classical-matrix-flow, snapshot LCU, and historical-topology material.
- [sections/matched-temporal-chebyshev.tex](sections/matched-temporal-chebyshev.tex) — separately reusable explanation for the paired temporal experiment.

The original editable IEEE Access project and all PDF/PNG figures were recovered from the author's own prior LaTeX package, which corresponded to the uploaded 21-page review PDF. The full bundled manuscript and figure sources, minus third-party font files, are distributed separately as `QCeNN_MI_IEEE_Access_Publication_Revision_2026.zip`. **The figure/photo binaries are not all tracked here:** do not infer that this GitHub directory alone is sufficient for PDF reproduction. You need the companion archive and a properly installed LaTeX toolchain with `ieeeaccess.cls`, **the licensed IEEE Access Formata/Times/Giovanni Type-1 font files**, TikZ, and usual IEEE/AMS packages. These font files are intentionally omitted from the downloadable source ZIP. A clean compilation from the ZIP alone will therefore fail until you supply your own template fonts (for example, the originals from your own source bundle).

From the companion source bundle (after extracting into one working directory):

```bash
pdflatex -interaction=nonstopmode -halt-on-error QCeNN_MI.tex
pdflatex -interaction=nonstopmode -halt-on-error QCeNN_MI.tex
pdflatex -interaction=nonstopmode -halt-on-error QCeNN_MI_Supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error QCeNN_MI_Supplement.tex
```

The source package also includes a SHA-256 manifest, PDFs, and notes on submission limitations. Output counts from the checked build: main 19 pages, supplement 5 pages, zero unresolved-reference or citation warnings. The separate GitHub workflow `temporal-publication.yml` validates the new paired experiment and numerical QSP phase/noise checks, **not** an IEEE editorial acceptance decision.

Before submission, each author must approve the exact revision and confirm references/DOIs, acknowledgments, any disclosure of assistance from generative AI, conflict declarations, journal length and PDF/LaTeX parity. The ideal/synthetic simulator evidence is not evidence of a calibrated hardware speedup.
