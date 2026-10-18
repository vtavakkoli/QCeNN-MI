# IEEE manuscript and EuroHPC evidence report

`manuscript.tex` is a full IEEEtran journal-style manuscript titled **Quantum
Matrix Inversion via Coherent Chebyshev-LCU Circuits: Theory and Simulation to
Dimension 1000**. Its primary target is a coherent quantum operation that
heralds a solution state proportional to `M^{-1}|b>`. The title, abstract,
introduction, input-output specification, circuit construction, and conclusion
are organized around this quantum computational objective.

The paper certifies the specific Chebyshev-LCU circuit, reports complete
simulated operator audits and noisy failures, and sets out a physical quantum
validation protocol. Classical dense spectral encoding remains an explicit
assumption; the quantum core is separate from the project's earlier hybrid
tracking and correction methods. `eurohpc_report.tex` is an accompanying
technical-feasibility report. Neither document claims physical execution or
end-to-end quantum speedup.

The builder reads a complete validation run and derives every numerical table,
figure and headline metric from its JSON/JUnit outputs:

```bash
python scripts/build_publication.py --results artifacts --output-dir artifacts/publication
```

Install TeX Live (`texlive-publishers`, `texlive-latex-recommended`, recommended
fonts and `latexmk`) and the Python `publication` extra, or use the Docker image.
The source package includes the official IEEEtran class actually used and the
generated tables/figures for independent LaTeX compilation.

For an applicant-specific report, supply `--project-metadata PATH` with an
untracked JSON file containing `application_id`, `emulator_node_hours`,
`storage_gb`, `duration_days`, and `allocation_source`. Do not publish account,
credential or personal allocation details to the public repository. A run with
no such file prints that allocation information was not supplied.

The author line follows the project's previously identified collaborators:
Vahid Tavakkoli, Kabeh Mohsenzadegan and Kyandoghere Kyamakya. Institutional
affiliations, venue-specific author metadata and copyright fields are not
invented. The manuscript includes an AI-assistance disclosure under the
[IEEE policy](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/).

The documents are submission drafts, not accepted publications or an official
EuroHPC form. Select the final IEEE venue and review author metadata before
submission. No submission is performed by the workflow.
