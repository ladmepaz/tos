# Research Code Notice

This repository contains research code for reproducing the Tree of Science 3 article workflow. It is not a Python package, software library, or maintained application.

## Intended Use

The code is intended to:

- Reproduce the Entrepreneurial Marketing case study used in the paper.
- Document the data-cleaning, citation-network, SAP, roots, trunk, branches, leaves, fruits, and visualization steps.
- Provide transparent scripts and outputs that other researchers can inspect, adapt, and rerun.

## What This Repository Is Not

This repository is not intended to provide:

- A stable Python API.
- A package installable with `pip install tos3`.
- Backward-compatible function signatures.
- General-purpose bibliometric software.
- A deployed web application.
- Automated literature-review writing.

## Reproducibility Scope

The main reproducibility entry point is:

```powershell
python src\run_pipeline.py
```

The pipeline assumes the BibFusion-style input files described in the main `README.md`. The current outputs and figures are tied to the Entrepreneurial Marketing dataset and the parameters in:

```text
config/tos3_config.json
```

Researchers can adapt the scripts for another field, but should expect to inspect and adjust data-cleaning rules, parameter choices, and visualization layout.

## Stability Expectations

The code favors clarity and article reproducibility over package-level abstraction. Scripts may be reorganized, renamed, or simplified as the paper and repository evolve.

If this project later becomes a reusable tool or web application, that should be developed as a separate software layer on top of the research workflow.

