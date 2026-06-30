# Paper Figures

This folder contains figure assets prepared for the ToS 3 manuscript.

## Files

- `figure_1_tos3_visualization.svg` / `.jpg`: final enhanced Tree of Science visualization.
- `figure_2_visual_encoding_legend.svg`: visual grammar for roots, trunk, branches, leaves, dormant leaves, and fruits.
- `figure_3_tos3_workflow.svg`: compact methodological workflow from bibliographic data to ToS 3 visualization.
- `figure_4_leaf_year_histogram.svg` / `.png`: year distribution of the 121 leaf-like frontier papers.

## Suggested Captions

**Figure 1. Enhanced Tree of Science 3 visualization for Entrepreneurial Marketing.** Roots, trunk papers, branches, leaves, dormant leaves, and fruits are represented using role-specific visual forms, sizes, and color gradients.

**Figure 2. Visual encoding used in Tree of Science 3.** The figure summarizes how analytical roles are mapped to shapes, colors, and size meanings.

**Figure 3. Overview of the Tree of Science 3 workflow.** Bibliographic records are harmonized, converted into a directed citation network, classified structurally, enriched with thematic and frontier signals, and rendered as a role-specific visualization.

**Figure 4. Publication-year distribution of leaf-like frontier papers.** The histogram shows the temporal composition of papers with internal indegree equal to zero and internal outdegree greater than or equal to one.

## Regeneration

Run the export script from the repository root:

```bash
python src/export_paper_figures.py
```
