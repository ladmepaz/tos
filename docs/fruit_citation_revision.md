# Fruit citation revision

This revision aligns the reproducible fruit tables with the revised manuscript.
It is a correction of missing metadata, not a new validation study. Ten missing
`cited_by` values were previously converted to zero during scoring.

## Source and scope

The counts were recovered by unique DOI match (case-insensitive) from the
[Scopus export, EM 7MAR26 243.csv](https://drive.google.com/file/d/13gnqMhYk2iXAbjdELTns_df9wDmzgpMO/view).
They refer to 7 March 2026, not to a new citation query. Each correction's SR,
DOI, count, source, export date and source link are deposited in
`data/corrections/fruit_citation_recovery_2026.json`.

The raw BibFusion exports are unchanged. Only the ten missing cells in
`data/processed/bibfusion/All_Articles_tidy.csv` are filled. Article-level source
flags are retained; the JSON documents the citation-specific source. No counts
are summed across databases. The corrected population retains 73 counts from
WoS Z9 (Total Times Cited) and 47 from Scopus Cited by, including the ten recovered
values. These database-wide counts can include citations from the local network;
"external" does not mean that local citations have been subtracted.

The network rebuilt for this check has 3,183 nodes and 12,481 edges. All 120
candidates and their graph/text features are retained. There are 91 candidates
matched to the graph and 11 classified as underabsorbed. An unmatched record's
stored indegree of zero is a default, not an observed degree, and does not make
that record eligible for selection as an underabsorbed fruit.

## Normalization and selection

The reference year is 2026 and the inclusive publication window is 2021-2026.
Paper age is `max(2026 - publication_year + 1, 1)`; external velocity is
`cited_by / age`. Within all 120 eligible candidates, `N` is min-max scaling;
a constant-valued indicator is mapped to 1, matching the implementation.

- Velocity score: `V = N(log(1 + external_velocity))`.
- Citation scores: `E = N(log(1 + cited_by))`, `I = N(log(1 + internal_indegree))`.
- Attention gap: `G = N(max(E - I, 0))`.
- Final score: `0.35 V + 0.35 G + 0.20 network_proximity + 0.10 text_relevance`.

Graph matching, positive external citations, internal indegree <= 1 and gap
score >= 0.40 are required for `external_fruit_underabsorbed`. The highest three
scores within this category are selected, not the highest three across all
candidate categories. The normalized velocity component is now exported as
`external_velocity_score` so the weighted sum is directly inspectable.

## Corrected manuscript results

| Paper | DOI | External citations | Local indegree | Fruit score |
| --- | --- | ---: | ---: | ---: |
| Sulaiman et al. (2024) | 10.1007/s43621-024-00710-8 | 26 | 0 | 0.667759 |
| Kakeesh (2025) | 10.1057/s41270-024-00310-5 | 8 | 0 | 0.520196 |
| Hanaysha (2022) | 10.3390/su141811444 | 11 | 0 | 0.514742 |

These three counts originate from WoS Z9 in the
[7 March 2026 WoS export](https://drive.google.com/file/d/1tJjaJAZEg134p4PozVAdwh8-il4nOgg0/view).
Their citation counts and selection order do not change. Their scores change
because normalization is recomputed over the corrected 120-candidate population.

Canonical outputs are `outputs/fruits/fruit_candidates.csv` (120 rows),
`fruit_candidates_top.csv` (30), `fruit_candidates_underabsorbed.csv` (11), and
`outputs/fruit_visualization/fruits_top3.csv` (3). The last file contains the
full-precision scores and refreshed visual-metric columns.

## Reproduce without redrawing figures

From the repository root, with its research dependencies installed:

```powershell
python src/tidy_articles.py --citation-corrections data/corrections/fruit_citation_recovery_2026.json
python src/fruit_candidates.py --current-year 2026
python src/export_fruit_visualization_svg.py --tables-only
python outputs/experiments/fruit_citation_recovery/recalculate.py
python -m unittest discover -s tests -v
```

The configured full pipeline also passes the correction file to the tidy stage
and fixes the fruit reference year to 2026. For another corpus or a newer citation
snapshot, remove/replace this case-specific correction argument and choose the
appropriate reference year. The correction procedure checks SR and DOI and
rejects conflicting reported counts, including reported zeros. Reapplication
to already corrected input is idempotent.

The archived `outputs/experiments/fruit_citation_recovery/baseline_candidates.csv`
preserves the pre-correction population. Its replay generates before/after
comparisons without modifying canonical outputs. Independent regression tests
rebuild the graph and recompute all candidates from the corrected metadata.

SVGs, paper figures and published interactive HTML are intentionally not
regenerated in this revision. Their embedded score labels or tooltip values,
where present, remain historical. Re-exporting them later will use the corrected
tables and can slightly change the middle fruit's red shade. This commit updates
reproducible data and scores, not the approved visual layout.
