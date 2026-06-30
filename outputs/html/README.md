# Interactive ToS 3 HTML

This folder contains a faithful interactive HTML version of the final Tree of Science 3 SVG.

## File

- `tos3_interactive.html`: standalone HTML file with the final SVG embedded, role filters, search, region hotspots, and metadata cards.

## Current Interaction Model

- The visual layout is preserved from `outputs/ToS 3 visualization.svg`.
- The tree has region-level hotspots for roots, trunk, branches, leaves, and fruits.
- The side panel exposes paper-level metadata from the analytical CSV outputs.
- Branch records include the branch-member role, so core, peripheral, background/methodological, and missing-metadata papers can be distinguished.

## Regeneration

Run from the repository root:

```bash
python src/export_interactive_tos_html.py
```

## Next Improvement

For true node-by-node interaction directly on the tree, each SVG element needs a stable paper ID or a coordinate table that maps paper IDs to visual positions.
