# Interactive ToS 3 HTML

This folder contains a faithful interactive HTML version of the final Tree of Science 3 SVG.

## File

- `tos3_interactive.html`: standalone HTML file with the final SVG embedded, paper-level selection, role filters, search, and metadata cards.

## Current Interaction Model

- The visual layout is preserved from `outputs/ToS 3 visualization.svg`.
- Every titled SVG paper symbol receives a stable `data-paper-id` during export.
- Every visible root, trunk paper, core branch paper, leaf, and fruit can be selected directly in the tree; hovering shows a short title and year tooltip.
- The selected symbol is highlighted and its available bibliographic and analytical metadata is displayed in the side panel.
- Invisible padded hit areas make small symbols easier to select without changing their appearance.
- Role chips and summary counts remain available for category-level filtering.
- The side panel exposes paper-level metadata from the analytical CSV outputs.
- Branch records include the branch-member role, so core, peripheral, background/methodological, and missing-metadata papers can be distinguished.
- Non-core branch records remain searchable in the panel but have no visual target because the final figure intentionally displays core branch papers only.

## Regeneration

Run from the repository root:

```bash
python src/export_interactive_tos_html.py
```

The exporter resolves duplicate titles in record order and uses SVG symbol type to prevent a leaf-shaped paper from being linked to a same-title circle record. Role-specific interaction keys also preserve papers that appear in both the broad leaf universe and a core branch.
