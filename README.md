# Tree of Science 3

Tree of Science 3 (ToS 3) is research code for building and visualizing an enhanced Tree of Science from BibFusion outputs. The project is designed to support the article workflow, not to provide a maintained Python package.

The current case study is Entrepreneurial Marketing. The code starts from BibFusion CSV files, builds a directed citation network, applies the SAP-based Tree of Science classification, enriches the graph with article metadata, and exports analytical outputs and publication-ready visualizations.

## What ToS 3 Does

ToS 3 organizes a local citation network into interpretable reading and writing roles:

- `Roots`: foundational papers cited by later work in the field.
- `Trunk`: structurally central papers that consolidate and connect the field.
- `Branches`: recent thematic conversations extending the structural core.
- `Leaves`: frontier papers that cite the field but have not yet been cited within the local network.
- `Fruits`: recent papers with strong external citation attention but limited local absorption.

The enhanced visualization uses different visual encodings for each role:

- Brown circles for structural papers: roots, trunk, and branches.
- Green or brown leaves for frontier and dormant frontier papers.
- Apples for fruits.
- Size for citation prominence, connection strength, or external citation velocity, depending on the role.
- Color intensity for temporal or citation-based signals.

## Expected Input Files

The pipeline expects BibFusion output files in:

```text
data/raw/bibfusion/
```

The current raw inputs are:

```text
All_Affiliation.csv
All_ArticleAuthor.csv
All_Articles.csv
All_Authors.csv
All_Citation.csv
All_Journal.csv
All_Scimagodb.csv
```

The two main files used by the ToS 3 pipeline are:

- `data/raw/bibfusion/All_Citation.csv`: citation links. The main columns are `SR` and `SR_ref`.
- `data/raw/bibfusion/All_Articles.csv`: article metadata. The main identifier is `SR`.

The cleaning scripts generate processed versions in:

```text
data/processed/bibfusion/
```

The processed files currently used by the graph builder are:

```text
All_Citation_tidy.csv
All_Articles_tidy.csv
```

## Environment Setup

Use Python 3.10 or newer. The current development environment was run on Windows.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Some visualization and diagnostic scripts also use `matplotlib`. If it is not already installed, run:

```powershell
python -m pip install matplotlib
```

## How To Run The Current Pipeline

Run commands from the repository root:

```powershell
cd F:\tos_3\tos
```

The main parameter file is:

```text
config/tos3_config.json
```

It records the current paper settings, including SAP counts, similarity weights, branch recency rules, fruit selection rules, color palettes, and visualization sizes.

The pipeline is organized into numbered stage scripts:

To run the full reproducibility pipeline:

```powershell
python src\run_pipeline.py
```

To preview the full pipeline without changing files:

```powershell
python src\run_pipeline.py --dry-run
```

To run only part of the pipeline:

```powershell
python src\run_pipeline.py --start-stage 4 --end-stage 8
```

The numbered stages can also be run individually:

```powershell
python src\01_tidy_data.py
python src\02_build_citation_network.py
python src\03_apply_sap.py
python src\04_roots_analysis.py
python src\05_trunk_analysis.py
python src\06_branches_analysis.py
python src\07_leaves_fruits.py
python src\08_export_visualizations.py
```

Each stage is a thin wrapper around the underlying research scripts. To inspect the commands without running them, use:

```powershell
python src\04_roots_analysis.py --dry-run
```

The stage scripts use `config/tos3_config.json` by default. To run with a different configuration file:

```powershell
python src\04_roots_analysis.py --config config\my_config.json
```

Important implementation note: in the current codebase, SAP is applied inside `src/build_citation_network.py` when the citation graph is created. Therefore, `src\03_apply_sap.py` currently validates and summarizes the ToS labels already written to `outputs/graphs/citation_network.gexf`.

## Main Parameters

The current configuration uses these core values:

- SAP selection: 20 roots, 20 trunk papers, 50 SAP leaves, and a maximum branch size of 15.
- Root similarity: text `0.45`, co-citation `0.35`, structural `0.20`, optional NTF `0.00`, with similarity threshold `0.35`.
- Trunk similarity: text `0.40`, references `0.35`, citers `0.25`, with similarity threshold `0.18`.
- Branches: three branches, five-year recency window, citation-path proximity `0.40`, text similarity `0.35`, reference overlap `0.25`.
- Branch core papers: semantic score `0.50`, recency `0.30`, trunk connection `0.20`, with core-paper threshold `0.55`.
- Leaves: internal indegree equal to zero and internal outdegree greater than or equal to one; visual leaf colors use a five-year recent frontier window.
- Fruits: five-year recent window, top 30 candidates evaluated, top three displayed.
- Fruit score: external citation velocity `0.35`, external attention gap `0.35`, network proximity `0.20`, text relevance `0.10`.
- Structural color gradient: `#C4AE97` to `#441A03`.
- Fruit color gradient: `#E9A195` to `#A32314`.

### 1. Clean BibFusion Data

These scripts create the tidy citation and article files used by the graph builder.

```powershell
python src\01_tidy_data.py
```

Outputs:

```text
data/processed/bibfusion/All_Citation_tidy.csv
data/processed/bibfusion/All_Articles_tidy.csv
```

### 2. Build The Citation Network

```powershell
python src\02_build_citation_network.py
python src\03_apply_sap.py
```

Output:

```text
outputs/graphs/citation_network.gexf
```

This graph is directed. It removes self-loops, applies graph cleaning, enriches nodes with selected article metadata, and adds the ToS classification and `sap_rank`.

### 3. Analyze Roots

The roots workflow estimates thematic similarity using text, co-citation, and structural signals, then exports root visualization metrics.

```powershell
python src\04_roots_analysis.py
```

Main outputs:

```text
outputs/root_combined/
outputs/root_visualization/
```

### 4. Analyze Trunk

```powershell
python src\05_trunk_analysis.py
```

Main outputs:

```text
outputs/trunk_combined/
outputs/trunk_visualization/
```

### 5. Analyze Branches

The branch workflow identifies recent thematic conversations, evaluates community structure, assigns branch roles, and exports branch SVG files.

```powershell
python src\06_branches_analysis.py
```

Main outputs:

```text
outputs/branch_leiden_comparison/
outputs/branch_assignment/
outputs/branch_member_roles/
outputs/branch_visualization/
```

### 6. Analyze Leaves And Fruits

Leaves are identified as papers with internal indegree equal to zero and internal outdegree greater than or equal to one. Fruits are selected from recent papers using external citation attention and local-network absorption signals.

```powershell
python src\07_leaves_fruits.py
```

Main outputs:

```text
outputs/leaf_like_nodes/
outputs/leaf_visualization/
outputs/fruits/
outputs/fruit_visualization/
```

### 7. Export Paper Figures

```powershell
python src\08_export_visualizations.py
```

Main outputs:

```text
outputs/paper_figures/
```

## How To Reproduce Figure 1

The current publication figure is stored in:

```text
outputs/Figure 1 ToS 3 new algorithm.svg
outputs/Figure 1 ToS 3 new algorithm.png
```

The final ToS visualization assets are also stored in:

```text
outputs/tos_3_visualization_final_3.svg
outputs/tos_3_visualization_final_3.jpg
outputs/paper_figures/figure_1_tos3_visualization.svg
outputs/paper_figures/figure_1_tos3_visualization.jpg
```

To refresh the paper figure copies from the current final visualization files, run:

```powershell
python src\08_export_visualizations.py
```

Important note: the final Figure 1 layout includes a designer-adjusted SVG composition. The analytical scripts generate the component data and visual assets, while the final figure file preserves the publication-ready layout.

## Interactive HTML Prototype

An interactive HTML prototype is available at:

```text
outputs/html/tos3_interactive.html
```

To regenerate it:

```powershell
python src\export_interactive_tos_html.py
```

Open the file directly in a browser:

```text
file:///F:/tos_3/tos/outputs/html/tos3_interactive.html
```

This prototype is intended as a faithful interactive version of the final figure. It is not yet a deployed web application.

## Output Organization

Important output folders:

- `outputs/graphs/`: citation network files for Gephi and downstream analysis.
- `outputs/paper_figures/`: paper-ready figure assets.
- `outputs/html/`: interactive HTML prototype.
- `outputs/root_visualization/`: root metrics and root SVG/GEXF visualization files.
- `outputs/trunk_visualization/`: trunk metrics and SVG exports.
- `outputs/branch_visualization/`: branch metrics and SVG exports.
- `outputs/leaf_visualization/`: leaf metrics and canopy SVG exports.
- `outputs/fruits/`: fruit candidate tables.
- `outputs/fruit_visualization/`: fruit SVG exports and top fruit table.
- `outputs/experiments/`: exploratory analyses that are useful for transparency but not required to reproduce the main ToS 3 pipeline or Figure 1.

The top-level `outputs/` folder is reserved for the documented reproducibility workflow and final paper assets. Experimental threshold tests, optional NTF runs, branch trend/novelty experiments, and leaf-count experiments are kept under `outputs/experiments/`.

## Current Status

This repository is article reproducibility code. The next recommended cleanup steps are:

- Add a single `src/run_pipeline.py` command that runs the main workflow in order.
- Expand `requirements.txt` to include all optional visualization dependencies.
- Add a short data dictionary for the main CSV outputs.
