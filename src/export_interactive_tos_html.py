from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SVG = Path("outputs/ToS 3 visualization.svg")
DEFAULT_OUTPUT = Path("outputs/html/tos3_interactive.html")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_ROOTS = Path("outputs/root_visualization/root_visualization_metrics.csv")
DEFAULT_TRUNK = Path("outputs/trunk_visualization/trunk_visualization_metrics.csv")
DEFAULT_BRANCHES = Path("outputs/branch_member_roles/branch_member_roles.csv")
DEFAULT_LEAVES = Path("outputs/leaf_like_nodes/indegree_0_outdegree_ge_1_nodes.csv")
DEFAULT_FRUITS = Path("outputs/fruit_visualization/fruits_top3.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a faithful interactive HTML wrapper for the final ToS 3 SVG."
    )
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--roots", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--trunk", type=Path, default=DEFAULT_TRUNK)
    parser.add_argument("--branches", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--leaves", type=Path, default=DEFAULT_LEAVES)
    parser.add_argument("--fruits", type=Path, default=DEFAULT_FRUITS)
    return parser.parse_args()


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_year(value: Any) -> int | None:
    number = clean_number(value)
    if number is None:
        return None
    return int(number)


def metadata_lookup(articles: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in articles.itertuples(index=False):
        sr = clean_value(getattr(row, "SR", ""))
        if not sr:
            continue
        lookup[sr] = {
            "doi": clean_value(getattr(row, "doi", "")) or clean_value(getattr(row, "__doi_norm", "")),
            "abstract": clean_value(getattr(row, "abstract", "")),
            "author": clean_value(getattr(row, "author", "")),
            "author_full_names": clean_value(getattr(row, "author_full_names", "")),
            "country": clean_value(getattr(row, "country", "")),
            "source_title": clean_value(getattr(row, "source_title", "")),
            "journal": clean_value(getattr(row, "journal", "")),
            "title": clean_value(getattr(row, "title", "")),
        }
    return lookup


def enrich(record: dict[str, Any], article_meta: dict[str, dict[str, str]]) -> dict[str, Any]:
    meta = article_meta.get(record["id"], {})
    for key in ["doi", "abstract", "author", "author_full_names", "country", "source_title", "journal", "title"]:
        if not record.get(key):
            record[key] = meta.get(key, "")
    if not record.get("title"):
        record["title"] = record["id"]
    return record


def root_records(path: Path, article_meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        records.append(
            enrich(
                {
                    "id": clean_value(row.node_id),
                    "role": "root",
                    "role_label": "Roots",
                    "visual_symbol": "brown circle",
                    "subgroup": clean_value(row.subtopic_id),
                    "subgroup_label": clean_value(row.subtopic_label),
                    "title": clean_value(row.title),
                    "year": clean_year(row.year),
                    "journal": clean_value(row.journal),
                    "doi": "",
                    "citations_received": clean_number(row.citations_received),
                    "citation_velocity": clean_number(row.citation_velocity),
                    "size": clean_number(row.size),
                    "color": clean_value(row.color),
                    "notes": "Foundational paper in a root subtopic.",
                },
                article_meta,
            )
        )
    return records


def trunk_records(path: Path, article_meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        records.append(
            enrich(
                {
                    "id": clean_value(row.node_id),
                    "role": "trunk",
                    "role_label": "Trunk",
                    "visual_symbol": "brown circle",
                    "subgroup": clean_value(row.trunk_subtopic_id),
                    "subgroup_label": clean_value(row.trunk_subtopic_label),
                    "title": clean_value(row.title),
                    "year": clean_year(row.year),
                    "journal": clean_value(row.journal),
                    "doi": "",
                    "citations_received": clean_number(row.citations_received),
                    "citation_velocity": clean_number(row.citation_velocity),
                    "size": clean_number(row.size),
                    "color": clean_value(row.color),
                    "notes": "Consolidating paper in a trunk subtopic.",
                },
                article_meta,
            )
        )
    return records


def branch_records(path: Path, article_meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        member_role = clean_value(row.branch_member_role)
        records.append(
            enrich(
                {
                    "id": clean_value(row.node_id),
                    "role": "branch",
                    "role_label": "Branches",
                    "visual_symbol": "brown circle",
                    "subgroup": clean_value(row.branch_label),
                    "subgroup_label": f"{clean_value(row.branch_label)} - {member_role}",
                    "branch_member_role": member_role,
                    "title": clean_value(row.title),
                    "year": clean_year(row.year),
                    "journal": clean_value(row.journal),
                    "doi": "",
                    "branch_core_score": clean_number(row.branch_core_score),
                    "semantic_similarity_to_branch": clean_number(row.semantic_similarity_to_branch),
                    "trunk_connection_score": clean_number(row.trunk_connection_score),
                    "notes": clean_value(row.role_reason),
                },
                article_meta,
            )
        )
    return records


def leaf_records(
    path: Path,
    fruit_ids: set[str],
    article_meta: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        node_id = clean_value(row.node_id)
        if node_id in fruit_ids:
            continue
        internal_outdegree = clean_number(row.internal_outdegree)
        effective_year = clean_year(getattr(row, "effective_year", ""))
        temporal_label = "dormant frontier paper"
        if effective_year is not None and effective_year >= 2021:
            temporal_label = "recent frontier paper"
        records.append(
            enrich(
                {
                    "id": node_id,
                    "role": "leaf",
                    "role_label": "Leaves",
                    "visual_symbol": "leaf",
                    "subgroup": temporal_label,
                    "subgroup_label": temporal_label,
                    "title": clean_value(row.title),
                    "year": effective_year,
                    "journal": clean_value(row.journal),
                    "source_title": clean_value(row.source_title),
                    "doi": clean_value(row.doi),
                    "internal_indegree": clean_number(row.internal_indegree),
                    "internal_outdegree": internal_outdegree,
                    "sap_rank": clean_number(row.sap_rank),
                    "notes": "Leaf-like frontier paper: internal indegree = 0 and internal outdegree >= 1.",
                },
                article_meta,
            )
        )
    return records


def fruit_records(path: Path, article_meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    records = []
    for row in df.itertuples(index=False):
        records.append(
            enrich(
                {
                    "id": clean_value(row.SR),
                    "role": "fruit",
                    "role_label": "Fruits",
                    "visual_symbol": "apple",
                    "subgroup": clean_value(row.fruit_signal_type),
                    "subgroup_label": "High external-attention paper",
                    "title": clean_value(row.title),
                    "year": clean_year(row.year),
                    "journal": clean_value(row.journal),
                    "source_title": clean_value(row.source_title),
                    "doi": clean_value(row.doi),
                    "cited_by": clean_number(row.cited_by),
                    "internal_indegree": clean_number(row.internal_indegree),
                    "external_citation_velocity": clean_number(row.external_citation_velocity),
                    "fruit_score": clean_number(row.fruit_score),
                    "color": clean_value(row.fruit_color),
                    "notes": "Fruit candidate with external citation attention and weak local absorption.",
                },
                article_meta,
            )
        )
    return records


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    articles = pd.read_csv(args.articles)
    article_meta = metadata_lookup(articles)
    fruits = fruit_records(args.fruits, article_meta)
    fruit_ids = {record["id"] for record in fruits}
    records = []
    records.extend(root_records(args.roots, article_meta))
    records.extend(trunk_records(args.trunk, article_meta))
    records.extend(branch_records(args.branches, article_meta))
    records.extend(leaf_records(args.leaves, fruit_ids, article_meta))
    records.extend(fruits)
    return records


def role_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["role"]] = counts.get(record["role"], 0) + 1
    return counts


def strip_xml_declaration(svg_text: str) -> str:
    return "\n".join(
        line for line in svg_text.splitlines() if not line.strip().startswith("<?xml")
    )


def build_html(svg_text: str, records: list[dict[str, Any]]) -> str:
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    counts_json = json.dumps(role_counts(records), ensure_ascii=False)
    escaped_count = html.escape(str(len(records)))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive Tree of Science 3</title>
  <style>
    :root {{
      --paper: #fbfaf7;
      --ink: #24180f;
      --muted: #76685d;
      --line: #d8d0c8;
      --root: #6a3a16;
      --leaf: #7da721;
      --fruit: #b42316;
      --panel: rgba(255, 255, 255, 0.94);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 10%, rgba(218, 232, 172, 0.22), transparent 30%),
        linear-gradient(135deg, #fffdfa, var(--paper));
      font-family: Georgia, "Times New Roman", serif;
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      gap: 18px;
      padding: 18px;
    }}
    .figure-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(52, 33, 18, 0.10);
    }}
    .figure-card {{
      position: relative;
      overflow: hidden;
      min-height: calc(100vh - 36px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }}
    .figure-wrap {{
      position: relative;
      width: min(100%, 1180px);
      aspect-ratio: 911.15852 / 732.26797;
    }}
    .figure-wrap > svg {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }}
    .hotspot {{
      fill: transparent;
      stroke: transparent;
      cursor: pointer;
      pointer-events: auto;
    }}
    .hotspot:hover, .hotspot.active {{
      fill: rgba(140, 100, 72, 0.08);
      stroke: rgba(92, 38, 9, 0.45);
      stroke-width: 1.2;
      stroke-dasharray: 5 4;
    }}
    .panel {{
      max-height: calc(100vh - 36px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .panel header {{
      padding: 22px 22px 14px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      font-size: 25px;
      line-height: 1.1;
      margin: 0 0 7px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
    }}
    .controls {{
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }}
    input[type="search"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 11px 14px;
      font: 15px/1.2 Georgia, "Times New Roman", serif;
      background: #fff;
      color: var(--ink);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font: 13px/1 Georgia, "Times New Roman", serif;
    }}
    .chip.active {{
      color: white;
      border-color: var(--root);
      background: var(--root);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
      margin-top: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 6px;
      text-align: center;
      background: #fff;
    }}
    .stat strong {{ display: block; font-size: 17px; }}
    .stat span {{ display: block; font-size: 10px; color: var(--muted); text-transform: uppercase; }}
    .content {{
      display: grid;
      grid-template-rows: minmax(180px, 1fr) auto;
      min-height: 0;
    }}
    .list {{
      overflow: auto;
      padding: 12px;
      min-height: 0;
    }}
    .item {{
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 15px;
      padding: 11px 12px;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .item:hover, .item.active {{
      background: #fff;
      border-color: var(--line);
    }}
    .item-title {{
      display: block;
      font-size: 14px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .item-meta {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }}
    .details {{
      border-top: 1px solid var(--line);
      padding: 16px 18px 18px;
      background: rgba(246, 241, 234, 0.65);
      max-height: 38vh;
      overflow: auto;
    }}
    .details h2 {{
      margin: 0 0 7px;
      font-size: 19px;
      line-height: 1.15;
    }}
    .details .meta {{
      color: var(--muted);
      margin: 0 0 10px;
      font-size: 13px;
    }}
    .field {{
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      gap: 10px;
      margin: 7px 0;
      font-size: 13px;
      line-height: 1.3;
    }}
    .field b {{ color: #4e3828; }}
    .abstract {{
      margin-top: 10px;
      color: #3a2a1f;
      font-size: 13px;
      line-height: 1.38;
    }}
    a {{ color: #7c2b12; }}
    .empty {{
      padding: 26px 16px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 1050px) {{
      .app {{ grid-template-columns: 1fr; }}
      .figure-card {{ min-height: auto; }}
      .panel {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <section class="figure-card" aria-label="Interactive Tree of Science 3 figure">
      <div class="figure-wrap" id="figureWrap">
        {strip_xml_declaration(svg_text)}
        <svg class="overlay" viewBox="0 0 911.15852 732.26797" aria-hidden="true">
          <rect class="hotspot" data-role="branch" data-subgroup="branch_1" x="0" y="92" width="330" height="340" rx="16" />
          <rect class="hotspot" data-role="branch" data-subgroup="branch_2" x="250" y="0" width="360" height="260" rx="16" />
          <rect class="hotspot" data-role="branch" data-subgroup="branch_3" x="560" y="95" width="340" height="360" rx="16" />
          <rect class="hotspot" data-role="leaf" x="0" y="0" width="911" height="470" rx="18" />
          <rect class="hotspot" data-role="trunk" x="405" y="285" width="120" height="300" rx="18" />
          <rect class="hotspot" data-role="root" x="260" y="525" width="420" height="205" rx="18" />
          <circle class="hotspot" data-role="fruit" cx="132" cy="378" r="55" />
          <circle class="hotspot" data-role="fruit" cx="355" cy="150" r="48" />
          <circle class="hotspot" data-role="fruit" cx="694" cy="248" r="40" />
        </svg>
      </div>
    </section>
    <aside class="panel" aria-label="ToS 3 metadata panel">
      <header>
        <h1>Interactive ToS 3</h1>
        <p class="subtitle">Faithful SVG view with searchable analytical metadata. Records loaded: {escaped_count}.</p>
      </header>
      <section class="controls">
        <input id="search" type="search" placeholder="Search author, title, DOI, role..." />
        <div class="chips" id="chips">
          <button class="chip active" data-role="all">All</button>
          <button class="chip" data-role="root">Roots</button>
          <button class="chip" data-role="trunk">Trunk</button>
          <button class="chip" data-role="branch">Branches</button>
          <button class="chip" data-role="leaf">Leaves</button>
          <button class="chip" data-role="fruit">Fruits</button>
        </div>
        <div class="stats" id="stats"></div>
      </section>
      <section class="content">
        <div class="list" id="list"></div>
        <div class="details" id="details">
          <h2>Select a paper</h2>
          <p class="meta">Use search, filters, or click a region of the tree.</p>
        </div>
      </section>
    </aside>
  </main>
  <script>
    const records = {data_json};
    const counts = {counts_json};
    const state = {{ role: "all", subgroup: "", query: "", selectedId: "" }};
    const roleLabels = {{
      root: "Roots",
      trunk: "Trunk",
      branch: "Branches",
      leaf: "Leaves",
      fruit: "Fruits"
    }};

    const search = document.getElementById("search");
    const list = document.getElementById("list");
    const details = document.getElementById("details");
    const stats = document.getElementById("stats");
    const chips = document.getElementById("chips");

    function compactNumber(value) {{
      if (value === null || value === undefined || value === "") return "";
      const n = Number(value);
      if (Number.isNaN(n)) return "";
      return Number.isInteger(n) ? String(n) : n.toFixed(3);
    }}

    function doiLink(doi) {{
      if (!doi) return "";
      const clean = String(doi).replace(/^https?:\\/\\/doi.org\\//i, "");
      return `<a href="https://doi.org/${{encodeURIComponent(clean)}}" target="_blank" rel="noreferrer">${{clean}}</a>`;
    }}

    function matches(record) {{
      if (state.role !== "all" && record.role !== state.role) return false;
      if (state.subgroup && record.subgroup !== state.subgroup) return false;
      if (!state.query) return true;
      const haystack = [
        record.id, record.title, record.role_label, record.subgroup_label,
        record.journal, record.source_title, record.doi, record.year,
        record.branch_member_role
      ].join(" ").toLowerCase();
      return haystack.includes(state.query.toLowerCase());
    }}

    function filteredRecords() {{
      return records.filter(matches).sort((a, b) => {{
        const roleOrder = {{ root: 1, trunk: 2, branch: 3, leaf: 4, fruit: 5 }};
        if (roleOrder[a.role] !== roleOrder[b.role]) return roleOrder[a.role] - roleOrder[b.role];
        return (Number(b.year || 0) - Number(a.year || 0)) || String(a.id).localeCompare(String(b.id));
      }});
    }}

    function renderStats() {{
      stats.innerHTML = ["root", "trunk", "branch", "leaf", "fruit"].map(role => `
        <button class="stat" data-role="${{role}}" title="Filter ${{roleLabels[role]}}">
          <strong>${{counts[role] || 0}}</strong>
          <span>${{roleLabels[role]}}</span>
        </button>
      `).join("");
      stats.querySelectorAll(".stat").forEach(button => {{
        button.addEventListener("click", () => setRole(button.dataset.role));
      }});
    }}

    function renderList() {{
      const items = filteredRecords();
      if (!items.length) {{
        list.innerHTML = `<div class="empty">No records match the current filter.</div>`;
        return;
      }}
      list.innerHTML = items.map(record => `
        <button class="item ${{record.id === state.selectedId ? "active" : ""}}" data-id="${{escapeHtml(record.id)}}">
          <span class="item-title">${{escapeHtml(record.id)}}</span>
          <span class="item-meta">${{escapeHtml(record.role_label || "")}}${{record.subgroup_label ? " - " + escapeHtml(record.subgroup_label) : ""}}${{record.year ? " - " + record.year : ""}}</span>
        </button>
      `).join("");
      list.querySelectorAll(".item").forEach(button => {{
        button.addEventListener("click", () => selectRecord(button.dataset.id));
      }});
    }}

    function field(label, value) {{
      if (value === null || value === undefined || value === "") return "";
      return `<div class="field"><b>${{label}}</b><span>${{value}}</span></div>`;
    }}

    function selectRecord(id) {{
      const record = records.find(item => item.id === id);
      if (!record) return;
      state.selectedId = id;
      details.innerHTML = `
        <h2>${{escapeHtml(record.title || record.id)}}</h2>
        <p class="meta">${{escapeHtml(record.id)}}${{record.year ? " - " + record.year : ""}}</p>
        ${{field("Role", escapeHtml(record.role_label || ""))}}
        ${{field("Group", escapeHtml(record.subgroup_label || record.subgroup || ""))}}
        ${{field("Journal", escapeHtml(record.journal || record.source_title || ""))}}
        ${{field("DOI", doiLink(record.doi))}}
        ${{field("Citations", compactNumber(record.citations_received ?? record.cited_by))}}
        ${{field("Velocity", compactNumber(record.citation_velocity ?? record.external_citation_velocity))}}
        ${{field("Core score", compactNumber(record.branch_core_score))}}
        ${{field("Fruit score", compactNumber(record.fruit_score))}}
        ${{field("Outdegree", compactNumber(record.internal_outdegree))}}
        ${{field("Notes", escapeHtml(record.notes || ""))}}
        ${{record.abstract ? `<div class="abstract">${{escapeHtml(record.abstract).slice(0, 1100)}}</div>` : ""}}
      `;
      renderList();
    }}

    function setRole(role, subgroup = "") {{
      state.role = role;
      state.subgroup = subgroup;
      document.querySelectorAll(".chip").forEach(chip => chip.classList.toggle("active", chip.dataset.role === role || (role === "all" && chip.dataset.role === "all")));
      document.querySelectorAll(".hotspot").forEach(hotspot => {{
        const active = hotspot.dataset.role === role && (!subgroup || hotspot.dataset.subgroup === subgroup);
        hotspot.classList.toggle("active", active);
      }});
      renderList();
    }}

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }}[char]));
    }}

    search.addEventListener("input", event => {{
      state.query = event.target.value;
      renderList();
    }});

    chips.addEventListener("click", event => {{
      const button = event.target.closest("[data-role]");
      if (!button) return;
      setRole(button.dataset.role);
    }});

    document.querySelectorAll(".hotspot").forEach(hotspot => {{
      hotspot.addEventListener("click", () => {{
        setRole(hotspot.dataset.role || "all", hotspot.dataset.subgroup || "");
      }});
    }});

    renderStats();
    renderList();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.svg.exists():
        raise FileNotFoundError(f"SVG not found: {args.svg}")

    records = build_records(args)
    svg_text = args.svg.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(svg_text, records), encoding="utf-8")

    counts = role_counts(records)
    print(f"Saved interactive HTML: {args.output.resolve()}")
    print(f"Records: {len(records):,}")
    print(
        "Role counts: "
        + ", ".join(f"{role}={counts.get(role, 0):,}" for role in ["root", "trunk", "branch", "leaf", "fruit"])
    )


if __name__ == "__main__":
    main()
