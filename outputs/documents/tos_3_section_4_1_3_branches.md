### 4.1.3 Branches

In the enhanced Tree of Science, branches are redefined as recent and coherent thematic conversations that remain connected to the intellectual structure of the field. This definition is intentionally more conservative than treating branches as trends. A trend implies additional evidence of acceleration, novelty, field-level momentum, and independent consolidation. A branch, in contrast, is designed to help the researcher identify what recent papers are discussing together and how those discussions extend the trunk of the literature.

This distinction is important because a structurally valid group of recent papers is not necessarily a scientific trend. Citation-based communities may contain substantive papers, methodological references, background theories, and peripheral studies. If all of them are interpreted as equal members of a trend, the resulting recommendation becomes difficult to read and difficult to use when writing a literature review. Therefore, the enhanced algorithm treats branches first as thematic reading structures and only secondarily evaluates whether they show evidence of emergence.

The branch-identification procedure begins with the citation network enriched with article metadata and SAP attributes. The original SAP behavior based on Louvain community detection is preserved as a baseline for comparability with previous Tree of Science implementations. However, for the enhanced version, Leiden community detection is used as an experimental improvement because it provides stronger guarantees of internally connected communities. This is relevant for Tree of Science because a branch should represent a connected intellectual trajectory rather than a disconnected modularity artifact.

After detecting branch communities, a cohesion-preserving selection step is applied. Instead of selecting recent branch papers purely by year, the algorithm expands from a strong recent candidate while preserving a connected induced subset of papers. This prevents the final branch from losing the bridge papers that make the branch structurally interpretable. In the current implementation, each branch is limited to a manageable number of papers, making the result useful for reading and writing rather than producing an overloaded visualization.

The branch layer is then evaluated through three complementary forms of evidence. First, each branch is assigned to the closest trunk subtopic using citation-path proximity, textual similarity, and shared-reference similarity. This indicates which structural conversation in the trunk the branch extends. The branch-to-trunk score is defined as:

```text
BTS(p, T) =
w1 * citation_path_proximity(p, T)
+ w2 * text_similarity(p, T)
+ w3 * shared_reference_similarity(p, T)
```

where `p` is a branch paper and `T` is a trunk subtopic. Citation-path proximity captures whether the branch paper is structurally close to the trunk. Text similarity is calculated from title and abstract information. Shared-reference similarity captures whether the branch paper and the trunk subtopic rely on similar intellectual foundations.

Second, branch-level diagnostics estimate recency, semantic coherence, citation velocity, and structural support. Third, an adapted emergence model evaluates whether a branch behaves like an active but heterogeneous candidate, a preliminary candidate, a mature niche, or a specialized application domain. These diagnostics are not used to force every branch into the category of "trend"; instead, they provide interpretive warnings and support the researcher's judgment.

The most important refinement is the classification of papers within each branch. Each paper is assigned one of four interpretive roles: core, peripheral, background/methodological, or missing metadata. Core papers are those with a direct topical signal, adequate semantic similarity to the branch conversation, and sufficient structural or temporal relevance. Peripheral papers are related to the branch but weaker as defining papers. Background or methodological papers may support the branch but should not be interpreted as substantive members of the thematic conversation. Missing-metadata records are retained for transparency but should be treated cautiously until their bibliographic information is completed.

The role assignment is supported by a branch core score:

```text
BCS(p) =
w1 * semantic_similarity_to_branch(p)
+ w2 * recency(p)
+ w3 * trunk_connection(p)
```

Semantic similarity measures how close the paper is to the branch centroid built from title and abstract terms. Recency gives priority to papers from the most recent window. Trunk connection uses the branch-to-trunk score to preserve the relationship between branches and the intellectual structure of the field. In addition to this score, rule-based checks identify methodological/background papers and records with insufficient metadata.

This role classification changes the practical meaning of branches. Instead of asking the researcher to read all branch papers as if they were equally representative, the algorithm identifies which papers define the branch and which papers should be interpreted as support, context, or possible noise. This makes the recommendation more useful for literature-review writing. A researcher can begin with the core papers to understand the recent thematic conversation, then decide whether peripheral and background papers are necessary for theoretical or methodological context.

Finally, branches are evaluated with optional emergence diagnostics. These diagnostics do not define what a branch is; instead, they help determine whether a branch may also be interpreted as an emerging direction, a mature niche, or a heterogeneous candidate. The diagnostic score is:

```text
BES(B) =
w1 * novelty_growth(B)
+ w2 * persistence_coherence(B)
+ w3 * scientific_impact(B)
+ w4 * uncertainty_reduction(B)
+ w5 * reference_novelty(B)
```

where `B` is a branch. Novelty and growth capture recent activity, persistence and coherence capture thematic and structural stability, scientific impact captures citation velocity and trunk connection, uncertainty reduction captures the branch's structural specificity, and reference novelty captures unusual recombinations of prior knowledge. Because some of these indicators depend on incomplete metadata or sparse reference data, they are used only as secondary interpretive signals.

The current results illustrate why this distinction is necessary. One branch captures a recent but heterogeneous conversation around contextual extensions of entrepreneurial marketing, including sustainability, immigrant entrepreneurship, direct selling, SMEs, and the marketing-entrepreneurship interface. A second branch captures a smaller emerging direction around organizational and knowledge-based enablers of entrepreneurial marketing, but it also includes adjacent knowledge-management and organizational-behavior references. A third branch is thematically coherent around entrepreneurial marketing in arts, culture, nonprofit, museum, and tourism contexts, but it behaves more like a mature niche than an emerging trend.

Thus, the enhanced branch algorithm does not claim that every branch is a trend. Its contribution is to identify recent, connected, and interpretable thematic conversations, while also showing whether each branch is best understood as an emerging direction, a heterogeneous candidate, a mature niche, or a specialized application domain. This approach preserves the recommender logic of Tree of Science: the algorithm provides structured inputs, but the final interpretation remains with the researcher.
