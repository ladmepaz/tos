from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = Path("data/raw/bibfusion/All_Articles.csv")
DEFAULT_OUTPUT = Path("data/processed/bibfusion/All_Articles_tidy.csv")

ARTICLE_REPLACEMENTS = {
    "COLLINSON E, 2001": "COLLINSON E, 2001, MANAGEMENT DECISION",
    "COLLINSON E, 2001, MANAG DECIS": "COLLINSON E, 2001, MANAGEMENT DECISION",
    "HILLS G": "HILLS G, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "HILLS GERALD E, 2010, INTERNATIONAL JOURNAL OF ENTREPRENEURSHIP AND INNOVATION MANAGEMENT": "HILLS G, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "ALQAHTANI N, 2018, J BUS RES": "ALQAHTANI N, 2020, J BUS RES",
    "KRAUS SASCHA, 2010, INTERNATIONAL JOURNAL OF ENTREPRENEURSHIP AND INNOVATION MANAGEMENT": "KRAUS S, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "JONES R, 2011, JOURNAL OF SMALL BUSINESS AND ENTREPRENEURSHIP": "JONES R, 2011, J SMALL BUS ENTREP",
    "LUMPKIN G, 1996, ACAD MANAGE REV": "LUMPKIN GT, 1996, ACAD MANAGE REV",
    "MILES M, 2006, EUR J MARK": "MILES MP, 2006, EUR J MARKETING",
    "MORRISH S, 2010, J STRATEG MARK": "MORRISH SC, 2011, J RES MARK ENTREP",
    "WHALEN P, 2015, J STRATEGIC MARKETIN": "WHALEN P, 2016, J STRATEG MARK",
    "MORRIS M, 2002": "MORRIS MH, 2002, J MARKET THEORY PRAC",
    "NARVER J, 1990, J MARKETING": "NARVER JC, 1990, J MARKETING",
    "SARASVATHY S, 2001, ACAD MANAGE REV": "SARASVATHY SD, 2001, ACAD MANAGE REV",
}

MULTI_DOI_CANONICAL_IDS = {
    "NARVER JC, 1990, J MARKETING",
    "FORNELL C, 1981, J MARKETING RES",
}

PREFERRED_SOURCE_ROWS = {
    "HILLS G, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE": "HILLS GERALD E, 2010, INTERNATIONAL JOURNAL OF ENTREPRENEURSHIP AND INNOVATION MANAGEMENT",
}

MANUAL_ARTICLE_METADATA = {
    "BJERKE B, 2002, ENTREPRENEURIAL MARK": {
        "title": "Entrepreneurial Marketing: The Growth of Small Firms in the New Economic Era",
        "abstract": (
            "The aim of this book is to provide greater insights into the marketing "
            "entrepreneurship interface by demonstrating the importance of both disciplines "
            "in the new economic era. Moreover, the book has been designed for students, "
            "scholars and practitioners of marketing and entrepreneurship. The emphasis is "
            "firmly placed on the small firm operating within the new economic era and "
            "consequently, the book argues that an understanding of entrepreneurial "
            "marketing is essential in order to achieve small firm growth in this era."
        ),
    },
    "MORRISH SC, 2011, J RES MARK ENTREP": {
        "title": "Entrepreneurial marketing: a strategy for the twenty-first century?",
        "abstract": (
            "Purpose The purpose of this paper is to present the author's view of the "
            "role of entrepreneurial marketing (EM) as a strategy to address the "
            "dynamic marketing environment of recent times. Design/methodology/approach "
            "The author reflects on some significant marketing changes and provides some "
            "contemporary example of companies that have successfully adopted EM "
            "approaches and challenged traditional marketing wisdom. Findings EM is best "
            "conceived not as a nexus between marketing and entrepreneurship, but as an "
            "augmented process, where both the entrepreneur and the customer are the core "
            "actors, co-creating value within the marketing environment. Originality/value "
            "While this is an opinion piece, the paper provides evidence of how EM can be "
            "adopted and applied by entrepreneurial firms and challenges marketers to "
            "create and control their own-marketing environment."
        ),
    },
    "MILES MP, 2006, EUR J MARKETING": {
        "title": (
            "Large firms, entrepreneurial marketing processes, and the cycle of "
            "competitive advantage"
        ),
        "abstract": (
            "Purpose - The paper aims to explore how large firms might leverage "
            "entrepreneurial marketing processes to gain and renew competitive "
            "advantage. Design/methodology/approach - The paper applies past research "
            "on entrepreneurial marketing and entrepreneurship with examples from a "
            "long-term case study of firms in New Zealand, Sweden, the UK, and the USA "
            "to illustrate how entrepreneurial marketing processes can be strategically "
            "employed by large firms to create or discover, assess, and exploit "
            "entrepreneurial opportunities more effectively and efficiently. Findings - "
            "The paper offers insight into how large firms leverage entrepreneurial "
            "marketing processes to gain advantage. The findings suggest that, in free "
            "and open markets, entrepreneurial marketing processes can be strategically "
            "employed to create superior value for the firm's customers and owners. "
            "Originality/value - The paper contributes to the work of both academics "
            "working at the marketing/entrepreneurship interface and executives seeking "
            "to leverage marketing to create competitive advantage."
        ),
    },
    "CARSON D, 1995, MARKETING AND ENTREPRENEURSHIP IN SMES - AN INNOVATIVE APPROACH": {
        "title": "Marketing and entrepreneurship in SMEs : an innovative approach",
        "abstract": (
            "The primary thrust of the text is on adapting traditional marketing tools "
            "appropriate for various situations in Small and Medium Enterprises. To that end, "
            "the text approaches both the concepts of marketing and entrepreneurship at the "
            "same time and uses accepted and established marketing theories as a foundation of the text."
        ),
    },
    "MORRIS MH, 2002, J MARKET THEORY PRAC": {
        "abstract": (
            "The purpose of this paper is to critically explore the construct of "
            "entrepreneurial marketing (EM). This term is used as an integrative "
            "conceptualization that reflects such alternative perspectives as guerrilla "
            "marketing, radical marketing, expeditionary marketing, disruptive marketing "
            "and others. Seven core dimensions of EM are identified, and an underlying "
            "theoretical foundation based on resource advantage theory is proposed. A "
            "conceptual model is introduced of key factors surrounding the phenomenon of "
            "entrepreneurial marketing. Conclusions and implications are drawn for theory "
            "and practice, and priorities are proposed for continuing research."
        ),
    },
    "STOKES D, 2000, J RES MARKETING ENTR": {
        "abstract": (
            "This paper considers how marketing can be made more appropriate in "
            "entrepreneurial contexts by proposing a conceptual model of the processes of "
            "marketing as undertaken by entrepreneurs. Although marketing is a key factor "
            "in the survival and development of business ventures, a number of "
            "entrepreneurial characteristics seem to be at variance with marketing "
            "according to the textbook. These include over-reliance on a restricted "
            "customer base, limited marketing expertise, and variable, unplanned effort. "
            "However, entrepreneurs and small business owners interpret marketing in ways "
            "that do not conform to standard textbook theory and practise. An examination "
            "of four key marketing concepts indicates ways in which entrepreneurial "
            "marketing differs from traditional marketing theory. Entrepreneurs tend to be "
            "\"innovation-oriented\", driven by new ideas and intuitive market feel, "
            "rather than customer oriented, or driven by rigorous assessment of market "
            "needs. They target markets through \"bottom-up\" self-selection and "
            "recommendations of customers and other influence groups, rather than relying "
            "on \"top-down\" segmentation, targeting and positioning processes. They "
            "prefer interactive marketing methods to the traditional mix of the four or "
            "seven \"P's\". They gather information through informal networking rather "
            "than formalised intelligence systems. These processes play to entrepreneurial "
            "strengths and represent marketing that is more appropriate in entrepreneurial "
            "contexts, rather than marketing which is second best due to resource limitations."
        ),
    },
    "SHANE S, 2000, ACAD MANAGE REV": {
        "abstract": (
            "To date, the phenomenon of entrepreneurship has lacked a conceptual "
            "framework. In this note we draw upon previous research conducted in the "
            "different social science disciplines and applied fields of business to "
            "create a conceptual framework for the field. With this framework we explain "
            "a set of empirical phenomena and predict a set of outcomes not explained or "
            "predicted by conceptual frameworks already in existence in other fields."
        ),
    },
    "MILLER D, 1983, MANAGE SCI": {
        "abstract": (
            "The objective of the research was to discover the chief determinants of "
            "entrepreneurship, the process by which organizations renew themselves and "
            "their markets by pioneering, innovation, and risk taking. Some authors have "
            "argued that personality factors of the leader are what determine "
            "entrepreneurship, others have highlighted the role played by the structure "
            "of the organization, while a final group have pointed to the importance of "
            "strategy making. We believed that the manner and extent to which "
            "entrepreneurship would be influenced by all of these factors would in large "
            "measure depend upon the nature of the organization. Based upon the work of a "
            "number of authors we derived a crude typology of firms: Simple firms are "
            "small and their power is centralized at the top. Planning firms are bigger, "
            "their goal being smooth and efficient operation through the use of formal "
            "controls and plans. Organic firms strive to be adaptive to their "
            "environments, emphasizing expertise-based power and open communications. The "
            "predictiveness of the typology was established upon a sample of 52 firms "
            "using hypothesis-testing and analysis of variance techniques. We conjectured "
            "that in Simple firms entrepreneurship would be determined by the "
            "characteristics of the leader; in Planning firms it would be facilitated by "
            "explicit and well integrated product-market strategies, and in Organic firms "
            "it would be a function of environment and structure. These hypotheses were "
            "largely borne out by correlational and multiple regression analyses. Any "
            "programs which aim to stimulate entrepreneurship would benefit greatly from "
            "tailoring recommendations to the nature of the target firms."
        ),
    },
    "MATSUNO K, 2002, J MARKETING": {
        "abstract": (
            "The recent literature suggests a potential tension between market "
            "orientation and entrepreneurial proclivity in achieving superior business "
            "performance. This is unsettling for marketers, because it could mean that "
            "being market oriented is detrimental to a firm that is also trying to be "
            "entrepreneurial and successful. To examine this unnerving potential, the "
            "authors investigate structural influences (both direct and indirect) of "
            "entrepreneurial proclivity and market orientation on business performance. "
            "The results indicate that entrepreneurial proclivity has not only a positive "
            "and direct relationship on market orientation but also an indirect and "
            "positive effect on market orientation through the reduction of "
            "departmentalization. The results also suggest that entrepreneurial "
            "proclivity's performance influence is positive when mediated by market "
            "orientation but negative or nonsignificant when not mediated by market "
            "orientation. The authors also provide a discussion and future research implications."
        ),
    },
    "BARNEY J, 1991, J MANAGE": {
        "abstract": (
            "Understanding sources of sustained competitive advantage has become a major "
            "area of research in strategic management. Building on the assumptions that "
            "strategic resources are heterogeneously distributed across firms and that "
            "these differences are stable over time, this article examines the link "
            "between firm resources and sustained competitive advantage. Four empirical "
            "indicators of the potential of firm resources to generate sustained "
            "competitive advantage-value, rareness, imitability, and substitutability-are "
            "discussed. The model is applied by analyzing the potential of several firm "
            "resources for generating sustained competitive advantages. The article "
            "concludes by examining implications of this firm resource model of sustained "
            "competitive advantage for other business disciplines."
        ),
    },
    "KOHLI AK, 1990, J MARKETING": {
        "abstract": (
            "The literature reflects remarkably little effort to develop a framework for "
            "understanding the implementation of the marketing concept. The authors "
            "synthesize extant knowledge on the subject and provide a foundation for "
            "future research by clarifying the construct's domain, developing research "
            "propositions, and constructing an integrating framework that includes "
            "antecedents and consequences of a market orientation. They draw on the "
            "occasional writings on the subject over the last 35 years in the marketing "
            "literature, work in related disciplines, and 62 field interviews with "
            "managers in diverse functions and organizations. Managerial implications of "
            "this research are discussed."
        ),
    },
    "HILLS G, 2011, JOURNAL OF SMALL BUSINESS AND ENTREPRENEURSHIP": {
        "abstract": (
            "Research in entrepreneurial marketing is about 30 years old. During this "
            "period research has followed many trajectories. Two important but divergent "
            "routes are small business marketing and entrepreneurial marketing, mirroring "
            "the discourse of small businesses versus entrepreneurial firms. Today, small "
            "business marketing and entrepreneurial marketing are regarded as separate but "
            "related research fields. Entrepreneurial marketing research has been very "
            "open-minded towards different approaches in methodology, especially compared "
            "to research within mainstream marketing in the US. During this rather long "
            "period of time, advances in other disciplines have been beneficial for our "
            "own research. One such example is the development of effectuation theory "
            "allowing us to understand entrepreneurial decision-making and, consequently, "
            "important aspects of entrepreneurial marketing behaviour. Many of the "
            "research questions regarded as important by scholars in a panel in 1986 when "
            "interest in marketing and entrepreneurship was evolving are still regarded as "
            "important (e.g. new venture growth). Other issues have lost their relevance. "
            "But, overall, many important questions still are waiting for an answer and "
            "the whole field of entrepreneurial marketing offers tremendous research opportunities."
        ),
    },
    "GILMORE A, 2011, J RES MARK ENTREP": {
        "abstract": (
            "Purpose The purpose of this paper is to consider marketing and its relevance "
            "to entrepreneurs and small to medium-sized enterprises (SMEs), and how "
            "entrepreneurs and SMEs owner/managers adapt and use marketing for their "
            "specific requirements during the life of an enterprise. Initially, the paper "
            "will give some background to the subject, including how entrepreneurs and "
            "SMEs owner/managers are defined and their value to the economy. "
            "Design/methodology/approach The discussion draws from the academic "
            "literature and from experience of working with entrepreneurs and SMEs over a "
            "number of years. The background characteristics and frameworks of "
            "entrepreneurial and SMEs marketing are considered, with emphasis on a "
            "pragmatic approach, to try to understand how entrepreneurs and SMEs actually "
            "\"do\" business. Findings The main body of the paper focuses on the nature "
            "of entrepreneurial marketing typically used by SMEs. The key themes of the "
            "discussion are how entrepreneurs and SME owner/managers adapt standard "
            "marketing frameworks to suit their own enterprises, how they use networks to "
            "improve their business activity, the use and development of marketing "
            "management competencies and how they try to use and develop innovative "
            "marketing. Research limitations/implications Finally, the paper comments on "
            "the inter-relationships and relevance of entrepreneurship and marketing for "
            "each other. Originality/value In practice, entrepreneurial and SMEs "
            "marketing is quite different from the marketing frameworks described in the "
            "standard marketing textbooks used to teach most undergraduate students. This "
            "paper illustrates how entrepreneurs and SMEs adapt and use marketing "
            "according to the needs of their enterprises."
        ),
    },
    "HANSEN DJ, 2010, J RES MARK ENTREP": {
        "abstract": (
            "Purpose A group of researchers met in Charleston, South Carolina, USA to "
            "discuss the past and future of the marketing/entrepreneurship interface. The "
            "purpose of this paper is to summarize main discussions from the three-day "
            "summit. Design/methodology/approach Roughly 16 hours of presentations and "
            "discussions were digitally recorded. The lead author reviewed the recordings "
            "making copious notes, which were organized into 17 themes for further "
            "analysis. Future research directions based on discussion around the most "
            "poignant themes are reported. Findings The paper presents nine categories of "
            "discussions around the interface including: the four research perspectives; "
            "\"the future is in the past;\" marketing; entrepreneurship; small business "
            "marketing; entrepreneurial marketing; practical significance; context of "
            "research; and modeling. Research limitations/implications Throughout the "
            "nine sections, this paper highlights considerations for future research. It "
            "suggests that scholars conducting research at the interface consider the "
            "theoretical perspective of their research to improve collective theory "
            "building and better positioning. It suggests that scholars also consider the "
            "firm and industry context of their empirical research. Finally, it suggests "
            "a number of research questions. Practical implications The paper suggests "
            "that during the research design phase, scholars make efforts to consider the "
            "practical significance that will result from their research. In particular, "
            "they should consider that research in start-ups (all businesses start "
            "somewhere) and small businesses (the vast majority of all enterprises) can "
            "have widespread impacts. Originality/value This paper provides a unique "
            "approach to conceptually organizing marketing/entrepreneurship interface "
            "research and provides an abundant source of ideas for future research."
        ),
    },
    "BECHERER RC, 2012, NEW ENGLAND JOURNAL OF ENTREPRENEURSHIP": {
        "abstract": (
            "This study examines how entrepreneurial marketing dimensions "
            "(proactiveness, opportunity focused, leveraging, innovativeness, risk "
            "taking, value creation, and customer intensity) are related to qualitative "
            "and quantitative outcome measures for the SME and the entrepreneur "
            "(including company success, customer success, financial success, "
            "satisfaction with return goals, satisfaction with growth goals, excellence, "
            "and the entrepreneur's standard of living). Using factor analysis, three "
            "success outcome variables (financial, customer, and strong company success) "
            "emerged together. A separate factor analysis identified satisfactory growth "
            "and return goals. Stepwise regression revealed entrepreneurial marketing "
            "impacts outcome variables, particularly value creation. Implications for "
            "entrepreneurs and areas for research are included."
        ),
    },
    "ALQAHTANI N, 2022, J STRATEG MARK": {
        "abstract": (
            "This research introduces a new scale (ENMAR) for measuring entrepreneurial "
            "marketing (EM). The interrelationships between EM, market orientation (MO), "
            "entrepreneurial orientation (EO), firm performance, and the moderating "
            "effects of network structure (i.e. size, diversity, and strength), "
            "environmental variables (i.e. market turbulence, technological turbulence, "
            "competitive intensity, supplier power, and market growth), and firm size "
            "are empirically examined. Using structural equation modeling, data from 401 "
            "U.S. based firms representing a broad spectrum of industries and firm sizes "
            "are analyzed. Empirical findings demonstrate that even after controlling for "
            "MO and EO, EM has a positive and significant impact on firm performance, and "
            "that impact becomes even more pronounced under highly uncertain market "
            "conditions. EM partially mediates the well-established relationships between "
            "MO, EO, and firm performance. While EM robustly boosts firm performance, it "
            "is more frequently practiced by young firms and those in B2B markets, though "
            "it may be particularly beneficial for mid-sized firms"
        ),
    },
}


def is_missing(value: Any) -> bool:
    """Return True for empty values that can be filled from another row."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def split_semicolon_values(value: Any) -> list[str]:
    """Split a semicolon-separated field into cleaned unique values."""
    if is_missing(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def canonicalize_article_sr(row: pd.Series) -> str | None:
    """Resolve article-level exceptions that cannot be handled by SR alone."""
    sr = str(row.get("SR", "")).strip()
    doi_norm = str(row.get("__doi_norm", "")).strip().lower()
    doi = str(row.get("doi", "")).strip().lower()

    if doi_norm == "10.1038/srep42717" or doi == "10.1038/srep42717":
        return None

    return ARTICLE_REPLACEMENTS.get(sr, sr)


def coalesce_article_group(group: pd.DataFrame) -> pd.Series:
    """Merge duplicate article rows, preferring the canonical SR row."""
    canonical_sr = group["SR_canonical"].iloc[0]
    canonical_rows = group[group["SR"] == canonical_sr]
    preferred_source_sr = PREFERRED_SOURCE_ROWS.get(canonical_sr)
    preferred_rows = (
        group[group["SR"] == preferred_source_sr] if preferred_source_sr is not None else group.iloc[0:0]
    )

    if not canonical_rows.empty:
        base = canonical_rows.iloc[0].copy()
        fill_rows = group.drop(index=canonical_rows.index[0])
    elif not preferred_rows.empty:
        base = preferred_rows.iloc[0].copy()
        fill_rows = group.drop(index=preferred_rows.index[0])
    else:
        base = group.iloc[0].copy()
        fill_rows = group.drop(index=group.index[0])

    if canonical_sr in MULTI_DOI_CANONICAL_IDS:
        doi_values = split_semicolon_values(base.get("doi"))
        doi_norm_values = split_semicolon_values(base.get("__doi_norm"))
        for _, candidate in fill_rows.iterrows():
            for value in split_semicolon_values(candidate.get("doi")):
                if value not in doi_values:
                    doi_values.append(value)
            for value in split_semicolon_values(candidate.get("__doi_norm")):
                if value not in doi_norm_values:
                    doi_norm_values.append(value)
        base["doi"] = ";".join(doi_values)
        base["__doi_norm"] = ";".join(doi_norm_values)

    for _, candidate in fill_rows.iterrows():
        for column in group.columns:
            if column == "SR_canonical":
                continue
            if canonical_sr in MULTI_DOI_CANONICAL_IDS and column in {"doi", "__doi_norm"}:
                continue
            if is_missing(base[column]) and not is_missing(candidate[column]):
                base[column] = candidate[column]

    base["SR"] = canonical_sr
    return base.drop(labels=["SR_canonical"])


def tidy_article_metadata(article_df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize article IDs and merge duplicate rows by filling missing fields."""
    tidy_df = article_df.copy()
    tidy_df["SR"] = tidy_df["SR"].astype(str).str.strip()
    tidy_df = tidy_df[~tidy_df["SR"].str.startswith("ANONYMOUS", na=False)]
    tidy_df["SR_canonical"] = tidy_df.apply(canonicalize_article_sr, axis=1)
    tidy_df = tidy_df[tidy_df["SR_canonical"].notna()]

    merged_rows = [
        coalesce_article_group(group)
        for _, group in tidy_df.groupby("SR_canonical", sort=False)
    ]
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    for sr, overrides in MANUAL_ARTICLE_METADATA.items():
        match_mask = merged_df["SR"] == sr
        if not match_mask.any():
            new_row = pd.Series({column: "" for column in article_df.columns})
            new_row["SR"] = sr
            for column, value in overrides.items():
                if column in new_row.index:
                    new_row[column] = value
            merged_df = pd.concat([merged_df, new_row.to_frame().T], ignore_index=True)
            continue
        for column, value in overrides.items():
            merged_df.loc[match_mask, column] = value
    return merged_df[article_df.columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tidied BibFusion article metadata file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_df = pd.read_csv(args.input)
    tidy_df = tidy_article_metadata(article_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(args.output, index=False)

    replacements = article_df["SR"].astype(str).str.strip().isin(ARTICLE_REPLACEMENTS).sum()
    print(f"Input rows: {len(article_df):,}")
    print(f"Rows with explicit replacements: {replacements:,}")
    print(f"Output rows after metadata coalescing: {len(tidy_df):,}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
