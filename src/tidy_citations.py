from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/bibfusion/All_Citation.csv")
DEFAULT_OUTPUT = Path("data/processed/bibfusion/All_Citation_tidy.csv")

REFERENCE_REPLACEMENTS = {
    "COLLINSON E, 2001": "COLLINSON E, 2001, MANAGEMENT DECISION",
    "COLLINSON E, 2001, MANAG DECIS": "COLLINSON E, 2001, MANAGEMENT DECISION",
    "EGGERS F, 2018, J BUS RES": "EGGERS F, 2020, J BUS RES",
    "HILLS G, 2009, INT J ENTREPRENEURSHIP INNOV MANAGE": "HILLS G, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "HILLS GERALD E, 2010, INTERNATIONAL JOURNAL OF ENTREPRENEURSHIP AND INNOVATION MANAGEMENT": "HILLS G, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "ALQAHTANI N, 2018, J BUS RES": "ALQAHTANI N, 2020, J BUS RES",
    "KRAUS S, 2009, INT J ENTREPRENEURSHIP INNOV MANAGE": "KRAUS S, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "KRAUS SASCHA, 2010, INTERNATIONAL JOURNAL OF ENTREPRENEURSHIP AND INNOVATION MANAGEMENT": "KRAUS S, 2010, INT J ENTREPRENEURSHIP INNOV MANAGE",
    "STOKES D, 2000": "STOKES D, 2000, J RES MARKETING ENTR",
    "STOKES D, 2000, J RES MARK ENTREP": "STOKES D, 2000, J RES MARKETING ENTR",
    "JONES R, 2011": "JONES R, 2011, INT SMALL BUS J",
    "JONES R, 2011, JOURNAL OF SMALL BUSINESS AND ENTREPRENEURSHIP": "JONES R, 2011, J SMALL BUS ENTREP",
    "LUMPKIN G, 1996, ACAD MANAGE REV": "LUMPKIN GT, 1996, ACAD MANAGE REV",
    "MILES M, 2006, EUR J MARK": "MILES MP, 2006, EUR J MARKETING",
    "MILES M, 2014, J STRATEG MARK": "MILES M, 2015, J STRATEG MARK",
    "MORRISH S, 2010, J STRATEG MARK": "MORRISH SC, 2011, J RES MARK ENTREP",
    "WHALEN P, 2015, J STRATEG MARK": "WHALEN P, 2016, J STRATEG MARK",
    "WHALEN P, 2015, J STRATEGIC MARKETIN": "WHALEN P, 2016, J STRATEG MARK",
    "MORRIS M, 2002": "MORRIS MH, 2002, J MARKET THEORY PRAC",
    "NARVER J, 1990, J MARKETING": "NARVER JC, 1990, J MARKETING",
    "SARASVATHY S, 2001, ACAD MANAGE REV": "SARASVATHY SD, 2001, ACAD MANAGE REV",
}


def tidy_citation_references(citation_df: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit reference-ID corrections before graph construction."""
    tidy_df = citation_df.copy()
    tidy_df["SR"] = tidy_df["SR"].astype(str).str.strip()
    tidy_df["SR_ref"] = tidy_df["SR_ref"].astype(str).str.strip()
    tidy_df["SR"] = tidy_df["SR"].replace(REFERENCE_REPLACEMENTS)
    tidy_df["SR_ref"] = tidy_df["SR_ref"].replace(REFERENCE_REPLACEMENTS)
    tidy_df = tidy_df[(tidy_df["SR"] != "") & (tidy_df["SR_ref"] != "")]
    tidy_df = tidy_df[
        ~tidy_df["SR"].str.startswith("ANONYMOUS", na=False)
        & ~tidy_df["SR_ref"].str.startswith("ANONYMOUS", na=False)
    ]
    tidy_df = tidy_df.drop_duplicates()
    return tidy_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tidied BibFusion citation edge file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    citation_df = pd.read_csv(args.input, usecols=["SR", "SR_ref"])
    tidy_df = tidy_citation_references(citation_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tidy_df.to_csv(args.output, index=False)

    replacements = sum(
        (citation_df["SR"].astype(str).str.strip().isin(REFERENCE_REPLACEMENTS))
        | (citation_df["SR_ref"].astype(str).str.strip().isin(REFERENCE_REPLACEMENTS))
    )
    print(f"Input rows: {len(citation_df):,}")
    print(f"Rows with explicit replacements: {replacements:,}")
    print(f"Output rows after de-duplication: {len(tidy_df):,}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
