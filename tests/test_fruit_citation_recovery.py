"""Regression checks for the historical 2026 fruit correction."""
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from citation_corrections import apply_citation_corrections
from export_fruit_visualization_svg import add_visual_metrics, select_top_fruits
from fruit_candidates import add_fruit_scores, build_sap_graph, normalize_series

CORRECTIONS = ROOT / "data/corrections/fruit_citation_recovery_2026.json"
EXPERIMENT = ROOT / "outputs/experiments/fruit_citation_recovery"


class CitationCorrectionsTest(unittest.TestCase):
    def setUp(self):
        self.records = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
        self.frame = pd.DataFrame(self.records)
        self.frame["cited_by"] = float("nan")

    def test_fills_missing_counts_by_sr_and_case_insensitive_doi(self):
        self.frame["doi"] = self.frame.doi.str.lower()
        corrected = apply_citation_corrections(self.frame, CORRECTIONS)
        self.assertEqual(corrected.cited_by.tolist(), [r["cited_by"] for r in self.records])
        self.assertTrue(self.frame.cited_by.isna().all())
        pd.testing.assert_frame_equal(corrected, apply_citation_corrections(corrected, CORRECTIONS))

    def test_never_replaces_reported_zero_or_conflicting_count(self):
        for value in [0, 99, "invalid"]:
            with self.subTest(value=value):
                frame = self.frame.astype({"cited_by": object})
                frame.loc[0, "cited_by"] = value
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    apply_citation_corrections(frame, CORRECTIONS)

    def test_rejects_doi_mismatch_and_missing_record(self):
        frame = self.frame.copy()
        frame.loc[0, "doi"] = "10.0000/unrelated"
        with self.assertRaisesRegex(ValueError, "DOI mismatch"):
            apply_citation_corrections(frame, CORRECTIONS)
        with self.assertRaisesRegex(ValueError, "SR not found"):
            apply_citation_corrections(self.frame.iloc[1:], CORRECTIONS)

    def test_constant_normalization_is_one(self):
        self.assertEqual(normalize_series(pd.Series([7., 7.])).tolist(), [1., 1.])


class FruitReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.articles = pd.read_csv(ROOT / "data/processed/bibfusion/All_Articles_tidy.csv", low_memory=False)
        cls.corrected = apply_citation_corrections(cls.articles, CORRECTIONS)
        cls.graph = build_sap_graph(
            ROOT / "data/processed/bibfusion/All_Citation_tidy.csv",
            ROOT / "data/processed/bibfusion/All_Articles_tidy.csv",
        )
        cls.actual = cls.score(cls.corrected)

    @classmethod
    def score(cls, articles):
        return add_fruit_scores(cls.graph, articles, 2026, 5, False, 3, .35, .35, .20, .10)

    def test_graph_population_and_all_saved_scores(self):
        pd.testing.assert_frame_equal(self.articles, self.corrected)
        self.assertEqual((len(self.graph), self.graph.number_of_edges()), (3183, 12481))
        self.assertEqual(len(self.actual), 120)
        self.assertTrue(self.actual.year.between(2021, 2026).all())
        for path in [EXPERIMENT / "candidates_recalculated.csv", ROOT / "outputs/fruits/fruit_candidates.csv"]:
            expected = pd.read_csv(path).set_index("SR").sort_index()
            expected["baseline_ToS"] = expected.baseline_ToS.fillna("")
            actual = self.actual.set_index("SR").sort_index()
            pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, atol=1e-12, rtol=1e-12)

    def test_future_papers_are_excluded(self):
        future = self.corrected.iloc[[0]].copy()
        future["SR"] = "FUTURE TEST, 2027"
        future["year"] = 2027
        for column in ["__is_main", "ismainarticle", "ismain_wos", "ismain_scopus"]:
            if column in future:
                future[column] = True
        result = self.score(pd.concat([self.corrected, future], ignore_index=True))
        pd.testing.assert_frame_equal(result.reset_index(drop=True), self.actual.reset_index(drop=True))

    def test_top3_and_derivative_tables(self):
        eligible = self.actual[self.actual.fruit_signal_type.eq("external_fruit_underabsorbed")]
        self.assertEqual(len(eligible), 11)
        selected = add_visual_metrics(select_top_fruits(self.actual, 3))
        self.assertEqual(selected.SR.tolist(), [
            "SULAIMAN MABA, 2024, DISCOV SUSTAIN",
            "KAKEESH DF, 2025, J MARK ANAL",
            "HANAYSHA JR, 2022, SUSTAINABILITY-BASEL",
        ])
        self.assertEqual(selected.cited_by.tolist(), [26, 8, 11])
        self.assertEqual(selected.internal_indegree.tolist(), [0, 0, 0])
        self.assertEqual(selected.fruit_score.round(6).tolist(), [.667759, .520196, .514742])
        for actual, relative in [
            (self.actual.head(30), "outputs/fruits/fruit_candidates_top.csv"),
            (eligible, "outputs/fruits/fruit_candidates_underabsorbed.csv"),
            (selected, "outputs/fruit_visualization/fruits_top3.csv"),
        ]:
            expected = pd.read_csv(ROOT / relative)
            expected["baseline_ToS"] = expected.baseline_ToS.fillna("")
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
