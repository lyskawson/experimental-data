"""Independent numerical checks for the lab calculations and data exports."""

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from analysis import DATA_DIR, detect_outliers, export_formats, fit_ols, load_anscombe, load_task3
from scipy import stats


class AnalysisChecks(unittest.TestCase):
    def test_ols_matches_independent_solver_and_known_statistics(self):
        for _, group in load_anscombe().groupby("series"):
            fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
            design = np.column_stack([np.ones(len(group)), group.x])
            coefficients = np.linalg.lstsq(design, group.y, rcond=None)[0]
            np.testing.assert_allclose([fit["beta0"], fit["beta1"]], coefficients, atol=1e-12)
            self.assertAlmostEqual(fit["pearson_r"], stats.pearsonr(group.x, group.y).statistic)
            self.assertAlmostEqual(fit["r_squared"], fit["pearson_r"] ** 2)
            self.assertEqual(fit["n"], 11)
            self.assertEqual(fit["mean_x"], 9)
            self.assertEqual(fit["var_x_sample"], 11)
            self.assertEqual(fit["var_x_population"], 10)

    def test_outlier_identity_threshold_stability_and_clean_fit(self):
        data = load_anscombe()
        third = data.loc[data.series == "III"]
        original = third.copy(deep=True)
        for threshold in (3, 3.5, 4):
            diagnostics = detect_outliers(third, threshold=threshold)
            removed = diagnostics.loc[diagnostics.is_outlier]
            self.assertEqual(len(removed), 1)
            self.assertEqual(tuple(removed[["x", "y"]].iloc[0]), (13.0, 12.74))
            retained = diagnostics.loc[~diagnostics.is_outlier]
            fit = fit_ols(retained.x.to_numpy(), retained.y.to_numpy())
            self.assertGreater(fit["r_squared"], 0.99999)
            self.assertLess(fit["rmse"], 0.003)
            self.assertFalse(detect_outliers(retained).is_outlier.any())
        pd.testing.assert_frame_equal(third, original)

    def test_exact_linear_majority_zero_mad_is_handled(self):
        frame = pd.DataFrame({"x": np.arange(7), "y": [1, 3, 5, 7, 9, 11, 40]})
        result = detect_outliers(frame)
        self.assertEqual(result.is_outlier.tolist(), [False] * 6 + [True])

    def test_degenerate_and_nonfinite_data_are_rejected(self):
        for x, y in (([8, 8, 8], [1, 2, 3]), ([1, 2, 3], [1, np.nan, 2])):
            with self.assertRaises(ValueError):
                fit_ols(np.array(x), np.array(y))

    def test_all_serialized_formats_round_trip_without_data_loss(self):
        source = load_anscombe()
        expected = source.to_dict(orient="records")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            sizes = export_formats(source, target)
            csv = pd.read_csv(target / "anscombe.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(csv, source)
            for filename in ("anscombe.json", "anscombe_pretty.json"):
                self.assertEqual(json.loads((target / filename).read_text()), expected)
            for filename in ("anscombe.xml", "anscombe_pretty.xml"):
                elements = ET.parse(target / filename).getroot()
                records = [
                    {
                        "series": row.findtext("series"),
                        "x": float(row.findtext("x")),
                        "y": float(row.findtext("y")),
                    }
                    for row in elements
                ]
                self.assertEqual(records, expected)
            for row in sizes.itertuples():
                self.assertEqual(row.bytes, len((target / row.file).read_bytes()))

    def test_task3_integrity_and_required_first_series(self):
        source = load_task3()
        self.assertEqual(len(source), 1846)
        self.assertEqual(source.groupby("series").size().tolist(), [142] * 13)
        provenance = json.loads((DATA_DIR / "provenance.json").read_text())
        digest = hashlib.sha256((DATA_DIR / "dane_zad3.csv").read_bytes()).hexdigest()
        self.assertEqual(digest, provenance["csv_sha256"])
        first = source.loc[source.series == "seria1"]
        np.testing.assert_array_equal(first.iloc[0][["x", "y"]].to_numpy(float), [55.3846, 97.1795])
        self.assertEqual(provenance["series_mapping"]["seria9"], "circle")
        self.assertEqual(provenance["series_mapping"]["seria10"], "bullseye")


if __name__ == "__main__":
    unittest.main()
