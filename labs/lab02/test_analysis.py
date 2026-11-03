"""Independent numerical checks: QR solutions, metric definitions, and KKT conditions."""

import unittest

import numpy as np
from analysis import (
    ALPHAS,
    analyze_1d,
    analyze_2d,
    design_2d,
    fit_ols,
    fit_regularized,
    generate_1d,
    generate_2d,
    row_coefficients,
    surface,
)
from numpy.testing import assert_allclose
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class AnalysisChecks(unittest.TestCase):
    def test_ols_against_qr(self):
        data = generate_1d()
        cases = [(np.vander(data.x, d + 1, increasing=True), data.y) for d in (1, 2, 3)]
        data_2d = generate_2d()
        cases += [(design_2d(data_2d.x, data_2d.y, d), data_2d.z) for d in (2, 3)]
        for design, response in cases:
            with self.subTest(shape=design.shape):
                q, r = np.linalg.qr(design, mode="reduced")
                expected = np.linalg.solve(r, q.T @ response)
                actual = fit_ols(design, response.to_numpy())
                assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
                assert_allclose(design.T @ (design @ actual - response), 0, atol=1e-8)

    def test_scores_use_the_correct_reference(self):
        data = generate_1d()
        for _, row in analyze_1d(data).iterrows():
            prediction = np.polynomial.polynomial.polyval(data.x, row_coefficients(row))
            for reference, mse, r2 in (
                (data.y_true, row.mse_true, row.r2_true),
                (data.y, row.mse_y, row.r2_y),
            ):
                self.assertAlmostEqual(mse, mean_squared_error(reference, prediction), places=13)
                self.assertAlmostEqual(r2, r2_score(reference, prediction), places=13)

    def test_regularization_optimality_and_intercept(self):
        data = generate_1d()
        for degree in (2, 3):
            # Use sklearn's scaler independently of the implementation's manual scaling.
            raw = np.column_stack([data.x**j for j in range(1, degree + 1)])
            scaler = StandardScaler().fit(raw)
            standardized = scaler.transform(raw)
            for method, alphas in ALPHAS.items():
                previous_norm = np.inf
                for alpha in alphas:
                    with self.subTest(degree=degree, method=method, alpha=alpha):
                        beta = fit_regularized(
                            data.x.to_numpy(), data.y.to_numpy(), degree, method, alpha
                        )
                        prediction = beta[0] + raw @ beta[1:]
                        residual = prediction - data.y.to_numpy()
                        weights = beta[1:] * scaler.scale_
                        self.assertAlmostEqual(float(residual.mean()), 0, places=12)
                        gradient = standardized.T @ residual
                        if method == "Ridge":
                            assert_allclose(gradient + alpha * weights, 0, atol=2e-10)
                            penalty_norm = float(np.linalg.norm(weights))
                        else:
                            gradient /= len(data)
                            active = weights != 0
                            assert_allclose(
                                gradient[active] + alpha * np.sign(weights[active]), 0, atol=2e-10
                            )
                            self.assertTrue(np.all(np.abs(gradient[~active]) <= alpha + 2e-10))
                            penalty_norm = float(np.sum(np.abs(weights)))
                        self.assertLessEqual(penalty_norm, previous_norm + 1e-10)
                        previous_norm = penalty_norm

    def test_surface_basis_and_grid_errors(self):
        data = generate_2d()
        self.assertEqual(len(data), 200)
        self.assertTrue(((data[["x", "y"]] >= 0) & (data[["x", "y"]] < 4)).all().all())
        assert_allclose(data.z, surface(data.x, data.y), atol=0)
        results, coefficients = analyze_2d(data)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 4, 40), np.linspace(0, 4, 40))
        # Explicit monomials check the basis order, including all mixed terms.
        terms = [
            np.ones_like(grid_x),
            grid_x,
            grid_y,
            grid_x**2,
            grid_x * grid_y,
            grid_y**2,
            grid_x**3,
            grid_x**2 * grid_y,
            grid_x * grid_y**2,
            grid_y**3,
        ]
        for row in results.itertuples():
            prediction = sum(value * term for value, term in zip(coefficients[row.degree], terms))
            error = surface(grid_x, grid_y) - prediction
            self.assertAlmostEqual(row.mse_grid, float(np.mean(error**2)), places=13)
            self.assertAlmostEqual(row.max_abs_error, float(np.abs(error).max()), places=12)
            self.assertEqual(row.rank, row.n_coefficients)
            peak = np.unravel_index(np.abs(error).argmax(), error.shape)
            self.assertEqual((row.max_error_x, row.max_error_y), (grid_x[peak], grid_y[peak]))


if __name__ == "__main__":
    unittest.main()
