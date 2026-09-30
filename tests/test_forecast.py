import unittest

import pandas as pd

from icx.forecast import MIN_HISTORY, carry_forward_forecast, ridge_forecast


class ForecastTests(unittest.TestCase):
    def test_carry_forward_uses_the_previous_price_only(self) -> None:
        dates = pd.date_range("2026-05-17", periods=MIN_HISTORY + 3, freq="D").strftime("%Y-%m-%d").tolist()
        price = pd.Series(range(len(dates)), index=dates, dtype=float)
        forecast = carry_forward_forecast(dates, price)
        self.assertEqual(forecast["date"].tolist(), dates[MIN_HISTORY:])
        self.assertEqual(forecast["y_pred"].tolist(), [float(MIN_HISTORY - 1), float(MIN_HISTORY), float(MIN_HISTORY + 1)])
        self.assertNotEqual(forecast["y_pred"].iloc[-1], forecast["y_true"].iloc[-1])

    def test_ridge_recovers_a_proportional_price(self) -> None:
        dates = pd.date_range("2026-05-17", periods=MIN_HISTORY + 5, freq="D").strftime("%Y-%m-%d").tolist()
        other = pd.DataFrame({"other__ON_DEMAND": [1 + 0.01 * i for i in range(len(dates))]}, index=dates)
        baseline = other["other__ON_DEMAND"] * 2
        forecast = ridge_forecast(dates, baseline, other)
        self.assertEqual(len(forecast), 5)
        self.assertLess(float((forecast["y_pred"] - forecast["y_true"]).abs().max()), 0.05)


if __name__ == "__main__":
    unittest.main()
