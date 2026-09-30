import unittest

import numpy as np
import pandas as pd

from icx.stability import MIN_OBSERVATIONS, ranked, score_matrix


def _row(series: str, gpu: str, billing: str, date: str, index: float) -> dict:
    return {
        "date": date,
        "series": series,
        "gpu_id": gpu,
        "billing": billing,
        "price": 1.0,
        "base_price": index,
        "index_value": index,
    }


class StabilityTests(unittest.TestCase):
    def test_flat_series_ranks_ahead_of_a_noisy_one(self) -> None:
        dates = pd.date_range("2026-05-17", periods=MIN_OBSERVATIONS, freq="D").strftime("%Y-%m-%d")
        rows = []
        rng = np.random.default_rng(0)
        noisy = 1.0 + rng.normal(0, 0.05, size=len(dates))
        for date, shock in zip(dates, noisy):
            rows.append(_row("nvidia-h100__SPOT", "nvidia-h100", "SPOT", date, 1.5))
            rows.append(_row("nvidia-a100__ON_DEMAND", "nvidia-a100", "ON_DEMAND", date, float(shock)))
        scores = score_matrix(pd.DataFrame(rows), n_dates=len(dates), baseline_gpu="nvidia-h100")
        same = ranked(scores, "same_card")
        other = ranked(scores, "other_card")
        self.assertEqual(same.iloc[0]["series"], "nvidia-h100__SPOT")
        self.assertEqual(same.iloc[0]["log_vol"], 0.0)
        self.assertGreater(other.iloc[0]["log_vol"], 0.0)
        self.assertGreater(other.iloc[0]["abs_dev"], same.iloc[0]["abs_dev"])

    def test_short_series_is_not_eligible(self) -> None:
        dates = pd.date_range("2026-05-17", periods=5, freq="D").strftime("%Y-%m-%d")
        rows = [_row("nvidia-a100__ON_DEMAND", "nvidia-a100", "ON_DEMAND", date, 1.0) for date in dates]
        scores = score_matrix(pd.DataFrame(rows), n_dates=40, baseline_gpu="nvidia-h100")
        self.assertFalse(bool(scores.iloc[0]["eligible"]))
        self.assertTrue(ranked(scores, "other_card").empty)


if __name__ == "__main__":
    unittest.main()
