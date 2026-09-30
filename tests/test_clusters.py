import unittest

import numpy as np
import pandas as pd

from icx.clusters import _composite, _wide_index, build_clusters


def _scores(series: list[str], log_vol: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series": series,
            "gpu_id": ["nvidia-a100"] * len(series),
            "billing": ["ON_DEMAND"] * len(series),
            "group": ["other_card"] * len(series),
            "eligible": [True] * len(series),
            "log_vol": log_vol,
            "n_unique_prices": [10] * len(series),
        }
    )


def _matrix(dates: list[str], columns: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for series, values in columns.items():
        for date, value in zip(dates, values):
            rows.append(
                {
                    "date": date,
                    "series": series,
                    "gpu_id": "nvidia-a100",
                    "billing": "ON_DEMAND",
                    "price": 1.0,
                    "base_price": value,
                    "index_value": value,
                }
            )
    return pd.DataFrame(rows)


class ClusterTests(unittest.TestCase):
    def test_missing_member_is_dropped_and_weights_are_renormalized(self) -> None:
        dates = pd.date_range("2026-05-17", periods=4, freq="D").strftime("%Y-%m-%d").tolist()
        level = [1.0, 1.1, 1.2, 1.3]
        columns = {"a": level, "b": level, "c": level[:-1]}
        wide = _wide_index(_matrix(dates, columns), dates, pd.Series(["a", "b", "c"]))
        weights = pd.Series({"a": 0.5, "b": 0.25, "c": 0.25})
        composite = _composite(wide, pd.Index(["a", "b", "c"]), weights, min_members=3)
        self.assertAlmostEqual(float(composite["index_value"].iloc[0]), 1.0, places=6)
        self.assertEqual(int(composite["n_members"].iloc[0]), 3)
        self.assertTrue(pd.isna(composite["index_value"].iloc[-1]))
        self.assertEqual(int(composite["n_members"].iloc[-1]), 2)
        # With the minimum lowered, the last date uses a and b only and still equals the level.
        relaxed = _composite(wide, pd.Index(["a", "b", "c"]), weights, min_members=2)
        self.assertAlmostEqual(float(relaxed["index_value"].iloc[-1]), 1.3, places=6)

    def test_two_shapes_land_in_different_clusters(self) -> None:
        dates = pd.date_range("2026-05-17", periods=20, freq="D").strftime("%Y-%m-%d").tolist()
        calm = np.linspace(1.0, 1.1, len(dates))
        wild = 1.0 + 0.2 * np.sin(np.linspace(0, 6, len(dates)))
        columns = {
            "calm-1": calm.tolist(),
            "calm-2": calm.tolist(),
            "calm-3": (calm * 1.01).tolist(),
            "wild-1": wild.tolist(),
            "wild-2": (wild * 1.02).tolist(),
            "wild-3": (wild * 0.98).tolist(),
        }
        membership, composites, stats = build_clusters(
            _matrix(dates, columns),
            _scores(list(columns), [0.01, 0.01, 0.01, 0.05, 0.05, 0.05]),
            dates,
            min_pairs=5,
            k_range=range(2, 3),
        )
        clusters = membership.set_index("series")["cluster"]
        self.assertEqual(len(set(clusters.loc[["calm-1", "calm-2", "calm-3"]])), 1)
        self.assertEqual(len(set(clusters.loc[["wild-1", "wild-2", "wild-3"]])), 1)
        self.assertNotEqual(int(clusters.loc["calm-1"]), int(clusters.loc["wild-1"]))
        self.assertFalse(composites["index_value"].isna().all())
        self.assertEqual(len(stats), 2)


if __name__ == "__main__":
    unittest.main()
