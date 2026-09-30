import unittest
from pathlib import Path

import pandas as pd

from icx.config import Baseline
from icx.indices import (
    build_matrix,
    build_vector,
    continuous_sample_start,
    matrix_with_gaps,
    sync_inverse_indices,
)
from icx.panel import PANEL_ARTIFACT, PANEL_COLUMNS, PANEL_KEYS, PANEL_RECIPE
from icx.store import Decision, OutputLayout


def _panel() -> pd.DataFrame:
    rows = [
        ("2026-04-01", "nvidia-h100__ON_DEMAND", "nvidia-h100", "ON_DEMAND", 8.0, 1, 8.0),
        ("2026-05-17", "nvidia-h100__ON_DEMAND", "nvidia-h100", "ON_DEMAND", 4.0, 1, 4.0),
        ("2026-05-18", "nvidia-h100__ON_DEMAND", "nvidia-h100", "ON_DEMAND", 4.4, 1, 4.4),
        ("2026-05-17", "nvidia-h100__SPOT", "nvidia-h100", "SPOT", 2.0, 1, 2.0),
        ("2026-05-18", "nvidia-h100__SPOT", "nvidia-h100", "SPOT", 2.2, 1, 2.2),
        ("2026-05-17", "nvidia-a100__ON_DEMAND", "nvidia-a100", "ON_DEMAND", 2.0, 1, 2.0),
    ]
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


class IndexMathTests(unittest.TestCase):
    def test_continuous_sample_skips_the_isolated_print(self) -> None:
        start = continuous_sample_start(pd.Series(["2026-04-01", "2026-05-17", "2026-05-18"]))
        self.assertEqual(start, "2026-05-17")

    def test_vector_starts_at_one_and_matrix_recovers_the_base_price(self) -> None:
        panel = _panel()
        vector = build_vector(panel, "nvidia-h100__ON_DEMAND")
        matrix = build_matrix(panel, "nvidia-h100__ON_DEMAND")
        self.assertEqual(vector["date"].tolist(), ["2026-05-17", "2026-05-18"])
        self.assertEqual(vector["index_value"].iloc[0], 1.0)
        self.assertAlmostEqual(vector["index_value"].iloc[1], 1.1)
        self.assertFalse(vector["index_value"].isna().any())

        wide = matrix_with_gaps(panel, vector, matrix, "nvidia-h100__ON_DEMAND")
        self.assertTrue(pd.isna(wide.loc["2026-05-18", "nvidia-a100__ON_DEMAND"]))
        self.assertAlmostEqual(wide.loc["2026-05-17", "nvidia-a100__ON_DEMAND"], 2.0)
        recovered = matrix["index_value"] * matrix["price"]
        self.assertTrue((recovered - matrix["base_price"]).abs().max() < 1e-6)


class IndexStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "outputs_test_tmp" / self.id().replace(".", "_")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".env").write_text("GPU_BASE_PLAN=nvidia-h100__ON_DEMAND\n", encoding="utf-8")
        self.layout = OutputLayout(self.root, Baseline.parse("nvidia-h100__ON_DEMAND"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def _write_panel(self, panel: pd.DataFrame, stamp: str) -> None:
        self.layout.shared.write_table(
            PANEL_ARTIFACT,
            panel,
            recipe=PANEL_RECIPE,
            inputs={"src": stamp},
            key_columns=PANEL_KEYS,
            coverage="full",
            changed_keys=None,
        )

    def test_return_to_baseline_skips_until_the_panel_changes(self) -> None:
        self._write_panel(_panel(), "1")
        vector, matrix = sync_inverse_indices(self.layout)
        self.assertEqual(vector.decision, Decision.FULL)
        self.assertEqual(matrix.decision, Decision.FULL)

        other = self.layout.store_for(Baseline.parse("nvidia-a100__ON_DEMAND"))
        other.write_table(
            "02_inverse/vector.csv",
            pd.DataFrame([{"date": "2026-05-17", "price": 1.0, "base_date": "2026-05-17", "base_price": 1.0, "index_value": 1.0}]),
            recipe="inverse-index-v1",
            inputs={"src": "other"},
            key_columns=["date"],
            coverage="full",
            changed_keys=None,
        )

        again_vector, again_matrix = sync_inverse_indices(self.layout)
        self.assertEqual(again_vector.decision, Decision.SKIP)
        self.assertEqual(again_matrix.decision, Decision.SKIP)
        stored_other = other.read_table("02_inverse/vector.csv")
        assert stored_other is not None
        self.assertEqual(stored_other.iloc[0]["price"], 1.0)

        changed = _panel()
        changed.loc[changed["date"].eq("2026-05-18") & changed["series"].eq("nvidia-h100__SPOT"), "price"] = 2.0
        self._write_panel(changed, "2")
        updated_vector, updated_matrix = sync_inverse_indices(self.layout)
        self.assertEqual(updated_vector.decision, Decision.INCREMENTAL)
        self.assertEqual(updated_vector.rows_upserted, 0)
        self.assertEqual(updated_matrix.decision, Decision.INCREMENTAL)
        self.assertEqual(updated_matrix.rows_upserted, 1)
        spot = updated_matrix.frame.loc[
            updated_matrix.frame["series"].eq("nvidia-h100__SPOT") & updated_matrix.frame["date"].eq("2026-05-18")
        ]
        self.assertAlmostEqual(float(spot["index_value"].iloc[0]), 4.4 / 2.0, places=6)
        unchanged = updated_matrix.frame.loc[
            updated_matrix.frame["series"].eq("nvidia-a100__ON_DEMAND") & updated_matrix.frame["date"].eq("2026-05-17")
        ]
        self.assertAlmostEqual(float(unchanged["index_value"].iloc[0]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
