import unittest
from pathlib import Path

import pandas as pd

from icx.config import Baseline
from icx.panel import PANEL_COLUMNS, sync_price_panel
from icx.store import Decision, OutputLayout


def _write_series(root: Path, series: str, provider_rows: list[tuple], summary_rows: list[tuple]) -> None:
    parts = root / "data" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    providers = pd.DataFrame(
        provider_rows,
        columns=["date", "provider_id", "hourly_per_gpu"],
    )
    providers.to_csv(parts / f"{series}.providers.csv", index=False)
    summary = pd.DataFrame(summary_rows, columns=["date", "avg_price"])
    summary.to_csv(parts / f"{series}.summary.csv", index=False)


class PanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / "outputs_test_tmp" / self.id().replace(".", "_")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".env").write_text("GPU_BASE_PLAN=nvidia-h100__ON_DEMAND\n", encoding="utf-8")
        self.layout = OutputLayout(self.root, Baseline.parse("nvidia-h100__ON_DEMAND"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def test_each_provider_has_one_vote(self) -> None:
        # Provider A posts two cheap quotes. Provider B posts one expensive quote.
        # A row-level median would follow A. A provider-level median stays in the middle.
        _write_series(
            self.root,
            "nvidia-h100__ON_DEMAND",
            [
                ("2026-05-17", "a", 1.0),
                ("2026-05-17", "a", 1.0),
                ("2026-05-17", "b", 3.0),
            ],
            [("2026-05-17", 1.5)],
        )
        result = sync_price_panel(self.layout)
        self.assertEqual(result.decision, Decision.FULL)
        row = result.frame.iloc[0]
        self.assertEqual(row["price"], 2.0)
        self.assertEqual(int(row["n_providers"]), 2)
        self.assertEqual(row["avg_price"], 1.5)
        self.assertEqual(list(result.frame.columns), PANEL_COLUMNS)

    def test_second_sync_skips_and_a_new_date_is_incremental(self) -> None:
        _write_series(
            self.root,
            "nvidia-h100__ON_DEMAND",
            [("2026-05-17", "a", 4.0)],
            [("2026-05-17", 4.0)],
        )
        _write_series(
            self.root,
            "nvidia-a100__SPOT",
            [("2026-05-17", "a", 1.0)],
            [("2026-05-17", 1.0)],
        )
        first = sync_price_panel(self.layout)
        self.assertEqual(first.decision, Decision.FULL)
        second = sync_price_panel(self.layout)
        self.assertEqual(second.decision, Decision.SKIP)
        self.assertEqual(len(second.frame), 2)

        _write_series(
            self.root,
            "nvidia-a100__SPOT",
            [("2026-05-17", "a", 1.0), ("2026-05-18", "a", 1.5)],
            [("2026-05-17", 1.0), ("2026-05-18", 1.4)],
        )
        third = sync_price_panel(self.layout)
        self.assertEqual(third.decision, Decision.INCREMENTAL)
        self.assertEqual(third.rows_upserted, 2)
        h100 = third.frame.loc[third.frame["series"] == "nvidia-h100__ON_DEMAND", "price"].tolist()
        self.assertEqual(h100, [4.0])
        a100 = third.frame.loc[third.frame["series"] == "nvidia-a100__SPOT"].sort_values("date")
        self.assertEqual(a100["price"].tolist(), [1.0, 1.5])

    def test_non_positive_quotes_are_dropped(self) -> None:
        _write_series(
            self.root,
            "nvidia-h100__ON_DEMAND",
            [("2026-05-17", "a", 0.0), ("2026-05-17", "b", -1.0), ("2026-05-17", "c", 5.0)],
            [("2026-05-17", 5.0)],
        )
        result = sync_price_panel(self.layout)
        self.assertEqual(len(result.frame), 1)
        self.assertEqual(result.frame.iloc[0]["price"], 5.0)
        self.assertEqual(int(result.frame.iloc[0]["n_providers"]), 1)


if __name__ == "__main__":
    unittest.main()
