import os
import unittest
from pathlib import Path

import pandas as pd

from icx.config import Baseline, load_baseline
from icx.store import Decision, Delta, OutputLayout, apply_frame_delta, file_fingerprint


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "series", "price"])


def _unexpected(label: str):
    def _call(*_args, **_kwargs):
        raise AssertionError(label)

    return _call


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.id().replace(".", "_"))
        self.project = Path.cwd() / "outputs_test_tmp" / self.root.name
        self.project.mkdir(parents=True, exist_ok=True)
        self.h100 = Baseline.parse("nvidia-h100__ON_DEMAND")
        self.a100 = Baseline.parse("nvidia-a100__SPOT")
        self.layout = OutputLayout(self.project, self.h100)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.project, ignore_errors=True)

    def test_baselines_do_not_overwrite_each_other(self) -> None:
        h100 = self.layout.baseline_store
        a100 = self.layout.store_for(self.a100)
        h100_frame = _frame([("2026-05-17", "base", 3.5)])
        a100_frame = _frame([("2026-05-17", "base", 1.2)])

        h100.sync_table(
            "02_inverse/vector.csv",
            recipe="v1",
            inputs={"src": "aaa"},
            key_columns=["date", "series"],
            build_full=lambda: h100_frame,
            build_delta=_unexpected("delta"),
        )
        a100.sync_table(
            "02_inverse/vector.csv",
            recipe="v1",
            inputs={"src": "bbb"},
            key_columns=["date", "series"],
            build_full=lambda: a100_frame,
            build_delta=_unexpected("delta"),
        )

        stored = h100.read_table("02_inverse/vector.csv")
        assert stored is not None
        self.assertEqual(stored.loc[0, "price"], 3.5)
        self.assertNotEqual(h100.directory, a100.directory)
        self.assertTrue((self.project / "outputs" / "baselines" / "nvidia-h100__ON_DEMAND").is_dir())
        self.assertTrue((self.project / "outputs" / "baselines" / "nvidia-a100__SPOT").is_dir())

    def test_return_to_baseline_skips_unchanged_inputs_and_applies_only_the_diff(self) -> None:
        store = self.layout.baseline_store
        calls = {"full": 0, "delta": 0}

        def build_full() -> pd.DataFrame:
            calls["full"] += 1
            return _frame(
                [
                    ("2026-05-17", "nvidia-a100__ON_DEMAND", 2.0),
                    ("2026-05-18", "nvidia-a100__ON_DEMAND", 2.1),
                ]
            )

        def build_delta(previous: pd.DataFrame, diff) -> Delta:
            calls["delta"] += 1
            self.assertEqual(diff.changed, ("series-b",))
            self.assertEqual(diff.added, ())
            self.assertEqual(len(previous), 2)
            return Delta(
                upsert=_frame([("2026-05-19", "nvidia-a100__ON_DEMAND", 2.2)]),
                drop_keys=pd.DataFrame(columns=["date", "series"]),
            )

        first = store.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"series-a": "1", "series-b": "1"},
            key_columns=["date", "series"],
            build_full=build_full,
            build_delta=build_delta,
        )
        self.assertEqual(first.decision, Decision.FULL)
        self.assertEqual(calls["full"], 1)

        other = self.layout.store_for(self.a100)
        other.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"series-a": "9"},
            key_columns=["date", "series"],
            build_full=lambda: _frame([("2026-05-17", "other", 9.0)]),
            build_delta=_unexpected("other delta"),
        )

        second = store.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"series-a": "1", "series-b": "1"},
            key_columns=["date", "series"],
            build_full=build_full,
            build_delta=build_delta,
        )
        self.assertEqual(second.decision, Decision.SKIP)
        self.assertEqual(calls["full"], 1)
        self.assertEqual(calls["delta"], 0)
        self.assertEqual(len(second.frame), 2)

        third = store.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"series-a": "1", "series-b": "2"},
            key_columns=["date", "series"],
            build_full=build_full,
            build_delta=build_delta,
        )
        self.assertEqual(third.decision, Decision.INCREMENTAL)
        self.assertEqual(calls["full"], 1)
        self.assertEqual(calls["delta"], 1)
        self.assertEqual(third.rows_upserted, 1)
        prices = dict(zip(third.frame["date"], third.frame["price"]))
        self.assertEqual(prices["2026-05-17"], 2.0)
        self.assertEqual(prices["2026-05-18"], 2.1)
        self.assertEqual(prices["2026-05-19"], 2.2)

        other_frame = other.read_table("panel/prices.csv")
        assert other_frame is not None
        self.assertEqual(other_frame.loc[0, "series"], "other")

    def test_recipe_change_rebuilds(self) -> None:
        store = self.layout.baseline_store
        calls = {"full": 0}

        def build_full() -> pd.DataFrame:
            calls["full"] += 1
            return _frame([("2026-05-17", "base", float(calls["full"]))])

        store.sync_table(
            "02_inverse/vector.csv",
            recipe="v1",
            inputs={"src": "a"},
            key_columns=["date", "series"],
            build_full=build_full,
            build_delta=_unexpected("delta"),
        )
        rebuilt = store.sync_table(
            "02_inverse/vector.csv",
            recipe="v2",
            inputs={"src": "a"},
            key_columns=["date", "series"],
            build_full=build_full,
            build_delta=_unexpected("delta"),
        )
        self.assertEqual(rebuilt.decision, Decision.FULL)
        self.assertEqual(calls["full"], 2)
        self.assertEqual(rebuilt.frame.loc[0, "price"], 2.0)

    def test_removed_input_drops_only_those_keys(self) -> None:
        store = self.layout.shared
        store.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"keep": "1", "gone": "1"},
            key_columns=["date", "series"],
            build_full=lambda: _frame(
                [
                    ("2026-05-17", "keep", 1.0),
                    ("2026-05-17", "gone", 2.0),
                ]
            ),
            build_delta=_unexpected("delta"),
        )

        def build_delta(previous: pd.DataFrame, diff) -> Delta:
            self.assertEqual(diff.removed, ("gone",))
            return Delta(
                upsert=pd.DataFrame(columns=["date", "series", "price"]),
                drop_keys=pd.DataFrame([{"date": "2026-05-17", "series": "gone"}]),
            )

        result = store.sync_table(
            "panel/prices.csv",
            recipe="median-v1",
            inputs={"keep": "1"},
            key_columns=["date", "series"],
            build_full=_unexpected("full"),
            build_delta=build_delta,
        )
        self.assertEqual(result.decision, Decision.INCREMENTAL)
        self.assertEqual(result.rows_dropped, 1)
        self.assertEqual(result.frame["series"].tolist(), ["keep"])

    def test_artifact_name_cannot_escape_the_baseline_directory(self) -> None:
        with self.assertRaises(ValueError):
            self.layout.baseline_store.artifact_path("../shared/panel/prices.csv")

    def test_apply_frame_delta_keeps_untouched_rows(self) -> None:
        previous = _frame(
            [
                ("2026-05-17", "a", 1.0),
                ("2026-05-18", "a", 1.5),
            ]
        )
        merged = apply_frame_delta(
            previous,
            Delta(
                upsert=_frame([("2026-05-18", "a", 1.7), ("2026-05-19", "a", 1.8)]),
                drop_keys=pd.DataFrame(columns=["date", "series"]),
            ),
            ["date", "series"],
        )
        self.assertEqual(merged["price"].tolist(), [1.0, 1.7, 1.8])


class BaselineEnvTests(unittest.TestCase):
    def test_load_baseline_reads_env_file(self) -> None:
        root = Path.cwd() / "outputs_test_tmp" / "env_case"
        root.mkdir(parents=True, exist_ok=True)
        try:
            (root / ".env").write_text('GPU_BASE_PLAN=nvidia-h100__ON_DEMAND\n', encoding="utf-8")
            saved = os.environ.pop("GPU_BASE_PLAN", None)
            try:
                baseline = load_baseline(root)
            finally:
                if saved is not None:
                    os.environ["GPU_BASE_PLAN"] = saved
            self.assertEqual(baseline.key, "nvidia-h100__ON_DEMAND")
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_parse_rejects_unknown_billing(self) -> None:
        with self.assertRaises(ValueError):
            Baseline.parse("nvidia-h100__HOURLY")


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_changes_when_file_grows(self) -> None:
        root = Path.cwd() / "outputs_test_tmp" / "hash_case"
        root.mkdir(parents=True, exist_ok=True)
        try:
            path = root / "series.csv"
            path.write_text("date,price\n2026-05-17,1\n", encoding="utf-8")
            first = file_fingerprint(path)
            path.write_text("date,price\n2026-05-17,1\n2026-05-18,2\n", encoding="utf-8")
            second = file_fingerprint(path)
            self.assertNotEqual(first, second)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
