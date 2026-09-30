"""How little each inverse index moves relative to the baseline.

Two differences, both on dates where the index exists:

* absolute: mean |i - median(i)|, and the mean absolute day-to-day change
* relative: those quantities divided by the level of i, and the standard
  deviation of day-to-day log changes

Day-to-day changes use adjacent calendar days only. A series is ranked only
when it has at least ``MIN_OBSERVATIONS`` index values and covers at least
``MIN_COVERAGE`` of the baseline dates. A short, quiet series would otherwise
look more stable than a series that was quoted for the whole sample.
Same-card plans and other cards are ranked separately, because a missing
H100 on-demand price may arrive together with the other H100 plans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icx.indices import MATRIX_ARTIFACT, VECTOR_ARTIFACT, load_matrix, load_vector
from icx.store import Delta, OutputLayout, SyncResult, file_fingerprint

STABILITY_RECIPE = "stability-v3"
MIN_COVERAGE = 0.75
SCORES_ARTIFACT = "03_stability/scores.csv"
MIN_OBSERVATIONS = 30
SCORE_COLUMNS = [
    "series",
    "gpu_id",
    "billing",
    "group",
    "n",
    "coverage",
    "median_index",
    "abs_dev",
    "rel_dev",
    "mean_abs_change",
    "mean_rel_change",
    "log_vol",
    "n_unique_prices",
    "eligible",
]


def sync_stability(layout: OutputLayout) -> SyncResult:
    store = layout.baseline_store
    inputs = {
        VECTOR_ARTIFACT: file_fingerprint(store.artifact_path(VECTOR_ARTIFACT)),
        MATRIX_ARTIFACT: file_fingerprint(store.artifact_path(MATRIX_ARTIFACT)),
    }
    vector = load_vector(layout)
    matrix = load_matrix(layout)
    n_dates = int(vector["date"].nunique())
    baseline_gpu = layout.baseline.gpu_id
    return store.sync_table(
        SCORES_ARTIFACT,
        recipe=STABILITY_RECIPE,
        inputs=inputs,
        key_columns=["series"],
        build_full=lambda: score_matrix(matrix, n_dates, baseline_gpu),
        build_delta=lambda previous, diff: _changed_scores(previous, score_matrix(matrix, n_dates, baseline_gpu)),
    )


def load_scores(layout: OutputLayout) -> pd.DataFrame:
    frame = layout.baseline_store.read_table(SCORES_ARTIFACT)
    if frame is None:
        raise FileNotFoundError("Stability scores are missing. Run 03_stability.ipynb first.")
    return frame


def score_matrix(matrix: pd.DataFrame, n_dates: int, baseline_gpu: str) -> pd.DataFrame:
    if n_dates <= 0:
        raise ValueError("The baseline vector has no dates.")
    rows = []
    if not matrix.empty:
        for series, group in matrix.groupby("series", sort=True):
            rows.append(_score_series(group, n_dates, baseline_gpu))
    if not rows:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    scores = pd.DataFrame(rows)
    return scores.sort_values(["eligible", "log_vol", "rel_dev"], ascending=[False, True, True]).reset_index(drop=True)


def ranked(scores: pd.DataFrame, group: str) -> pd.DataFrame:
    eligible = scores.loc[scores["group"].eq(group) & scores["eligible"].astype(bool)].copy()
    return eligible.sort_values(["log_vol", "rel_dev", "series"]).reset_index(drop=True)


def _score_series(group: pd.DataFrame, n_dates: int, baseline_gpu: str) -> dict:
    ordered = group.loc[:, ["date", "index_value", "price", "gpu_id", "billing"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered["index_value"] = pd.to_numeric(ordered["index_value"], errors="coerce")
    ordered = ordered.dropna(subset=["index_value"]).sort_values("date").drop_duplicates("date")
    values = ordered["index_value"]
    median = float(values.median()) if len(values) else np.nan
    abs_dev = float((values - median).abs().mean()) if len(values) else np.nan
    rel_dev = float(abs_dev / median) if median and np.isfinite(median) else np.nan
    previous_date = ordered["date"].shift()
    previous_value = values.shift()
    adjacent = ordered.loc[(ordered["date"] - previous_date).dt.days.eq(1)].copy()
    adjacent_previous = previous_value.loc[adjacent.index]
    if adjacent.empty:
        mean_abs_change = np.nan
        mean_rel_change = np.nan
        log_vol = np.nan
    else:
        change = adjacent["index_value"] - adjacent_previous
        mean_abs_change = float(change.abs().mean())
        mean_rel_change = float((change.abs() / adjacent_previous).mean())
        log_change = np.log(adjacent["index_value"] / adjacent_previous)
        log_vol = float(log_change.std(ddof=1)) if len(log_change) >= 2 else np.nan
    n = int(len(ordered))
    return {
        "series": str(group["series"].iloc[0]),
        "gpu_id": str(ordered["gpu_id"].iloc[0]) if n else str(group["gpu_id"].iloc[0]),
        "billing": str(ordered["billing"].iloc[0]) if n else str(group["billing"].iloc[0]),
        "group": "same_card" if str(group["gpu_id"].iloc[0]) == baseline_gpu else "other_card",
        "n": n,
        "coverage": n / n_dates,
        "median_index": median,
        "abs_dev": abs_dev,
        "rel_dev": rel_dev,
        "mean_abs_change": mean_abs_change,
        "mean_rel_change": mean_rel_change,
        "log_vol": log_vol,
        "n_unique_prices": int(pd.to_numeric(ordered["price"], errors="coerce").round(6).nunique()) if n else 0,
        "eligible": bool(n >= MIN_OBSERVATIONS and (n / n_dates) >= MIN_COVERAGE and np.isfinite(log_vol)),
    }


def _changed_scores(previous: pd.DataFrame, current: pd.DataFrame) -> Delta:
    from icx.indices import _replace_changed

    return _replace_changed(
        previous,
        current,
        ["series"],
        [
            "n",
            "coverage",
            "median_index",
            "abs_dev",
            "rel_dev",
            "mean_abs_change",
            "mean_rel_change",
            "log_vol",
            "n_unique_prices",
        ],
    )
