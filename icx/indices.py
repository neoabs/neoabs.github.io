"""Inverse price indices against the baseline card and plan.

``index_value`` is defined so that ``index_value * price = baseline price``
on the same date.

The vector is the baseline price divided by its price on the base date, so
the first row is 1. The base date is the first day of the continuous sample
(the history has one isolated print on 2026-04-01, then a gap, then a daily
run). Dates before that, and dates with no baseline price, are left out.
The vector has no missing values.

The matrix has one row per other series and vector date. A cell is stored
only when both prices exist. Callers that need an explicit missing value
reindex this long table onto the full date × series grid.
"""

from __future__ import annotations

import pandas as pd

from icx.panel import PANEL_ARTIFACT, load_price_panel
from icx.store import Delta, InputDiff, OutputLayout, SyncResult, file_fingerprint

INDEX_RECIPE = "inverse-index-v1"
VECTOR_ARTIFACT = "02_inverse/vector.csv"
MATRIX_ARTIFACT = "02_inverse/matrix.csv"
VECTOR_COLUMNS = ["date", "price", "base_date", "base_price", "index_value"]
MATRIX_COLUMNS = ["date", "series", "gpu_id", "billing", "price", "base_price", "index_value"]
VECTOR_KEYS = ["date"]
MATRIX_KEYS = ["date", "series"]


def sync_inverse_indices(layout: OutputLayout) -> tuple[SyncResult, SyncResult]:
    panel_path = layout.shared.artifact_path(PANEL_ARTIFACT)
    if not panel_path.exists():
        raise FileNotFoundError("Price panel is missing. Run 01_panel.ipynb first.")
    relative = panel_path.resolve().relative_to(layout.root).as_posix()
    inputs = {relative: file_fingerprint(panel_path)}
    panel = load_price_panel(layout)
    baseline_key = layout.baseline.key

    vector_result = layout.baseline_store.sync_table(
        VECTOR_ARTIFACT,
        recipe=INDEX_RECIPE,
        inputs=inputs,
        key_columns=VECTOR_KEYS,
        build_full=lambda: build_vector(panel, baseline_key),
        build_delta=lambda previous, diff: _replace_changed(
            previous,
            build_vector(panel, baseline_key),
            VECTOR_KEYS,
            ["price", "base_price", "index_value"],
        ),
    )
    matrix_result = layout.baseline_store.sync_table(
        MATRIX_ARTIFACT,
        recipe=INDEX_RECIPE,
        inputs=inputs,
        key_columns=MATRIX_KEYS,
        build_full=lambda: build_matrix(panel, baseline_key),
        build_delta=lambda previous, diff: _replace_changed(previous, build_matrix(panel, baseline_key), MATRIX_KEYS, ["price", "base_price", "index_value"]),
    )
    return vector_result, matrix_result


def load_vector(layout: OutputLayout) -> pd.DataFrame:
    frame = layout.baseline_store.read_table(VECTOR_ARTIFACT)
    if frame is None:
        raise FileNotFoundError("Inverse vector is missing. Run 02_inverse_indices.ipynb first.")
    return frame


def load_matrix(layout: OutputLayout) -> pd.DataFrame:
    frame = layout.baseline_store.read_table(MATRIX_ARTIFACT)
    if frame is None:
        raise FileNotFoundError("Inverse matrix is missing. Run 02_inverse_indices.ipynb first.")
    return frame


def build_vector(panel: pd.DataFrame, baseline_key: str) -> pd.DataFrame:
    base_date, base_rows = _baseline_rows(panel, baseline_key)
    base_price = float(base_rows.loc[base_rows["date"] == base_date, "price"].iloc[0])
    vector = base_rows.loc[:, ["date", "price"]].copy()
    vector["base_date"] = base_date
    vector["base_price"] = base_price
    vector["index_value"] = vector["price"] / base_price
    vector = _round_index_frame(vector, ["price", "base_price", "index_value"])
    if vector["index_value"].isna().any() or (vector["index_value"] <= 0).any():
        raise RuntimeError("Baseline vector contains a missing or non-positive index.")
    return vector.loc[:, VECTOR_COLUMNS].sort_values("date").reset_index(drop=True)


def build_matrix(panel: pd.DataFrame, baseline_key: str) -> pd.DataFrame:
    base_date, base_rows = _baseline_rows(panel, baseline_key)
    by_date = base_rows.set_index("date")["price"]
    others = panel.loc[(panel["series"] != baseline_key) & (panel["date"] >= base_date)].copy()
    others = others.loc[others["date"].isin(by_date.index)]
    if others.empty:
        return pd.DataFrame(columns=MATRIX_COLUMNS)
    others["base_price"] = others["date"].map(by_date)
    others = others.dropna(subset=["price", "base_price"])
    others = others.loc[others["price"] > 0]
    others["index_value"] = others["base_price"] / others["price"]
    others = _round_index_frame(others, ["price", "base_price", "index_value"])
    return others.loc[:, MATRIX_COLUMNS].sort_values(MATRIX_KEYS).reset_index(drop=True)


def matrix_with_gaps(panel: pd.DataFrame, vector: pd.DataFrame, matrix: pd.DataFrame, baseline_key: str) -> pd.DataFrame:
    """Wide matrix aligned to vector dates. Missing pairs are NA."""
    base_date = str(vector["base_date"].iloc[0])
    series = sorted(panel.loc[panel["series"] != baseline_key, "series"].unique())
    dates = list(vector["date"])
    if matrix.empty:
        wide = pd.DataFrame(index=dates, columns=series, dtype=float)
    else:
        wide = matrix.pivot(index="date", columns="series", values="index_value")
        wide = wide.reindex(index=dates, columns=series)
    wide.index.name = "date"
    wide.attrs["base_date"] = base_date
    return wide


def continuous_sample_start(dates: pd.Series) -> str:
    """First date of the trailing run of consecutive calendar days."""
    unique = pd.to_datetime(pd.Series(dates).dropna().unique()).sort_values()
    if unique.empty:
        raise ValueError("No dates to place the base date on.")
    start = unique[-1]
    for current, previous in zip(unique[:0:-1], unique[-2::-1]):
        if (current - previous).days == 1:
            start = previous
            continue
        break
    return start.strftime("%Y-%m-%d")


def _baseline_rows(panel: pd.DataFrame, baseline_key: str) -> tuple[str, pd.DataFrame]:
    base = panel.loc[panel["series"] == baseline_key].copy()
    if base.empty:
        raise RuntimeError(f"{baseline_key} has no rows in the price panel.")
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    base["price"] = pd.to_numeric(base["price"], errors="coerce")
    base = base.dropna(subset=["price"])
    base = base.loc[base["price"] > 0]
    if base.empty:
        raise RuntimeError(f"{baseline_key} has no positive prices.")
    start = continuous_sample_start(panel["date"])
    in_sample = base.loc[base["date"] >= start].sort_values("date")
    if in_sample.empty:
        start = continuous_sample_start(base["date"])
        in_sample = base.loc[base["date"] >= start].sort_values("date")
    if start not in set(in_sample["date"]):
        start = str(in_sample["date"].iloc[0])
        in_sample = in_sample.loc[in_sample["date"] >= start]
    return start, in_sample.reset_index(drop=True)


def _round_index_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rounded = frame.copy()
    for column in columns:
        if column == "index_value":
            rounded[column] = rounded[column].round(8)
        else:
            rounded[column] = rounded[column].round(6)
    return rounded


def _replace_changed(previous: pd.DataFrame, current: pd.DataFrame, keys: list[str], values: list[str]) -> Delta:
    """Upsert rows whose values changed and drop keys that disappeared."""
    if previous is None or previous.empty:
        return Delta(upsert=current, drop_keys=pd.DataFrame(columns=keys))
    previous = previous.copy()
    current = current.copy()
    for frame in (previous, current):
        for column in values:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prev_keys = previous.loc[:, keys].astype(str).agg("\0".join, axis=1)
    cur_keys = current.loc[:, keys].astype(str).agg("\0".join, axis=1)
    previous = previous.assign(_key=prev_keys)
    current = current.assign(_key=cur_keys)
    merged = current.merge(previous.loc[:, ["_key", *values]], on="_key", how="left", suffixes=("", "_old"))
    changed = pd.Series(False, index=merged.index)
    for column in values:
        old = pd.to_numeric(merged[f"{column}_old"], errors="coerce")
        new = pd.to_numeric(merged[column], errors="coerce")
        changed = changed | old.isna() | ~new.sub(old).abs().le(1e-8)
    upsert = current.loc[current["_key"].isin(set(merged.loc[changed, "_key"]))].drop(columns="_key")
    removed = previous.loc[~previous["_key"].isin(set(cur_keys)), keys]
    return Delta(upsert=upsert.reset_index(drop=True), drop_keys=removed.reset_index(drop=True))
