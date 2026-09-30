"""Forecasts of the baseline price using data available at that date.

Each method may use other series' prices on the same day. It may not use the
baseline price on that day. Parameters are estimated only on earlier dates.

The first ``MIN_HISTORY`` baseline dates are the fitting prefix. Forecasts
start on the next date. Published forecasts are written under
``methods/<method_id>/`` so the comparison notebook can pick up a new method
by reading that folder, without a change to its code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from icx.clusters import build_clusters
from icx.indices import MATRIX_ARTIFACT, VECTOR_ARTIFACT, load_matrix, load_vector
from icx.stability import score_matrix
from icx.store import Decision, OutputLayout, SyncResult, file_fingerprint

MIN_HISTORY = 40
FORECAST_COLUMNS = ["date", "y_true", "y_pred"]


@dataclass(frozen=True)
class Method:
    method_id: str
    stage: str
    label: str
    description: str
    recipe: str


METHODS: tuple[Method, ...] = (
    Method(
        "03_fixed_price",
        "03",
        "Fixed-price index",
        "Median past inverse index of the lowest-volatility series, times that series' price today. "
        "The lowest-volatility series in this sample have a price that never moves, so the forecast is a level.",
        "fixed-price-v1",
    ),
    Method(
        "03_moving_price",
        "03",
        "Moving-price index",
        "Same conversion, restricted to series whose price took at least five distinct values on the fitting prefix.",
        "moving-price-v1",
    ),
    Method(
        "04_cluster_basket",
        "04",
        "Cluster basket",
        "Clusters and inverse-volatility weights are fit on the prefix only. "
        "The basket with the lowest prefix volatility is applied forward, dropping members that are missing that day.",
        "cluster-basket-v1",
    ),
    Method(
        "05_carry_forward",
        "05",
        "Carry forward",
        "Yesterday's baseline price.",
        "carry-forward-v1",
    ),
    Method(
        "05_ridge",
        "05",
        "Ridge on log prices",
        "Expanding ridge of the log baseline price on the log prices of other series. Features are contemporaneous; the fit uses only earlier days.",
        "ridge-v1",
    ),
)


def sync_forecasts(layout: OutputLayout) -> list[tuple[Method, SyncResult]]:
    store = layout.baseline_store
    inputs = {
        VECTOR_ARTIFACT: file_fingerprint(store.artifact_path(VECTOR_ARTIFACT)),
        MATRIX_ARTIFACT: file_fingerprint(store.artifact_path(MATRIX_ARTIFACT)),
    }
    vector = load_vector(layout)
    matrix = load_matrix(layout)
    dates, baseline_price, other_price, other_index = _panels(vector, matrix)
    builders = {
        "03_fixed_price": lambda: fixed_price_forecast(dates, baseline_price, other_price, other_index, moving=False),
        "03_moving_price": lambda: fixed_price_forecast(dates, baseline_price, other_price, other_index, moving=True),
        "04_cluster_basket": lambda: cluster_forecast(dates, baseline_price, other_price, other_index),
        "05_carry_forward": lambda: carry_forward_forecast(dates, baseline_price),
        "05_ridge": lambda: ridge_forecast(dates, baseline_price, other_price),
    }
    results = []
    for method in METHODS:
        name = f"methods/{method.method_id}/predictions.csv"
        plan = store.plan(name, recipe=method.recipe, inputs=inputs)
        if plan.decision is Decision.SKIP:
            _write_meta(store.artifact_path(f"methods/{method.method_id}/meta.json"), method)
            results.append((method, SyncResult(Decision.SKIP, store.read_table(name), 0, 0)))
            continue
        frame = builders[method.method_id]()
        result = store.sync_table(
            name,
            recipe=method.recipe,
            inputs=inputs,
            key_columns=["date"],
            build_full=lambda frame=frame: frame,
            build_delta=lambda previous, diff, frame=frame: _replace_predictions(previous, frame),
        )
        _write_meta(store.artifact_path(f"methods/{method.method_id}/meta.json"), method)
        results.append((method, result))
    return results


def load_published_methods(layout: OutputLayout) -> list[tuple[dict, pd.DataFrame]]:
    """Every method directory under the active baseline, whatever wrote it."""
    root = layout.baseline_store.directory / "methods"
    if not root.exists():
        return []
    published = []
    for meta_path in sorted(root.glob("*/meta.json")):
        predictions = meta_path.parent / "predictions.csv"
        if not predictions.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(predictions)
        published.append((meta, frame))
    return published


def error_table(published: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for meta, frame in published:
        scored = _numeric(frame)
        error = scored["y_pred"] - scored["y_true"]
        rows.append(
            {
                "method_id": meta["method_id"],
                "stage": meta.get("stage", ""),
                "label": meta.get("label", meta["method_id"]),
                "n": int(len(scored)),
                "mae": float(error.abs().mean()) if len(scored) else np.nan,
                "rmse": float(np.sqrt((error ** 2).mean())) if len(scored) else np.nan,
                "mape": float((error.abs() / scored["y_true"]).mean()) if len(scored) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["mae", "method_id"]).reset_index(drop=True)


def carry_forward_forecast(dates: list[str], baseline_price: pd.Series) -> pd.DataFrame:
    price = baseline_price.astype(float)
    predicted = price.shift(1)
    return _forecast_frame(dates, price, predicted)


def fixed_price_forecast(
    dates: list[str],
    baseline_price: pd.Series,
    other_price: pd.DataFrame,
    other_index: pd.DataFrame,
    *,
    moving: bool,
) -> pd.DataFrame:
    prefix = dates[:MIN_HISTORY]
    chosen = _choose_series(prefix, other_price, other_index, moving=moving)
    predicted = pd.Series(np.nan, index=dates, dtype=float)
    if chosen is None:
        return _forecast_frame(dates, baseline_price, predicted)
    history = other_index[chosen]
    quote = other_price[chosen]
    for position in range(MIN_HISTORY, len(dates)):
        past = history.iloc[:position].dropna()
        today = quote.iloc[position]
        if past.empty or pd.isna(today):
            continue
        predicted.iloc[position] = float(past.median() * today)
    return _forecast_frame(dates, baseline_price, predicted)


def cluster_forecast(
    dates: list[str],
    baseline_price: pd.Series,
    other_price: pd.DataFrame,
    other_index: pd.DataFrame,
) -> pd.DataFrame:
    predicted = pd.Series(np.nan, index=dates, dtype=float)
    prefix = dates[:MIN_HISTORY]
    membership = _prefix_membership(prefix, other_price, other_index)
    if membership is None:
        return _forecast_frame(dates, baseline_price, predicted)
    members = membership["series"].tolist()
    weights = membership.set_index("series")["weight"].astype(float)
    for position in range(MIN_HISTORY, len(dates)):
        past = other_index.iloc[:position]
        level = past[members].median(skipna=True)
        today = other_price.iloc[position][members]
        usable = level.notna() & today.notna() & weights.reindex(members).notna()
        if int(usable.sum()) < 3:
            continue
        weight = weights.reindex(members).loc[usable]
        weight = weight / weight.sum()
        predicted.iloc[position] = float((weight * level.loc[usable] * today.loc[usable]).sum())
    return _forecast_frame(dates, baseline_price, predicted)


def ridge_forecast(dates: list[str], baseline_price: pd.Series, other_price: pd.DataFrame) -> pd.DataFrame:
    predicted = pd.Series(np.nan, index=dates, dtype=float)
    features = _ridge_features(dates[:MIN_HISTORY], other_price)
    if not features:
        return _forecast_frame(dates, baseline_price, predicted)
    design = np.log(other_price[features].astype(float))
    target = np.log(baseline_price.astype(float))
    for position in range(MIN_HISTORY, len(dates)):
        train = design.iloc[:position]
        y = target.iloc[:position]
        complete = train.notna().all(axis=1) & y.notna()
        today = design.iloc[position]
        if int(complete.sum()) < 10 or today.isna().any():
            continue
        x_train = train.loc[complete]
        y_train = y.loc[complete]
        x_center = x_train.mean()
        x_scale = x_train.std(ddof=0).replace(0, 1.0)
        y_center = float(y_train.mean())
        y_scale = float(y_train.std(ddof=0)) or 1.0
        model = Ridge(alpha=1.0)
        model.fit(((x_train - x_center) / x_scale).to_numpy(), ((y_train - y_center) / y_scale).to_numpy())
        today_scaled = ((today - x_center) / x_scale).to_numpy(dtype=float).reshape(1, -1)
        predicted.iloc[position] = float(np.exp(model.predict(today_scaled)[0] * y_scale + y_center))
    return _forecast_frame(dates, baseline_price, predicted)


def _panels(vector: pd.DataFrame, matrix: pd.DataFrame) -> tuple[list[str], pd.Series, pd.DataFrame, pd.DataFrame]:
    ordered = vector.sort_values("date")
    dates = ordered["date"].astype(str).tolist()
    baseline_price = pd.Series(ordered["price"].to_numpy(dtype=float), index=dates)
    if matrix.empty:
        empty = pd.DataFrame(index=dates)
        return dates, baseline_price, empty, empty
    other_price = matrix.pivot(index="date", columns="series", values="price").reindex(dates)
    other_index = matrix.pivot(index="date", columns="series", values="index_value").reindex(dates)
    return dates, baseline_price, other_price, other_index


def _choose_series(prefix: list[str], other_price: pd.DataFrame, other_index: pd.DataFrame, *, moving: bool) -> str | None:
    scores = _prefix_scores(prefix, other_price, other_index)
    if scores.empty:
        return None
    scores = scores.loc[scores["eligible"].map(lambda value: str(value).lower() in {"1", "true"})]
    if scores.empty:
        return None
    if moving:
        scores = scores.loc[scores["n_unique_prices"] >= 5]
    if scores.empty:
        return None
    return str(scores.sort_values(["log_vol", "series"]).iloc[0]["series"])


def _prefix_scores(prefix: list[str], other_price: pd.DataFrame, other_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series in other_index.columns:
        index_values = other_index.loc[prefix, series]
        prices = other_price.loc[prefix, series]
        paired = pd.DataFrame({"date": prefix, "index_value": index_values.to_numpy(), "price": prices.to_numpy()})
        paired = paired.dropna()
        if paired.empty:
            continue
        paired["series"] = series
        paired["gpu_id"] = series.split("__", 1)[0]
        paired["billing"] = series.split("__", 1)[1]
        rows.append(paired)
    if not rows:
        return pd.DataFrame()
    return score_matrix(pd.concat(rows, ignore_index=True), n_dates=len(prefix), baseline_gpu="")


def _prefix_membership(prefix: list[str], other_price: pd.DataFrame, other_index: pd.DataFrame) -> pd.DataFrame | None:
    scores = _prefix_scores(prefix, other_price, other_index)
    eligible = scores.loc[scores["eligible"].map(lambda value: str(value).lower() in {"1", "true"})]
    if len(eligible) < 4:
        return None
    records = []
    for series in eligible["series"]:
        for date in prefix:
            index_value = other_index.at[date, series]
            price = other_price.at[date, series]
            if pd.isna(index_value) or pd.isna(price):
                continue
            gpu_id, billing = series.split("__", 1)
            records.append(
                {
                    "date": date,
                    "series": series,
                    "gpu_id": gpu_id,
                    "billing": billing,
                    "price": price,
                    "base_price": index_value * price,
                    "index_value": index_value,
                }
            )
    if not records:
        return None
    try:
        membership, _composites, stats = build_clusters(
            pd.DataFrame(records),
            scores,
            prefix,
            min_pairs=10,
        )
    except RuntimeError:
        return None
    usable = stats.dropna(subset=["composite_log_vol"]).sort_values(["composite_log_vol", "cluster"])
    if usable.empty:
        return None
    chosen = int(usable.iloc[0]["cluster"])
    return membership.loc[membership["cluster"].eq(chosen)].copy()


def _ridge_features(prefix: list[str], other_price: pd.DataFrame) -> list[str]:
    if other_price.empty:
        return []
    block = other_price.loc[prefix]
    complete = [column for column in block.columns if block[column].notna().all() and (block[column] > 0).all()]
    ranked = sorted(complete, key=lambda column: (-block[column].nunique(), column))
    return ranked[:15]


def _forecast_frame(dates: list[str], baseline_price: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": dates,
            "y_true": baseline_price.reindex(dates).to_numpy(dtype=float),
            "y_pred": predicted.reindex(dates).to_numpy(dtype=float),
        }
    )
    frame = frame.iloc[MIN_HISTORY:]
    frame = frame.dropna(subset=["y_pred"]).reset_index(drop=True)
    return frame.loc[:, FORECAST_COLUMNS]


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["y_true"] = pd.to_numeric(scored["y_true"])
    scored["y_pred"] = pd.to_numeric(scored["y_pred"])
    return scored.dropna(subset=["y_true", "y_pred"])


def _replace_predictions(previous: pd.DataFrame, current: pd.DataFrame):
    from icx.indices import _replace_changed
    from icx.store import Delta

    if previous is None or previous.empty:
        return Delta(upsert=current, drop_keys=pd.DataFrame(columns=["date"]))
    return _replace_changed(previous, current, ["date"], ["y_true", "y_pred"])


def _write_meta(path: Path, method: Method) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method_id": method.method_id,
        "stage": method.stage,
        "label": method.label,
        "description": method.description,
        "recipe": method.recipe,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
