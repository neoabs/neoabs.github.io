"""Composite inverse indices from clusters of series.

Distance between two series is one minus the correlation of their adjacent-day
log index changes, using only days where both changes exist. Pairs with fewer
than ``MIN_PAIRS`` overlapping changes are treated as far apart, so a short
overlap cannot glue them together.

Clusters come from average-linkage hierarchical clustering. The number of
clusters is the silhouette maximum between 2 and 8.

A composite is the inverse-volatility weighted mean of its members. On a date
when some members are missing, the weights of the members that do have a price
are scaled to sum to 1. If fewer than three members are present, that date is
missing. A cluster with fewer than three members is not turned into a composite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from icx.indices import MATRIX_ARTIFACT, load_matrix, load_vector
from icx.stability import SCORES_ARTIFACT, load_scores
from icx.store import Decision, Delta, OutputLayout, SyncResult, file_fingerprint

CLUSTER_RECIPE = "clusters-v1"
MEMBERSHIP_ARTIFACT = "04_clusters/membership.csv"
COMPOSITE_ARTIFACT = "04_clusters/composites.csv"
STATS_ARTIFACT = "04_clusters/cluster_stats.csv"
MIN_PAIRS = 20
MIN_MEMBERS = 3
MEMBERSHIP_COLUMNS = ["series", "cluster", "weight", "log_vol", "n_unique_prices"]
COMPOSITE_COLUMNS = ["date", "cluster", "index_value", "n_members"]
STATS_COLUMNS = [
    "cluster",
    "n_members",
    "best_member_log_vol",
    "median_member_log_vol",
    "composite_log_vol",
    "mean_members_present",
]


def sync_clusters(layout: OutputLayout) -> tuple[SyncResult, SyncResult, SyncResult]:
    store = layout.baseline_store
    inputs = {
        MATRIX_ARTIFACT: file_fingerprint(store.artifact_path(MATRIX_ARTIFACT)),
        SCORES_ARTIFACT: file_fingerprint(store.artifact_path(SCORES_ARTIFACT)),
    }
    artifacts = (MEMBERSHIP_ARTIFACT, COMPOSITE_ARTIFACT, STATS_ARTIFACT)
    plans = [store.plan(name, recipe=CLUSTER_RECIPE, inputs=inputs) for name in artifacts]
    if all(plan.decision is Decision.SKIP for plan in plans):
        return tuple(SyncResult(Decision.SKIP, store.read_table(name), 0, 0) for name in artifacts)  # type: ignore[return-value]
    matrix = load_matrix(layout)
    scores = load_scores(layout)
    dates = load_vector(layout)["date"]
    membership, composites, stats = build_clusters(matrix, scores, dates)

    def store_table(name, frame, keys, values):
        return store.sync_table(
            name,
            recipe=CLUSTER_RECIPE,
            inputs=inputs,
            key_columns=keys,
            build_full=lambda frame=frame: frame,
            build_delta=lambda previous, diff, frame=frame, keys=keys, values=values: _replace_changed(
                previous, frame, keys, values
            ),
        )

    return (
        store_table("04_clusters/membership.csv", membership, ["series"], ["cluster", "weight", "log_vol", "n_unique_prices"]),
        store_table("04_clusters/composites.csv", composites, ["date", "cluster"], ["index_value", "n_members"]),
        store_table("04_clusters/cluster_stats.csv", stats, ["cluster"], ["n_members", "best_member_log_vol", "median_member_log_vol", "composite_log_vol", "mean_members_present"]),
    )


def build_clusters(
    matrix: pd.DataFrame,
    scores: pd.DataFrame,
    dates: pd.Series,
    *,
    min_pairs: int = MIN_PAIRS,
    min_members: int = MIN_MEMBERS,
    k_range: range = range(2, 9),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = _eligible_scores(scores)
    if len(eligible) < 4:
        raise RuntimeError("Need at least 4 eligible series to cluster.")
    wide = _wide_index(matrix, dates, eligible["series"])
    distance = _distance(wide, min_pairs)
    labels, k, _score = _choose_k(distance, k_range)
    membership = eligible.copy()
    membership["cluster"] = labels
    weights = 1.0 / membership["log_vol"].clip(lower=1e-4)
    membership["weight"] = weights / weights.groupby(membership["cluster"]).transform("sum")
    membership = membership.loc[:, MEMBERSHIP_COLUMNS].sort_values(["cluster", "series"]).reset_index(drop=True)

    composite_rows = []
    stat_rows = []
    for cluster, members in membership.groupby("cluster"):
        member_rows = members.set_index("series")
        if len(member_rows) < min_members:
            stat_rows.append(_empty_stats(int(cluster), len(member_rows), member_rows["log_vol"]))
            continue
        composite = _composite(wide, member_rows.index, member_rows["weight"], min_members)
        composite["cluster"] = int(cluster)
        composite_rows.append(composite)
        stat_rows.append(
            {
                "cluster": int(cluster),
                "n_members": int(len(member_rows)),
                "best_member_log_vol": float(member_rows["log_vol"].min()),
                "median_member_log_vol": float(member_rows["log_vol"].median()),
                "composite_log_vol": _log_vol(composite["date"], composite["index_value"]),
                "mean_members_present": float(composite["n_members"].mean()),
            }
        )
    composites = (
        pd.concat(composite_rows, ignore_index=True)
        if composite_rows
        else pd.DataFrame(columns=COMPOSITE_COLUMNS)
    )
    if not composites.empty:
        composites = composites.loc[:, COMPOSITE_COLUMNS].sort_values(["cluster", "date"]).reset_index(drop=True)
    stats = pd.DataFrame(stat_rows, columns=STATS_COLUMNS).sort_values("cluster").reset_index(drop=True)
    return membership, composites, stats


def _eligible_scores(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    frame["eligible"] = frame["eligible"].map(lambda value: str(value).lower() in {"1", "true"})
    frame["log_vol"] = pd.to_numeric(frame["log_vol"], errors="coerce")
    frame["n_unique_prices"] = pd.to_numeric(frame["n_unique_prices"], errors="coerce").fillna(0).astype(int)
    eligible = frame.loc[frame["eligible"] & frame["log_vol"].ge(0)].copy()
    return eligible.sort_values("series").reset_index(drop=True)


def _wide_index(matrix: pd.DataFrame, dates: pd.Series, series: pd.Series) -> pd.DataFrame:
    ordered_dates = list(pd.Series(dates).astype(str))
    columns = list(series.astype(str))
    if matrix.empty:
        return pd.DataFrame(index=ordered_dates, columns=columns, dtype=float)
    wide = matrix.pivot(index="date", columns="series", values="index_value")
    wide.index = wide.index.astype(str)
    return wide.reindex(index=ordered_dates, columns=columns).astype(float)


def _distance(wide: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Index(wide.index))
    logged = np.log(wide.where(wide > 0))
    steps = []
    for position in range(1, len(dates)):
        if (dates[position] - dates[position - 1]).days != 1:
            continue
        steps.append(logged.iloc[position] - logged.iloc[position - 1])
    if not steps:
        raise RuntimeError("No adjacent days to measure co-movement.")
    changes = pd.DataFrame(steps)
    correlation = changes.corr(min_periods=min_pairs)
    overlap = changes.notna().astype(int).T.dot(changes.notna().astype(int))
    distance = 1.0 - correlation
    distance = distance.where(overlap >= min_pairs)
    distance = distance.clip(lower=0, upper=2)
    distance = distance.fillna(2.0)
    symmetric = ((distance + distance.T) / 2.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(symmetric, 0.0)
    return pd.DataFrame(symmetric, index=distance.index, columns=distance.columns)


def _choose_k(distance: pd.DataFrame, k_range: range) -> tuple[np.ndarray, int, float]:
    values = distance.to_numpy(dtype=float)
    n = len(values)
    best: tuple[float, int, np.ndarray] | None = None
    for k in k_range:
        if k >= n or k < 2:
            continue
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(values)
        if len(set(labels)) < 2 or np.min(np.bincount(labels)) < 1:
            continue
        try:
            score = float(silhouette_score(values, labels, metric="precomputed"))
        except ValueError:
            continue
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        labels = np.zeros(n, dtype=int)
        return labels, 1, float("nan")
    score, k, labels = best
    return labels, k, score


def _composite(wide: pd.DataFrame, members: pd.Index, weights: pd.Series, min_members: int) -> pd.DataFrame:
    values = wide.loc[:, list(members)]
    weight = weights.reindex(values.columns).astype(float)
    present = values.notna()
    weighted = values.mul(weight, axis=1)
    denominator = present.mul(weight, axis=1).sum(axis=1)
    numerator = weighted.sum(axis=1, min_count=1)
    count = present.sum(axis=1)
    index_value = (numerator / denominator).where(count >= min_members)
    return pd.DataFrame(
        {
            "date": values.index.astype(str),
            "index_value": index_value.to_numpy(),
            "n_members": count.to_numpy(),
        }
    )


def _log_vol(dates: pd.Series, values: pd.Series) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "value": pd.to_numeric(values)})
    frame = frame.dropna().sort_values("date")
    if len(frame) < 3:
        return float("nan")
    adjacent = frame["date"].diff().dt.days.eq(1)
    change = np.log(frame["value"]).diff().loc[adjacent]
    if change.dropna().shape[0] < 2:
        return float("nan")
    return float(change.std(ddof=1))


def _empty_stats(cluster: int, n_members: int, log_vol: pd.Series) -> dict:
    return {
        "cluster": cluster,
        "n_members": n_members,
        "best_member_log_vol": float(log_vol.min()) if len(log_vol) else float("nan"),
        "median_member_log_vol": float(log_vol.median()) if len(log_vol) else float("nan"),
        "composite_log_vol": float("nan"),
        "mean_members_present": float("nan"),
    }


def _replace_changed(previous: pd.DataFrame, current: pd.DataFrame, keys: list[str], values: list[str]) -> Delta:
    from icx.indices import _replace_changed as replace

    return replace(previous, current, keys, values)
