"""Daily price panel from the historical provider quotes.

The working price on a date is the median across providers of each provider's
median ``hourly_per_gpu``. Each provider has one vote, however many listings
it posted that day. Non-positive and missing quotes are ignored.

``avg_price`` from the summary files is stored beside that median so the two
can be compared. It is not the price the indices use.

The panel does not depend on ``GPU_BASE_PLAN``. It lives in ``outputs/shared``
and is rebuilt only for series whose source files changed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icx.store import Delta, InputDiff, OutputLayout, SyncResult, fingerprint_paths

PANEL_RECIPE = "provider-median-hourly-per-gpu-v1"
PANEL_ARTIFACT = "panel/median_hourly_per_gpu.csv"
PANEL_COLUMNS = ["date", "series", "gpu_id", "billing", "price", "n_providers", "avg_price"]
PANEL_KEYS = ["date", "series"]


def sync_price_panel(layout: OutputLayout) -> SyncResult:
    parts = layout.root / "data" / "parts"
    files = sorted(parts.glob("*.providers.csv")) + sorted(parts.glob("*.summary.csv"))
    inputs = fingerprint_paths(files, layout.root)
    return layout.shared.sync_table(
        PANEL_ARTIFACT,
        recipe=PANEL_RECIPE,
        inputs=inputs,
        key_columns=PANEL_KEYS,
        build_full=lambda: build_panel(layout.root),
        build_delta=lambda previous, diff: update_panel(layout.root, previous, diff),
    )


def load_price_panel(layout: OutputLayout) -> pd.DataFrame:
    frame = layout.shared.read_table(PANEL_ARTIFACT)
    if frame is None:
        raise FileNotFoundError(
            f"Price panel is missing. Run sync_price_panel first ({PANEL_ARTIFACT})."
        )
    return frame


def build_panel(root: Path) -> pd.DataFrame:
    parts = root / "data" / "parts"
    frames = []
    for path in sorted(parts.glob("*.providers.csv")):
        series = _series_name(path.name)
        frame = _series_frame(root, series)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(PANEL_KEYS).reset_index(drop=True)


def update_panel(root: Path, previous: pd.DataFrame, diff: InputDiff) -> Delta:
    affected = sorted({name for key in diff.affected if (name := _series_from_key(key))})
    upserts: list[pd.DataFrame] = []
    drops: list[pd.DataFrame] = []
    for series in affected:
        rows = _series_frame(root, series)
        previous_keys = previous.loc[previous["series"] == series, PANEL_KEYS]
        if rows.empty:
            if not previous_keys.empty:
                drops.append(previous_keys)
            continue
        kept_dates = set(rows["date"])
        removed = previous_keys.loc[~previous_keys["date"].astype(str).isin(kept_dates)]
        if not removed.empty:
            drops.append(removed)
        upserts.append(rows)
    upsert = pd.concat(upserts, ignore_index=True) if upserts else pd.DataFrame(columns=PANEL_COLUMNS)
    drop_keys = pd.concat(drops, ignore_index=True) if drops else pd.DataFrame(columns=PANEL_KEYS)
    return Delta(upsert=upsert, drop_keys=drop_keys)


def _series_frame(root: Path, series: str) -> pd.DataFrame:
    gpu_id, billing = series.rsplit("__", 1)
    parts = root / "data" / "parts"
    daily = _daily_from_providers(parts / f"{series}.providers.csv")
    if daily.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    averages = _average_from_summary(parts / f"{series}.summary.csv")
    merged = daily.merge(averages, on="date", how="left")
    merged.insert(1, "series", series)
    merged.insert(2, "gpu_id", gpu_id)
    merged.insert(3, "billing", billing)
    return merged.loc[:, PANEL_COLUMNS]


def _daily_from_providers(path: Path) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "price", "n_providers"])
    if not path.exists() or path.stat().st_size == 0:
        return empty
    quotes = pd.read_csv(path, usecols=["date", "provider_id", "hourly_per_gpu"])
    if quotes.empty:
        return empty
    quotes["hourly_per_gpu"] = pd.to_numeric(quotes["hourly_per_gpu"], errors="coerce")
    quotes = quotes.dropna(subset=["date", "provider_id", "hourly_per_gpu"])
    quotes = quotes.loc[quotes["hourly_per_gpu"] > 0]
    if quotes.empty:
        return empty
    per_provider = quotes.groupby(["date", "provider_id"], as_index=False)["hourly_per_gpu"].median()
    daily = per_provider.groupby("date", as_index=False).agg(
        price=("hourly_per_gpu", "median"),
        n_providers=("provider_id", "size"),
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily["n_providers"] = daily["n_providers"].astype(int)
    return daily.sort_values("date").reset_index(drop=True)


def _average_from_summary(path: Path) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["date", "avg_price"])
    if not path.exists() or path.stat().st_size == 0:
        return empty
    summary = pd.read_csv(path, usecols=["date", "avg_price"])
    if summary.empty:
        return empty
    summary["avg_price"] = pd.to_numeric(summary["avg_price"], errors="coerce")
    summary["date"] = pd.to_datetime(summary["date"]).dt.strftime("%Y-%m-%d")
    summary = summary.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return summary.loc[:, ["date", "avg_price"]]


def _series_name(filename: str) -> str:
    for suffix in (".providers.csv", ".summary.csv"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    raise ValueError(f"Unrecognised parts file {filename!r}.")


def _series_from_key(key: str) -> str | None:
    try:
        return _series_name(Path(key).name)
    except ValueError:
        return None
