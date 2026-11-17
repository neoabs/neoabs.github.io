"""Silicon Data-style rental price levels.

An offer is one GPU, one rental type, one geography, one interconnect build,
and one availability class. The level for a day is the weighted average of
provider median ``hourly_per_gpu`` prices. The weight is the number of GPUs
that provider lists on that offer that day. Scraped company market share is
not in the inputs yet, so it is not applied.

Storage, independent of ``GPU_BASE_PLAN``::

    outputs/shared/SiliconData/SiliconData_{gpu_id}.csv
    outputs/shared/SiliconData/SiliconData_catalog.csv
    outputs/shared/SiliconData/SiliconData_recipe.json

``CUSTOM`` billing and ``UNAVAILABLE`` quotes are left out. A blank interconnect
is stored as ``UNKNOWN``. ``UNKNOWN`` availability stays its own class.
``TOTAL`` rows pool every kept availability, every interconnect, or both,
using the same provider median and listed-GPU weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from icx.config import find_root

RECIPE = "silicon-data-v2"
DIRECTORY_NAME = "SiliconData"

BILLING_KEPT = ("ON_DEMAND", "RESERVATION", "SPOT")
AVAILABILITY_DROPPED = "UNAVAILABLE"

THEATER = {
    "US": "AMER",
    "CA": "AMER",
    "BR": "AMER",
    "KN": "AMER",
    "GB": "EMEA",
    "FR": "EMEA",
    "DE": "EMEA",
    "FI": "EMEA",
    "NL": "EMEA",
    "CH": "EMEA",
    "SE": "EMEA",
    "LU": "EMEA",
    "PL": "EMEA",
    "IT": "EMEA",
    "LT": "EMEA",
    "RS": "EMEA",
    "UA": "EMEA",
    "AE": "EMEA",
    "IN": "APAC",
    "SG": "APAC",
    "JP": "APAC",
    "HK": "APAC",
    "CN": "APAC",
}

QUOTE_COLUMNS = [
    "date",
    "provider_id",
    "provider_country",
    "gpu_model",
    "billing_type",
    "hourly_per_gpu",
    "gpu_count",
    "availability",
    "interconnect",
]

INDEX_COLUMNS = [
    "date",
    "gpu_id",
    "billing",
    "geography_level",
    "geography",
    "interconnect",
    "availability",
    "index_value",
    "n_providers",
    "n_quotes",
    "listed_gpus",
]


def silicon_directory(root: Path | None = None) -> Path:
    return find_root(root) / "outputs" / "shared" / DIRECTORY_NAME


def build_silicon_index(quotes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per offer per day. ``index_value`` is USD per GPU-hour."""
    frame = _prepare_quotes(quotes)
    if frame.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    expanded = _with_geographies(frame)
    providers = _provider_prices(_with_totals(expanded))
    return _aggregate(providers)


def write_silicon_index(quotes: pd.DataFrame, root: Path | None = None) -> Path:
    """Write per-GPU files plus a catalog. Returns the output directory."""
    index = build_silicon_index(quotes)
    directory = silicon_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("SiliconData_*.csv"):
        stale.unlink()
    catalog_rows = []
    for gpu_id, group in index.groupby("gpu_id", sort=True):
        path = directory / f"SiliconData_{gpu_id}.csv"
        group.to_csv(path, index=False)
        for keys, days in group.groupby(
            ["billing", "geography_level", "geography", "interconnect", "availability"],
            sort=False,
        ):
            catalog_rows.append(
                {
                    "gpu_id": gpu_id,
                    "billing": keys[0],
                    "geography_level": keys[1],
                    "geography": keys[2],
                    "interconnect": keys[3],
                    "availability": keys[4],
                    "n_days": int(len(days)),
                    "first_date": days["date"].min(),
                    "last_date": days["date"].max(),
                    "file": path.name,
                }
            )
    catalog = pd.DataFrame(catalog_rows)
    if catalog.empty:
        catalog = pd.DataFrame(
            columns=[
                "gpu_id",
                "billing",
                "geography_level",
                "geography",
                "interconnect",
                "availability",
                "n_days",
                "first_date",
                "last_date",
                "file",
            ]
        )
    catalog.to_csv(directory / "SiliconData_catalog.csv", index=False)
    recipe = {
        "recipe": RECIPE,
        "unit": "USD per GPU-hour",
        "weight": "listed gpu_count within the offer and day",
        "company_share": "not applied; no scraped share file yet",
        "dropped_billing": ["CUSTOM"],
        "dropped_availability": [AVAILABILITY_DROPPED],
        "blank_interconnect": "UNKNOWN",
        "geographies": ["country", "theater", "global"],
        "totals": {
            "availability": "TOTAL pools every kept availability for that interconnect",
            "interconnect": "TOTAL pools every interconnect for that availability",
            "both": "TOTAL availability and TOTAL interconnect pool the whole offer",
        },
        "theaters": {"AMER": ["US", "CA", "BR", "KN"], "EMEA": sorted(c for c, t in THEATER.items() if t == "EMEA"), "APAC": sorted(c for c, t in THEATER.items() if t == "APAC")},
        "n_rows": int(len(index)),
        "n_gpus": int(index["gpu_id"].nunique()) if not index.empty else 0,
    }
    (directory / "SiliconData_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return directory


def load_quotes(root: Path | None = None) -> pd.DataFrame:
    path = find_root(root) / "data" / "historical_providers.csv"
    return pd.read_csv(path, usecols=QUOTE_COLUMNS)


def _prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    frame = quotes.loc[:, QUOTE_COLUMNS].copy()
    frame["date"] = frame["date"].astype(str)
    frame["hourly_per_gpu"] = pd.to_numeric(frame["hourly_per_gpu"], errors="coerce")
    frame["gpu_count"] = pd.to_numeric(frame["gpu_count"], errors="coerce")
    frame = frame[frame["billing_type"].isin(BILLING_KEPT)]
    frame = frame[frame["availability"] != AVAILABILITY_DROPPED]
    frame = frame[frame["hourly_per_gpu"] > 0]
    frame = frame[frame["gpu_count"] > 0]
    frame = frame[frame["provider_id"].notna() & frame["provider_country"].notna()]
    frame["interconnect"] = frame["interconnect"].fillna("UNKNOWN").astype(str).str.strip()
    frame.loc[frame["interconnect"].isin({"", "nan", "None"}), "interconnect"] = "UNKNOWN"
    frame["availability"] = frame["availability"].fillna("UNKNOWN").astype(str).str.strip()
    frame.loc[frame["availability"].isin({"", "nan", "None"}), "availability"] = "UNKNOWN"
    unknown = sorted(set(frame["provider_country"]) - set(THEATER))
    if unknown:
        raise ValueError(f"No theater for provider countries: {', '.join(unknown)}")
    return frame


def _with_geographies(frame: pd.DataFrame) -> pd.DataFrame:
    country = frame.assign(geography_level="country", geography=frame["provider_country"])
    theater = frame.assign(geography_level="theater", geography=frame["provider_country"].map(THEATER))
    global_ = frame.assign(geography_level="global", geography="GLOBAL")
    return pd.concat([country, theater, global_], ignore_index=True)


def _with_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pooled availability, pooled interconnect, and the pool of both."""
    by_availability = frame.assign(availability="TOTAL")
    by_interconnect = frame.assign(interconnect="TOTAL")
    both = frame.assign(availability="TOTAL", interconnect="TOTAL")
    return pd.concat([frame, by_availability, by_interconnect, both], ignore_index=True)


def _provider_prices(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "date",
        "gpu_model",
        "billing_type",
        "geography_level",
        "geography",
        "interconnect",
        "availability",
        "provider_id",
    ]
    return frame.groupby(keys, as_index=False, sort=False).agg(
        price=("hourly_per_gpu", "median"),
        listed=("gpu_count", "sum"),
        n_quotes=("hourly_per_gpu", "size"),
    )


def _aggregate(providers: pd.DataFrame) -> pd.DataFrame:
    providers = providers.copy()
    providers["weighted"] = providers["price"] * providers["listed"]
    keys = [
        "date",
        "gpu_model",
        "billing_type",
        "geography_level",
        "geography",
        "interconnect",
        "availability",
    ]
    grouped = providers.groupby(keys, as_index=False, sort=False).agg(
        weighted=("weighted", "sum"),
        listed_gpus=("listed", "sum"),
        n_providers=("provider_id", "nunique"),
        n_quotes=("n_quotes", "sum"),
    )
    grouped["index_value"] = (grouped["weighted"] / grouped["listed_gpus"]).round(6)
    grouped = grouped.rename(columns={"gpu_model": "gpu_id", "billing_type": "billing"})
    grouped = grouped[INDEX_COLUMNS].sort_values(
        ["gpu_id", "date", "billing", "geography_level", "geography", "interconnect", "availability"]
    )
    return grouped.reset_index(drop=True)
