"""Index series for the local dashboard.

The baseline vector is included as a series. Every other card and plan comes
from the inverse-index matrix. ``index_value * price`` is the baseline price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from icx.config import BILLING_TYPES, find_root, load_baseline
from icx.indices import load_matrix, load_vector
from icx.store import open_store

PLAN_LABELS = {
    "ON_DEMAND": "On demand",
    "SPOT": "Spot",
    "RESERVATION": "Reservation",
    "CUSTOM": "Custom",
}


@dataclass(frozen=True)
class Point:
    date: str
    index: float
    price: float


@dataclass(frozen=True)
class Series:
    id: str
    gpu_id: str
    name: str
    billing: str
    baseline: bool
    points: tuple[Point, ...]

    def catalog_row(self) -> dict:
        last = self.points[-1] if self.points else None
        return {
            "id": self.id,
            "gpu_id": self.gpu_id,
            "name": self.name,
            "billing": self.billing,
            "plan": PLAN_LABELS.get(self.billing, self.billing),
            "baseline": self.baseline,
            "n": len(self.points),
            "last_date": last.date if last else None,
            "last_index": last.index if last else None,
        }

    def history(self) -> dict:
        row = self.catalog_row()
        row["points"] = [
            {
                "date": point.date,
                "index": point.index,
                "price": None if math.isnan(point.price) else point.price,
            }
            for point in self.points
        ]
        return row


def load_series(root=None) -> tuple[dict, dict[str, Series]]:
    project = find_root(root)
    layout = open_store(project)
    baseline = load_baseline(project)
    names = _gpu_names(project / "data" / "gpus.csv")
    vector = load_vector(layout)
    matrix = load_matrix(layout)
    series: dict[str, Series] = {}
    series[baseline.key] = _from_frame(
        baseline.key,
        baseline.gpu_id,
        names.get(baseline.gpu_id, baseline.gpu_id),
        baseline.billing,
        True,
        vector.rename(columns={"index_value": "index"}),
    )
    if not matrix.empty:
        for series_id, group in matrix.groupby("series", sort=False):
            gpu_id = str(group["gpu_id"].iloc[0])
            billing = str(group["billing"].iloc[0])
            series[str(series_id)] = _from_frame(
                str(series_id),
                gpu_id,
                names.get(gpu_id, gpu_id),
                billing,
                False,
                group.rename(columns={"index_value": "index"}),
            )
    meta = {
        "baseline_id": baseline.key,
        "baseline_name": names.get(baseline.gpu_id, baseline.gpu_id),
        "baseline_plan": PLAN_LABELS.get(baseline.billing, baseline.billing),
        "plans": [{"id": billing, "label": PLAN_LABELS[billing]} for billing in BILLING_TYPES],
    }
    return meta, series


def _gpu_names(path) -> dict[str, str]:
    if not path.exists():
        return {}
    catalog = pd.read_csv(path, usecols=["id", "name"])
    return dict(zip(catalog["id"].astype(str), catalog["name"].astype(str)))


def _from_frame(series_id: str, gpu_id: str, name: str, billing: str, baseline: bool, frame: pd.DataFrame) -> Series:
    ordered = frame.loc[:, ["date", "index", "price"]].copy()
    ordered["date"] = ordered["date"].astype(str)
    ordered["index"] = pd.to_numeric(ordered["index"], errors="coerce")
    ordered["price"] = pd.to_numeric(ordered["price"], errors="coerce")
    ordered = ordered.dropna(subset=["date", "index"]).sort_values("date")
    points = tuple(
        Point(date=row.date, index=float(row.index), price=float(row.price) if pd.notna(row.price) else float("nan"))
        for row in ordered.itertuples(index=False)
    )
    return Series(series_id, gpu_id, name, billing, baseline, points)
