"""Silicon Data levels for the dashboard, read from the shared CSV output."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dashboard.data import PLAN_LABELS, _gpu_names
from icx.config import find_root
from icx.silicon_data import DIRECTORY_NAME

AVAILABILITY_LABELS = {
    "AVAILABLE": "Available",
    "WAITLIST": "Waitlist",
    "UNKNOWN": "Unknown",
    "TOTAL": "Total",
}

DEFAULT_ID = "nvidia-h100__ON_DEMAND__global__GLOBAL__SXM__AVAILABLE"


@dataclass(frozen=True)
class SiliconSeries:
    id: str
    gpu_id: str
    name: str
    billing: str
    geography_level: str
    geography: str
    interconnect: str
    availability: str
    points: tuple[tuple[str, float], ...]

    def label(self) -> str:
        availability = AVAILABILITY_LABELS.get(self.availability, self.availability)
        if self.interconnect == "TOTAL":
            interconnect = "Total"
        elif self.interconnect == "UNKNOWN":
            interconnect = "Unknown"
        else:
            interconnect = self.interconnect
        return f"{self.name} · {self.geography} · {interconnect} · {availability}"

    def catalog_row(self) -> dict:
        last = self.points[-1] if self.points else None
        return {
            "id": self.id,
            "gpu_id": self.gpu_id,
            "name": self.name,
            "label": self.label(),
            "billing": self.billing,
            "plan": PLAN_LABELS.get(self.billing, self.billing),
            "geography_level": self.geography_level,
            "geography": self.geography,
            "interconnect": self.interconnect,
            "availability": self.availability,
            "n": len(self.points),
            "last_date": last[0] if last else None,
            "last_index": last[1] if last else None,
        }

    def history(self) -> dict:
        row = self.catalog_row()
        row["points"] = [{"date": date, "index": value} for date, value in self.points]
        return row


def series_id(gpu_id: str, billing: str, level: str, geography: str, interconnect: str, availability: str) -> str:
    return f"{gpu_id}__{billing}__{level}__{geography}__{interconnect}__{availability}"


def load_silicon(root=None) -> tuple[dict, dict[str, SiliconSeries]]:
    project = find_root(root)
    directory = project / "outputs" / "shared" / DIRECTORY_NAME
    names = _gpu_names(project / "data" / "gpus.csv")
    series: dict[str, SiliconSeries] = {}
    if directory.is_dir():
        for path in sorted(directory.glob("SiliconData_*.csv")):
            if path.name == "SiliconData_catalog.csv":
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            keys = ["gpu_id", "billing", "geography_level", "geography", "interconnect", "availability"]
            for values, group in frame.groupby(keys, sort=False):
                gpu_id, billing, level, geography, interconnect, availability = (str(item) for item in values)
                ordered = group.sort_values("date")
                points = tuple(
                    (str(row.date), float(row.index_value))
                    for row in ordered.itertuples(index=False)
                    if pd.notna(row.index_value)
                )
                item = SiliconSeries(
                    id=series_id(gpu_id, billing, level, geography, interconnect, availability),
                    gpu_id=gpu_id,
                    name=names.get(gpu_id, gpu_id),
                    billing=billing,
                    geography_level=level,
                    geography=geography,
                    interconnect=interconnect,
                    availability=availability,
                    points=points,
                )
                series[item.id] = item
    plans = [{"id": billing, "label": PLAN_LABELS[billing]} for billing in ("ON_DEMAND", "RESERVATION", "SPOT")]
    meta = {
        "plans": plans,
        "geographies": [
            {"id": "global", "label": "Global"},
            {"id": "theater", "label": "Theater"},
            {"id": "country", "label": "Country"},
        ],
        "availabilities": [
            {"id": "TOTAL", "label": "Total"},
            {"id": "AVAILABLE", "label": "Available"},
            {"id": "UNKNOWN", "label": "Unknown"},
            {"id": "WAITLIST", "label": "Waitlist"},
        ],
        "interconnects": [
            {"id": "TOTAL", "label": "Total"},
            {"id": "SXM", "label": "SXM"},
            {"id": "SXM4", "label": "SXM4"},
            {"id": "SXM5", "label": "SXM5"},
            {"id": "PCIe", "label": "PCIe"},
            {"id": "NVL", "label": "NVL"},
            {"id": "NVL72", "label": "NVL72"},
            {"id": "OAM", "label": "OAM"},
            {"id": "UNKNOWN", "label": "Unknown"},
        ],
        "default_id": DEFAULT_ID if DEFAULT_ID in series else None,
    }
    return meta, series


def catalog_payload(meta: dict, series: dict[str, SiliconSeries]) -> dict:
    rows = [item.catalog_row() for item in series.values()]
    level_order = {"global": 0, "theater": 1, "country": 2}

    def rank(value: str) -> tuple:
        return (0 if value == "TOTAL" else 1, value)

    rows.sort(key=lambda row: (
        row["name"].lower(),
        row["billing"],
        level_order.get(row["geography_level"], 9),
        row["geography"],
        rank(row["interconnect"]),
        rank(row["availability"]),
    ))
    return {"meta": meta, "series": rows}


def gpu_history_payload(series: dict[str, SiliconSeries], gpu_id: str) -> dict:
    chosen = [item.history() for item in series.values() if item.gpu_id == gpu_id]
    return {"series": chosen}
