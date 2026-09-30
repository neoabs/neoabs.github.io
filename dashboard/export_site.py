"""Write the static site GitHub Pages serves from docs/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import load_series

STATIC = Path(__file__).resolve().parent / "static"
DOCS = ROOT / "docs"


def main() -> None:
    meta, series = load_series(ROOT)
    rows = [item.catalog_row() for item in series.values()]
    rows.sort(key=lambda row: (not row["baseline"], row["name"].lower(), row["billing"]))
    DOCS.mkdir(exist_ok=True)
    data = DOCS / "data"
    data.mkdir(exist_ok=True)
    for name in ("index.html", "app.css", "app.js"):
        shutil.copyfile(STATIC / name, DOCS / name)
    (DOCS / ".nojekyll").write_text("")
    (data / "catalog.json").write_text(
        json.dumps({"meta": meta, "series": rows}, allow_nan=False),
        encoding="utf-8",
    )
    (data / "history.json").write_text(
        json.dumps({"series": [item.history() for item in series.values()]}, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {DOCS} ({len(series)} series)")


if __name__ == "__main__":
    main()
