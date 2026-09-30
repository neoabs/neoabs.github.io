"""Local dashboard for inverse-index history.

Run from the project folder:

    .venv/bin/python dashboard/server.py

Then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import load_series

HOST = "127.0.0.1"
PORT = 8765
STATIC = Path(__file__).resolve().parent / "static"
META, SERIES = load_series()


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.css":
            self._send_file(STATIC / "app.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._send_file(STATIC / "app.js", "text/javascript; charset=utf-8")
            return
        if parsed.path == "/data/catalog.json":
            rows = [series.catalog_row() for series in SERIES.values()]
            rows.sort(key=lambda row: (not row["baseline"], row["name"].lower(), row["billing"]))
            self._send_json(200, {"meta": META, "series": rows})
            return
        if parsed.path == "/data/history.json":
            self._send_json(200, {"series": [series.history() for series in SERIES.values()]})
            return
        if parsed.path.startswith("/static/"):
            relative = unquote(parsed.path.removeprefix("/static/"))
            path = (STATIC / relative).resolve()
            if STATIC.resolve() not in path.parents and path != STATIC.resolve():
                self._send_json(404, {"error": "Not found"})
                return
            content_type = "text/css; charset=utf-8" if path.suffix == ".css" else "text/javascript; charset=utf-8"
            self._send_file(path, content_type)
            return
        if parsed.path == "/api/catalog":
            rows = [series.catalog_row() for series in SERIES.values()]
            rows.sort(key=lambda row: (not row["baseline"], row["name"].lower(), row["billing"]))
            self._send_json(200, {"meta": META, "series": rows})
            return
        if parsed.path == "/api/history":
            requested = parse_qs(parsed.query).get("series", [""])[0]
            ids = [item for item in requested.split(",") if item]
            unknown = [item for item in ids if item not in SERIES]
            if unknown:
                self._send_json(404, {"error": "Unknown series", "ids": unknown})
                return
            self._send_json(200, {"meta": META, "series": [SERIES[item].history() for item in ids]})
            return
        self._send_json(404, {"error": "Not found"})

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json(404, {"error": "Not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Index dashboard: http://{HOST}:{PORT}/", flush=True)
    print(f"Baseline: {META['baseline_name']} · {META['baseline_plan']} ({len(SERIES)} series)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
