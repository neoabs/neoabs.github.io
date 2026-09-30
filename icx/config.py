"""Project paths and the GPU_BASE_PLAN baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BILLING_TYPES = ("ON_DEMAND", "RESERVATION", "SPOT", "CUSTOM")
BASELINE_ENV = "GPU_BASE_PLAN"


@dataclass(frozen=True)
class Baseline:
    """Card and billing plan that price indices are expressed against.

    ``key`` matches the historical-data file stem: ``nvidia-h100__ON_DEMAND``.
    """

    gpu_id: str
    billing: str

    def __post_init__(self) -> None:
        if not self.gpu_id or self.gpu_id != self.gpu_id.strip():
            raise ValueError(f"{BASELINE_ENV} is missing a gpu id.")
        if "/" in self.gpu_id or "\\" in self.gpu_id or self.gpu_id in {".", ".."}:
            raise ValueError(f"{BASELINE_ENV} gpu id {self.gpu_id!r} is not a safe path segment.")
        if self.billing not in BILLING_TYPES:
            allowed = ", ".join(BILLING_TYPES)
            raise ValueError(f"{BASELINE_ENV} billing must be one of {allowed}; got {self.billing!r}.")

    @property
    def key(self) -> str:
        return f"{self.gpu_id}__{self.billing}"

    @classmethod
    def parse(cls, value: str) -> Baseline:
        raw = value.strip()
        if "__" not in raw:
            raise ValueError(
                f"{BASELINE_ENV} must look like nvidia-h100__ON_DEMAND; got {value!r}."
            )
        gpu_id, billing = raw.rsplit("__", 1)
        return cls(gpu_id.strip(), billing.strip())


def find_root(start: Path | None = None) -> Path:
    """Walk parents until a directory containing ``.env`` is found."""
    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / ".env").exists():
            return candidate
    return cursor


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file. Does not modify ``os.environ``."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_baseline(root: Path | None = None) -> Baseline:
    """Read ``GPU_BASE_PLAN`` from the environment, then from ``.env``.

    A process environment value wins over the file, so a one-off override
    still works. The gpu id is checked against ``data/gpus.csv`` when that
    catalog is present.
    """
    project = find_root(root)
    from_file = read_env_file(project / ".env")
    raw = os.environ.get(BASELINE_ENV, "").strip() or from_file.get(BASELINE_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"Missing {BASELINE_ENV}. Set it in {project / '.env'} "
            "(example: nvidia-h100__ON_DEMAND)."
        )
    baseline = Baseline.parse(raw)
    catalog = project / "data" / "gpus.csv"
    if catalog.exists():
        known = _gpu_ids(catalog)
        if baseline.gpu_id not in known:
            raise ValueError(
                f"{BASELINE_ENV} gpu {baseline.gpu_id!r} is not listed in {catalog}."
            )
    return baseline


def _gpu_ids(catalog: Path) -> set[str]:
    ids: set[str] = set()
    lines = catalog.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ids
    for line in lines[1:]:
        if not line.strip():
            continue
        ids.add(line.split(",", 1)[0].strip())
    return ids
