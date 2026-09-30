"""Output storage keyed by ``GPU_BASE_PLAN``.

Layout::

    outputs/
      shared/                          # artifacts that do not depend on the baseline
        manifest.json
        panel/prices.csv
      baselines/
        nvidia-h100__ON_DEMAND/        # one directory per GPU_BASE_PLAN value
          manifest.json
          02_inverse/vector.csv
        nvidia-a100__SPOT/
          manifest.json
          ...

Changing ``GPU_BASE_PLAN`` selects another directory, so a previous baseline's
files stay where they are. Each artifact records the recipe and the fingerprint
of the inputs that produced it. On a later run for that same baseline:

* identical inputs → ``Decision.SKIP`` (the stored table is reused)
* changed inputs, same recipe → ``Decision.INCREMENTAL`` (only ``build_delta`` runs)
* missing artifact, or a different recipe → ``Decision.FULL``
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

from icx.config import Baseline, find_root, load_baseline

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


class Decision(str, Enum):
    FULL = "full"
    SKIP = "skip"
    INCREMENTAL = "incremental"


@dataclass(frozen=True)
class InputDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def affected(self) -> tuple[str, ...]:
        return self.added + self.removed + self.changed


@dataclass(frozen=True)
class Delta:
    """Rows to insert-or-replace, and keys to delete, relative to the stored table."""

    upsert: pd.DataFrame
    drop_keys: pd.DataFrame


@dataclass(frozen=True)
class UpdatePlan:
    decision: Decision
    diff: InputDiff
    record: dict | None


@dataclass(frozen=True)
class SyncResult:
    decision: Decision
    frame: pd.DataFrame
    rows_upserted: int
    rows_dropped: int


def diff_inputs(previous: Mapping[str, str], current: Mapping[str, str]) -> InputDiff:
    prev_keys = set(previous)
    cur_keys = set(current)
    added = tuple(sorted(cur_keys - prev_keys))
    removed = tuple(sorted(prev_keys - cur_keys))
    changed = tuple(sorted(key for key in cur_keys & prev_keys if previous[key] != current[key]))
    unchanged = tuple(sorted(key for key in cur_keys & prev_keys if previous[key] == current[key]))
    return InputDiff(added, removed, changed, unchanged)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(paths: Sequence[Path], root: Path) -> dict[str, str]:
    """Map a stable relative path to the sha256 of that file."""
    base = root.resolve()
    fingerprints: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            key = resolved.relative_to(base).as_posix()
        except ValueError:
            key = resolved.as_posix()
        fingerprints[key] = file_fingerprint(resolved)
    return dict(sorted(fingerprints.items()))


def list_saved_baselines(root: Path | None = None) -> list[str]:
    directory = find_root(root) / "outputs" / "baselines"
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())


def open_store(root: Path | None = None) -> OutputLayout:
    project = find_root(root)
    return OutputLayout(project, load_baseline(project))


class OutputLayout:
    """Shared artifacts plus the directory for the active baseline."""

    def __init__(self, root: Path, baseline: Baseline):
        self.root = root.resolve()
        self.baseline = baseline
        self.shared = ArtifactStore(self.root / "outputs" / "shared")
        self.baseline_store = ArtifactStore(
            self.root / "outputs" / "baselines" / baseline.key,
            baseline=baseline.key,
        )

    def store_for(self, baseline: Baseline) -> ArtifactStore:
        return ArtifactStore(
            self.root / "outputs" / "baselines" / baseline.key,
            baseline=baseline.key,
        )


class ArtifactStore:
    def __init__(self, directory: Path, *, baseline: str | None = None):
        self.directory = directory
        self.baseline = baseline

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def artifact_path(self, name: str) -> Path:
        if not name or name.startswith(("/", "\\")) or "\\" in name:
            raise ValueError(f"Artifact name {name!r} must be a relative path.")
        path = (self.directory / name).resolve()
        directory = self.directory.resolve()
        if path != directory and directory not in path.parents:
            raise ValueError(f"Artifact name {name!r} escapes {directory}.")
        return path

    def load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return self._empty_manifest()
        if payload.get("version") != MANIFEST_VERSION:
            return self._empty_manifest()
        if self.baseline is not None and payload.get("baseline") != self.baseline:
            return self._empty_manifest()
        payload.setdefault("artifacts", {})
        return payload

    def plan(self, name: str, *, recipe: str, inputs: Mapping[str, str]) -> UpdatePlan:
        self._check_inputs(inputs)
        manifest = self.load_manifest()
        record = (manifest.get("artifacts") or {}).get(name)
        current = dict(inputs)
        if not isinstance(record, dict) or record.get("recipe") != recipe:
            previous_inputs = record.get("inputs") if isinstance(record, dict) else {}
            if not isinstance(previous_inputs, dict):
                previous_inputs = {}
            return UpdatePlan(Decision.FULL, diff_inputs(previous_inputs, current), record if isinstance(record, dict) else None)
        previous_inputs = record.get("inputs") or {}
        if not isinstance(previous_inputs, dict):
            previous_inputs = {}
        diff = diff_inputs(previous_inputs, current)
        if not diff.affected and self.artifact_path(name).exists():
            return UpdatePlan(Decision.SKIP, diff, record)
        if not self.artifact_path(name).exists():
            return UpdatePlan(Decision.FULL, diff, record)
        return UpdatePlan(Decision.INCREMENTAL, diff, record)

    def read_table(self, name: str) -> pd.DataFrame | None:
        path = self.artifact_path(name)
        if not path.exists():
            return None
        return pd.read_csv(path)

    def sync_table(
        self,
        name: str,
        *,
        recipe: str,
        inputs: Mapping[str, str],
        key_columns: Sequence[str],
        build_full: Callable[[], pd.DataFrame],
        build_delta: Callable[[pd.DataFrame, InputDiff], Delta],
    ) -> SyncResult:
        """Reuse a stored table, or rebuild only the input diff.

        ``build_full`` runs when this baseline has no usable table for ``recipe``.
        ``build_delta`` runs when the recipe matches and some inputs changed. It
        receives the stored table and the input diff, and returns rows to upsert
        plus keys to drop. Unmentioned keys stay as stored.
        """
        if not key_columns:
            raise ValueError("key_columns must name at least one column.")
        plan = self.plan(name, recipe=recipe, inputs=inputs)
        if plan.decision is Decision.SKIP:
            frame = self.read_table(name)
            assert frame is not None
            return SyncResult(plan.decision, frame, 0, 0)
        if plan.decision is Decision.FULL:
            frame = build_full()
            self.write_table(
                name,
                frame,
                recipe=recipe,
                inputs=inputs,
                key_columns=key_columns,
                coverage="full",
                changed_keys=None,
            )
            return SyncResult(plan.decision, self.read_table(name), len(frame), 0)  # type: ignore[arg-type]

        previous = self.read_table(name)
        if previous is None:
            frame = build_full()
            self.write_table(
                name,
                frame,
                recipe=recipe,
                inputs=inputs,
                key_columns=key_columns,
                coverage="full",
                changed_keys=None,
            )
            return SyncResult(Decision.FULL, self.read_table(name), len(frame), 0)  # type: ignore[arg-type]

        delta = build_delta(previous, plan.diff)
        merged = apply_frame_delta(previous, delta, key_columns)
        changed = _key_records(delta.upsert, key_columns) + _key_records(delta.drop_keys, key_columns)
        self.write_table(
            name,
            merged,
            recipe=recipe,
            inputs=inputs,
            key_columns=key_columns,
            coverage="incremental",
            changed_keys=changed,
        )
        return SyncResult(
            Decision.INCREMENTAL,
            self.read_table(name),  # type: ignore[arg-type]
            len(delta.upsert),
            len(delta.drop_keys),
        )

    def write_table(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        recipe: str,
        inputs: Mapping[str, str],
        key_columns: Sequence[str],
        coverage: str,
        changed_keys: list[list[str]] | None,
    ) -> None:
        self._check_inputs(inputs)
        path = self.artifact_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        manifest = self.load_manifest()
        artifacts = manifest.setdefault("artifacts", {})
        artifacts[name] = {
            "recipe": recipe,
            "inputs": dict(sorted(inputs.items())),
            "key_columns": list(key_columns),
            "rows": int(len(frame)),
            "coverage": coverage,
            "changed_keys": changed_keys,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(manifest)

    def _empty_manifest(self) -> dict:
        payload: dict = {"version": MANIFEST_VERSION, "artifacts": {}}
        if self.baseline is not None:
            payload["baseline"] = self.baseline
        return payload

    def _write_manifest(self, manifest: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    @staticmethod
    def _check_inputs(inputs: Mapping[str, str]) -> None:
        for key, value in inputs.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("Input fingerprints must be str → str.")


def apply_frame_delta(previous: pd.DataFrame | None, delta: Delta, key_columns: Sequence[str]) -> pd.DataFrame:
    """Keep stored rows whose keys are neither replaced nor dropped."""
    keys = list(key_columns)
    if previous is None or previous.empty:
        base = delta.upsert.copy()
    else:
        base = previous.copy()
        drop = set(_key_series(delta.drop_keys, keys))
        drop.update(_key_series(delta.upsert, keys))
        if drop:
            base = base.loc[~_key_series(base, keys).isin(drop)].copy()
        if not delta.upsert.empty:
            base = pd.concat([base, delta.upsert], ignore_index=True)
    if base.empty:
        return base.reset_index(drop=True)
    return base.sort_values(keys).reset_index(drop=True)


def _key_series(frame: pd.DataFrame, key_columns: Sequence[str]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=str)
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing key columns: {missing}")
    return frame.loc[:, list(key_columns)].astype(str).agg("\0".join, axis=1)


def _key_records(frame: pd.DataFrame, key_columns: Sequence[str]) -> list[list[str]]:
    if frame is None or frame.empty:
        return []
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing key columns: {missing}")
    return frame.loc[:, list(key_columns)].astype(str).values.tolist()
