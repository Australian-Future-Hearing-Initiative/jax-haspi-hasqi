"""Loader for the recorded pyclarity 0.9.0 reference values."""

import dataclasses
import functools
import json
import pathlib
from typing import Any

import numpy as np

# Inside the package, not beside it: resolving relative to the source tree
# worked only in an editable install and raised FileNotFoundError from a wheel.
_GOLDENS = pathlib.Path(__file__).resolve().parent / "_golden_data"


@dataclasses.dataclass(frozen=True)
class Case:
  """One golden case: the inputs, and what the reference returned for them."""

  index: int
  name: str
  comment: str
  reference: np.ndarray
  processed: np.ndarray
  levels: np.ndarray
  reference_sample_rate: int
  processed_sample_rate: int
  audiogram: str
  level1: float
  itype: int
  equalisation: int
  haspi: dict[str, Any]
  hasqi: dict[str, Any]

  def score(self, metric: str, variant: str = "noise_free") -> float | None:
    """Reference score, or None if the reference raised for this input."""
    entry = getattr(self, metric)
    if variant not in entry:
      return None
    value = entry[variant]
    return float(value["combined"] if metric == "hasqi" else value)

  @property
  def raises(self) -> str | None:
    """Exception type the reference raised, if it did."""
    return self.haspi.get("raises") or self.hasqi.get("raises")


@functools.cache
def manifest() -> dict[str, Any]:
  """Case metadata, versions and seeds."""
  return json.loads((_GOLDENS / "manifest.json").read_text())


@functools.cache
def _arrays() -> dict[str, np.ndarray]:
  with np.load(_GOLDENS / "reference_values.npz") as data:
    return {key: data[key] for key in data.files}


@functools.cache
def cases() -> tuple[Case, ...]:
  """Every golden case, in manifest order."""
  arrays = _arrays()
  return tuple(
    Case(
      reference=arrays[f"case{entry['index']:02d}__reference"],
      processed=arrays[f"case{entry['index']:02d}__processed"],
      levels=arrays[f"case{entry['index']:02d}__levels"],
      **{
        key: entry[key]
        for key in (
          "index",
          "name",
          "comment",
          "reference_sample_rate",
          "processed_sample_rate",
          "audiogram",
          "level1",
          "itype",
          "equalisation",
          "haspi",
          "hasqi",
        )
      },
    )
    for entry in manifest()["cases"]
  )


def case(name: str) -> Case:
  """Look up a single case by name."""
  for entry in cases():
    if entry.name == name:
      return entry
  raise KeyError(name)


def stage(key: str) -> np.ndarray:
  """Per-stage value, e.g. 'ear_model__clinical__itype0__reference_db'."""
  return _arrays()[f"stage__{key}"]


def raw(key: str) -> np.ndarray:
  """Escape hatch for arrays with no accessor."""
  return _arrays()[key]
