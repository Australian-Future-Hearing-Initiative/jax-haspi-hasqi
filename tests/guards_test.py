"""The guards that keep a caller from getting a plausible wrong answer."""

import pathlib
import subprocess
import sys

import jax
import pytest

from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import precision

# conftest enables x64 process-wide, so the float32 path can only be exercised
# in a fresh interpreter.
_FLOAT32_SCRIPT = """
import numpy as np
from jax_haspi_hasqi import haspi, hasqi
levels = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])
signal = np.sin(2 * np.pi * 220 * np.arange(6000) / 24000)
try:
  haspi.haspi_v2(signal, 24000, signal, 24000, levels)
except RuntimeError as error:
  print("RAISED:", error)
"""


def test_require_x64_passes_when_enabled():
  assert jax.config.jax_enable_x64
  precision.require_x64()


def test_require_x64_raises_when_disabled():
  with jax.enable_x64(False):
    with pytest.raises(RuntimeError, match="requires float64"):
      precision.require_x64()


def test_the_error_names_both_ways_to_fix_it():
  """A caller hitting this needs the remedy, not a diagnosis."""
  with jax.enable_x64(False):
    with pytest.raises(RuntimeError) as caught:
      precision.require_x64()
  message = str(caught.value)
  assert "jax_enable_x64" in message
  assert "enable_x64(True)" in message


def test_float32_callers_get_an_exception_not_a_nan():
  """The whole point: float32 used to return NaN from HASQI and a misleading
  'Signal below threshold' from HASPI."""
  result = subprocess.run(
    [sys.executable, "-c", _FLOAT32_SCRIPT],
    capture_output=True,
    text=True,
    check=True,
  )
  assert "RAISED:" in result.stdout
  assert "float64" in result.stdout


def test_goldens_live_inside_the_package():
  """Resolving them beside the source tree broke every non-editable install."""
  data = pathlib.Path(goldens.__file__).resolve().parent / "_golden_data"
  assert data.is_dir()
  assert (data / "reference_values.npz").is_file()
  assert (data / "manifest.json").is_file()


def test_the_loader_reads_from_that_directory():
  assert goldens._GOLDENS.name == "_golden_data"
  assert goldens._GOLDENS.parent.name == "jax_haspi_hasqi"
  assert len(goldens.cases()) > 0
