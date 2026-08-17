"""Precision policy: float64 internally, whatever the caller has configured."""

import subprocess
import sys

import jax
import numpy as np
import pytest

from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi
from jax_haspi_hasqi import precision

SR = 24000
LEVELS = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])

# conftest enables x64 process-wide, so the float32 path can only be exercised
# honestly in a fresh interpreter.
_FLOAT32_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np

from jax_haspi_hasqi import haspi, hasqi

assert not jax.config.jax_enable_x64, "expected a float32 process"
before = jnp.zeros(1).dtype

levels = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])
rng = np.random.default_rng(3)
t = np.arange(12000) / 24000
ref = 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
proc = ref + 0.02 * rng.standard_normal(len(ref))

score, _ = haspi.haspi_v2(
  np.float32(ref), 24000, np.float32(proc), 24000, levels
)
quality, *_ = hasqi.hasqi_v2(
  np.float32(ref), 24000, np.float32(proc), 24000, levels
)
after = jnp.zeros(1).dtype

print(f"HASPI {float(score):.9f}")
print(f"HASQI {float(quality):.9f}")
print(f"DTYPE {before} {after}")
"""


def _signals(n=12000, seed=3):
  rng = np.random.default_rng(seed)
  t = np.arange(n) / SR
  ref = (
    0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
  )
  return ref, ref + 0.02 * rng.standard_normal(n)


def test_float32_process_gets_a_score_not_an_exception():
  """A float32 caller used to get a NaN from HASQI and a misleading
  'Signal below threshold' from HASPI. Now it gets the float64 answer."""
  result = subprocess.run(
    [sys.executable, "-c", _FLOAT32_SCRIPT],
    capture_output=True,
    text=True,
    check=True,
  )
  lines = dict(
    line.split(" ", 1) for line in result.stdout.strip().splitlines()
  )
  got_haspi = float(lines["HASPI"])
  got_hasqi = float(lines["HASQI"])

  ref, proc = _signals()
  want_haspi = float(haspi.haspi_v2(ref, SR, proc, SR, LEVELS)[0])
  want_hasqi = float(hasqi.hasqi_v2(ref, SR, proc, SR, LEVELS)[0])

  # float32 inputs carry ~1e-7 of their own, so this is agreement on the
  # computation, not on the inputs.
  assert abs(got_haspi - want_haspi) < 1e-6
  assert abs(got_hasqi - want_hasqi) < 1e-6


def test_the_callers_dtype_is_restored():
  """Promotion must not leak: a float32 process stays float32 afterwards."""
  result = subprocess.run(
    [sys.executable, "-c", _FLOAT32_SCRIPT],
    capture_output=True,
    text=True,
    check=True,
  )
  dtypes = [
    line.split()[1:]
    for line in result.stdout.splitlines()
    if line.startswith("DTYPE")
  ][0]
  assert dtypes == ["float32", "float32"]


def test_in_float64_restores_the_previous_setting():
  @precision.in_float64
  def check():
    assert jax.config.jax_enable_x64
    return True

  with jax.enable_x64(False):
    assert not jax.config.jax_enable_x64
    assert check()
    assert not jax.config.jax_enable_x64


def test_in_float64_is_a_no_op_when_already_enabled():
  assert jax.config.jax_enable_x64

  @precision.in_float64
  def check():
    return jax.config.jax_enable_x64

  assert check()
  assert jax.config.jax_enable_x64


def test_in_float64_restores_even_when_the_call_raises():
  @precision.in_float64
  def boom():
    raise ValueError("boom")

  with jax.enable_x64(False):
    with pytest.raises(ValueError, match="boom"):
      boom()
    assert not jax.config.jax_enable_x64


def test_require_x64_still_available_for_callers_who_want_it():
  """No longer used by the entry points, but kept as a public assertion."""
  assert jax.config.jax_enable_x64
  precision.require_x64()

  with jax.enable_x64(False):
    with pytest.raises(RuntimeError, match="requires float64"):
      precision.require_x64()
