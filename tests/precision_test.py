"""Precision policy: respect input/caller dtype (float32 or float64)."""

import subprocess
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi
from jax_haspi_hasqi import precision

SR = 24000
LEVELS = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])

# conftest enables x64, so default float32 runs in a fresh interpreter.
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

print(f"HASPI {float(score):.9f} {score.dtype}")
print(f"HASQI {float(quality):.9f} {quality.dtype}")
print(f"DTYPE {before} {after}")
"""


_IN_FLOAT64_FROM_FLOAT32_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np

from jax_haspi_hasqi import haspi, hasqi, precision

assert not jax.config.jax_enable_x64, "expected a float32 process"
before = jnp.zeros(1).dtype

levels = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0], dtype=np.float32)
rng = np.random.default_rng(3)
t = np.arange(12000) / 24000
ref = 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
proc = (ref + 0.02 * rng.standard_normal(len(ref))).astype(np.float32)
ref = ref.astype(np.float32)

haspi64 = precision.in_float64(haspi.haspi_v2)
hasqi64 = precision.in_float64(hasqi.hasqi_v2)

h_np, hc_np = haspi64(ref, 24000, proc, 24000, levels)
q_np, qnl_np, ql_np, qr_np = hasqi64(ref, 24000, proc, 24000, levels)

jref, jproc, jlevels = (
  jnp.asarray(ref),
  jnp.asarray(proc),
  jnp.asarray(levels),
)
assert jref.dtype == jnp.float32
h_jnp, hc_jnp = haspi64(jref, 24000, jproc, 24000, jlevels)
q_jnp, qnl_jnp, ql_jnp, qr_jnp = hasqi64(jref, 24000, jproc, 24000, jlevels)

after = jnp.zeros(1).dtype
assert not jax.config.jax_enable_x64

print(f"HASPI_NP {float(h_np)!r} {h_np.dtype} {hc_np.dtype}")
print(f"HASPI_JNP {float(h_jnp)!r} {h_jnp.dtype} {hc_jnp.dtype}")
print(
  f"HASQI_NP {float(q_np)!r} {q_np.dtype} {qnl_np.dtype} {ql_np.dtype} {qr_np.dtype}"
)
print(
  f"HASQI_JNP {float(q_jnp)!r} {q_jnp.dtype} {qnl_jnp.dtype} {ql_jnp.dtype} {qr_jnp.dtype}"
)
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
  """A float32 caller runs natively in float32 without NaN or exception."""
  result = subprocess.run(
    [sys.executable, "-c", _FLOAT32_SCRIPT],
    capture_output=True,
    text=True,
    check=True,
  )
  lines = dict(
    line.split(" ", 1) for line in result.stdout.strip().splitlines()
  )
  haspi_val, haspi_dtype = lines["HASPI"].split()
  hasqi_val, hasqi_dtype = lines["HASQI"].split()
  got_haspi = float(haspi_val)
  got_hasqi = float(hasqi_val)
  assert haspi_dtype == "float32"
  assert hasqi_dtype == "float32"

  ref, proc = _signals()
  want_haspi = float(haspi.haspi_v2(ref, SR, proc, SR, LEVELS)[0])
  want_hasqi = float(hasqi.hasqi_v2(ref, SR, proc, SR, LEVELS)[0])

  assert abs(got_haspi - want_haspi) < 1e-3
  assert abs(got_hasqi - want_hasqi) < 1e-3


def test_input_dtype_is_preserved_when_x64_is_enabled():
  """Under jax_enable_x64, float32 inputs run in float32 and float64 in float64."""
  ref, proc = _signals()
  score32, corr32 = haspi.haspi_v2(
    jnp.float32(ref), SR, jnp.float32(proc), SR, jnp.float32(LEVELS)
  )
  qual32, _, _, raw32 = hasqi.hasqi_v2(
    jnp.float32(ref), SR, jnp.float32(proc), SR, jnp.float32(LEVELS)
  )
  assert score32.dtype == corr32.dtype == jnp.float32
  assert qual32.dtype == raw32.dtype == jnp.float32

  score64, corr64 = haspi.haspi_v2(ref, SR, proc, SR, LEVELS)
  qual64, _, _, raw64 = hasqi.hasqi_v2(ref, SR, proc, SR, LEVELS)
  assert score64.dtype == corr64.dtype == jnp.float64
  assert qual64.dtype == raw64.dtype == jnp.float64
  assert abs(float(score32) - float(score64)) < 1e-3
  assert abs(float(qual32) - float(qual64)) < 1e-3


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


def test_in_float64_promotes_metric_inputs_when_x64_is_enabled():
  """`in_float64` promotes float32 inputs even when x64 is already enabled."""
  assert jax.config.jax_enable_x64
  ref, proc = _signals()
  ref32 = ref.astype(np.float32)
  proc32 = proc.astype(np.float32)
  levels32 = LEVELS.astype(np.float32)

  # Direct calls without `in_float64` still preserve float32 inputs.
  h32, _ = haspi.haspi_v2(ref32, SR, proc32, SR, levels32)
  q32, *_ = hasqi.hasqi_v2(ref32, SR, proc32, SR, levels32)
  assert h32.dtype == jnp.float32
  assert q32.dtype == jnp.float32

  want_h64, _ = haspi.haspi_v2(
    ref32.astype(np.float64),
    SR,
    proc32.astype(np.float64),
    SR,
    levels32.astype(np.float64),
  )
  want_q64, *_ = hasqi.hasqi_v2(
    ref32.astype(np.float64),
    SR,
    proc32.astype(np.float64),
    SR,
    levels32.astype(np.float64),
  )
  assert abs(float(h32) - float(want_h64)) > 1e-4
  assert abs(float(q32) - float(want_q64)) > 1e-7

  haspi64 = precision.in_float64(haspi.haspi_v2)
  hasqi64 = precision.in_float64(hasqi.hasqi_v2)

  for r, p, lv in (
    (ref32, proc32, levels32),
    (jnp.asarray(ref32), jnp.asarray(proc32), jnp.asarray(levels32)),
  ):
    h, h_corr = haspi64(r, SR, p, SR, lv)
    q, q_nl, q_lin, q_raw = hasqi64(r, SR, p, SR, lv)
    assert h.dtype == h_corr.dtype == jnp.float64
    assert q.dtype == q_nl.dtype == q_lin.dtype == q_raw.dtype == jnp.float64
    assert float(h) == float(want_h64)
    assert float(q) == float(want_q64)


def test_in_float64_promotes_metric_inputs_from_float32_process():
  """From a float32 process, `in_float64` runs in float64 and restores x64."""
  result = subprocess.run(
    [sys.executable, "-c", _IN_FLOAT64_FROM_FLOAT32_SCRIPT],
    capture_output=True,
    text=True,
    check=True,
  )
  lines = dict(
    line.split(" ", 1) for line in result.stdout.strip().splitlines()
  )
  assert lines["DTYPE"] == "float32 float32"

  ref, proc = _signals()
  ref32 = ref.astype(np.float32)
  proc32 = proc.astype(np.float32)
  levels32 = LEVELS.astype(np.float32)
  want_h64, _ = haspi.haspi_v2(
    ref32.astype(np.float64),
    SR,
    proc32.astype(np.float64),
    SR,
    levels32.astype(np.float64),
  )
  want_q64, *_ = hasqi.hasqi_v2(
    ref32.astype(np.float64),
    SR,
    proc32.astype(np.float64),
    SR,
    levels32.astype(np.float64),
  )

  for key in ("HASPI_NP", "HASPI_JNP"):
    val, *dtypes = lines[key].split()
    assert dtypes == ["float64", "float64"]
    assert float(val) == float(want_h64)

  for key in ("HASQI_NP", "HASQI_JNP"):
    val, *dtypes = lines[key].split()
    assert dtypes == ["float64", "float64", "float64", "float64"]
    assert float(val) == float(want_q64)


class _Pair(NamedTuple):
  a: object
  b: object


class _Opaque:
  pass


@pytest.mark.parametrize("x64_before", (False, True))
def test_in_float64_promotes_nested_and_preserves_non_floating(x64_before):
  """Floating/complex leaves promote; non-floating args pass through."""
  seen = {}

  @precision.in_float64
  def inspect(*args, **kwargs):
    seen["x64"] = jax.config.jax_enable_x64
    seen["args"] = args
    seen["kwargs"] = kwargs
    return "ok"

  opaque = _Opaque()
  arr64_np = np.array([1.0], dtype=np.float64)
  arr_i32_np = np.array([1, 2], dtype=np.int32)

  with jax.enable_x64(x64_before):
    arr32_jnp = jnp.array([1.0], dtype=jnp.float32)
    arr_bf16_jnp = jnp.array([1.0], dtype=jnp.bfloat16)
    arr_c64_jnp = jnp.array([1.0 + 2.0j], dtype=jnp.complex64)
    arr_i32_jnp = jnp.array([1, 2], dtype=jnp.int32)
    arr_bool_jnp = jnp.array([True], dtype=jnp.bool_)

    out = inspect(
      np.array([1.0], dtype=np.float32),
      arr32_jnp,
      arr64_np,
      np.float32(2.5),
      np.complex64(1.0 + 2.0j),
      24000,
      True,
      65.0,
      "label",
      None,
      np.int32(7),
      np.bool_(False),
      arr_i32_np,
      arr_i32_jnp,
      arr_bool_jnp,
      opaque,
      nested={
        "tuple": (np.array([1.0], dtype=np.float16), arr_bf16_jnp),
        "list": [np.array([1.0 + 2.0j], dtype=np.complex64), arr_c64_jnp],
        "named": _Pair(np.float32(3.0), 4),
      },
    )
    assert jax.config.jax_enable_x64 is x64_before

  assert out == "ok"
  assert seen["x64"] is True
  a = seen["args"]
  assert isinstance(a[0], np.ndarray) and a[0].dtype == np.float64
  assert isinstance(a[1], jax.Array) and a[1].dtype == jnp.float64
  assert a[2] is arr64_np
  assert isinstance(a[3], np.float64) and a[3] == 2.5
  assert isinstance(a[4], np.complex128) and a[4] == 1.0 + 2.0j
  assert a[5] == 24000 and type(a[5]) is int
  assert a[6] is True
  assert a[7] == 65.0 and type(a[7]) is float
  assert a[8] == "label"
  assert a[9] is None
  assert isinstance(a[10], np.int32) and a[10] == 7
  assert isinstance(a[11], np.bool_) and not a[11]
  assert a[12] is arr_i32_np
  assert a[13] is arr_i32_jnp
  assert a[14] is arr_bool_jnp
  assert a[15] is opaque

  nested = seen["kwargs"]["nested"]
  assert isinstance(nested["tuple"], tuple)
  assert nested["tuple"][0].dtype == np.float64
  assert nested["tuple"][1].dtype == jnp.float64
  assert isinstance(nested["list"], list)
  assert nested["list"][0].dtype == np.complex128
  assert nested["list"][1].dtype == jnp.complex128
  assert isinstance(nested["named"], _Pair)
  assert isinstance(nested["named"].a, np.float64)
  assert nested["named"].b == 4 and type(nested["named"].b) is int


def test_in_float64_with_jit():
  """`in_float64` composes with `jax.jit` as outer or (under x64) inner wrap."""

  def fn(x):
    return x + jnp.asarray(1.0)

  wrapped_jit = precision.in_float64(jax.jit(fn))
  with jax.enable_x64(False):
    out = wrapped_jit(jnp.float32(2.0))
    assert not jax.config.jax_enable_x64
  assert out.dtype == jnp.float64
  assert float(out) == 3.0

  assert jax.config.jax_enable_x64
  jitted_wrapped = jax.jit(precision.in_float64(fn))
  out2 = jitted_wrapped(jnp.float32(2.0))
  assert out2.dtype == jnp.float64
  assert float(out2) == 3.0
