"""No float64 FFT reaches the compiled graph.

Some accelerators reject float64 FFT operands while accepting float64 dot
products, reductions and elementwise arithmetic. Code that is correct on a
host CPU, where float64 is legal everywhere, can therefore fail to compile on
such a device -- which is how this was found, at runtime and outside this
suite. Running the metrics cannot catch it here: the suite runs on CPU, which
accepts every dtype, so a reintroduced float64 transform stays green.

These tests instead lower the metrics to StableHLO and inspect the emitted
instructions. Every `stablehlo.fft` must carry single-precision operands and
results. That is a property of the graph rather than of any one function, so
it holds for a float64 transform introduced anywhere the metrics trace
through, including somewhere nobody thought to check.

The lowering is captured for real `haspi_v2` and `hasqi_v2` calls rather than
by tracing them directly: both resolve their alignment in numpy, so their
output shape depends on their input values and neither can be placed under a
single `jit`. Capturing what they compile instead covers every region they
actually compile, at the shapes they actually use.
"""

import contextlib
import pathlib
import re
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi
from jax_haspi_hasqi import nalr

# Widest float an FFT may use for the constrained devices to accept it.
MAX_FFT_WIDTH = 32

AUDIOGRAMS = ("flat_zero", "clinical")


def levels(name):
  return np.asarray(goldens.manifest()["audiograms"][name])


@contextlib.contextmanager
def captured_lowerings():
  """Collect the StableHLO of every module lowered inside the block.

  Asks JAX to dump each module it lowers, which reaches the regions the
  metrics compile internally and not just a top-level call. Every caller
  asserts it saw at least one module and
  test_the_guard_detects_a_float64_transform plants one to prove the whole
  path still reports.
  """
  with tempfile.TemporaryDirectory() as directory:
    jax.config.update("jax_dump_ir_to", directory)
    try:
      # Lowerings are cached, so only a cold cache lowers anything at all.
      jax.clear_caches()
      modules = []
      yield modules
    finally:
      jax.config.update("jax_dump_ir_to", None)
    modules.extend(
      path.read_text() for path in sorted(pathlib.Path(directory).iterdir())
    )


# Operands and results of a printed op follow the final colon, as
# `(tensor<...>, ...) -> tensor<...>`; f64 there is what this forbids.
_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_FLOAT_WIDTH = re.compile(r"\bf(?P<width>\d+)\b")


def fft_float_widths(modules):
  """Floating width of every FFT operand and result across the modules."""
  widths = []
  for module in modules:
    for line in module.splitlines():
      match = _FFT_LINE.match(line)
      if match is None:
        continue
      types = match.group("types")
      found = _FLOAT_WIDTH.findall(types)
      # An FFT operand is float or complex-float, so no float element type at
      # all means this is reading the wrong thing rather than a transform to
      # allow.
      assert found, f"FFT operand of unexpected type {types.strip()}"
      widths.extend(int(width) for width in found)
  return widths


def assert_single_precision_transforms(modules):
  assert modules, "no lowering captured; the dump is no longer being written"
  offending = sorted(
    width for width in set(fft_float_widths(modules)) if width > MAX_FFT_WIDTH
  )
  assert not offending, f"FFT operands wider than float32: {offending}"


def test_the_guard_detects_a_float64_transform():
  """The guard is worth nothing unless it fails on what it exists to forbid."""
  size = 64

  def offending(x):
    return jnp.fft.irfft(jnp.fft.rfft(x, size), size)

  with captured_lowerings() as modules:
    jax.jit(offending)(jnp.zeros(size, jnp.float64))

  assert 64 in fft_float_widths(modules)
  with pytest.raises(AssertionError, match="wider than float32"):
    assert_single_precision_transforms(modules)


def test_the_guard_passes_a_single_precision_transform():
  size = 64

  def permitted(x):
    single = jnp.fft.irfft(jnp.fft.rfft(x.astype(jnp.float32), size), size)
    return single.astype(x.dtype)

  with captured_lowerings() as modules:
    jax.jit(permitted)(jnp.zeros(size, jnp.float64))

  assert fft_float_widths(modules), "expected the transform to be lowered"
  assert_single_precision_transforms(modules)


def test_the_nalr_design_emits_no_float64_transform():
  """The filter design is the one FFT the metrics reach outside a correlation."""
  with captured_lowerings() as modules:
    jax.block_until_ready(nalr.build(jnp.asarray(levels("clinical"))))

  assert_single_precision_transforms(modules)


@pytest.mark.parametrize("audiogram", AUDIOGRAMS)
def test_haspi_emits_no_float64_transform(audiogram):
  reference = goldens.stage("reference_signal")
  processed = goldens.stage("processed_signal")

  with captured_lowerings() as modules:
    jax.block_until_ready(
      haspi.haspi_v2(reference, 24000, processed, 24000, levels(audiogram))
    )

  assert_single_precision_transforms(modules)


@pytest.mark.parametrize("audiogram", AUDIOGRAMS)
@pytest.mark.parametrize("equalisation", (1, 2))
def test_hasqi_emits_no_float64_transform(audiogram, equalisation):
  """equalisation=1 designs the NAL-R filter inside the graph; 2 does not."""
  reference = goldens.stage("reference_signal")
  processed = goldens.stage("processed_signal")

  with captured_lowerings() as modules:
    jax.block_until_ready(
      hasqi.hasqi_v2(
        reference,
        24000,
        processed,
        24000,
        levels(audiogram),
        equalisation=equalisation,
      )
    )

  assert_single_precision_transforms(modules)


def test_design_cache_survives_tracing():
  """The design cache must outlive the trace that fills it.

  `_inverse_transform` is `lru_cache`d and first runs inside whichever `jit`
  traces first, so returning a JAX array would cache a tracer and every later
  caller would get one belonging to a dead trace.
  """
  nalr._inverse_transform.cache_clear()

  jax.block_until_ready(
    jax.jit(lambda x: x * nalr._inverse_transform(97, 513).sum())(
      jnp.float32(1.0)
    )
  )

  assert isinstance(nalr._inverse_transform(97, 513), np.ndarray)
