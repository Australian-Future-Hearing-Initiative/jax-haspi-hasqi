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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

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

  Hooks the lowering entry point, which is not public API. If a JAX upgrade
  moves it the patch stops collecting, so every caller asserts it saw at least
  one module and test_the_guard_detects_a_float64_transform plants one to
  prove the whole path still reports.
  """
  modules = []
  original = mlir.lower_jaxpr_to_module

  def record(*args, **kwargs):
    result = original(*args, **kwargs)
    modules.append(result.module)
    return result

  mlir.lower_jaxpr_to_module = record
  try:
    # Lowerings are cached, so only a cold cache lowers anything at all.
    jax.clear_caches()
    yield modules
  finally:
    mlir.lower_jaxpr_to_module = original


def _operations(operation):
  """Every operation below `operation`, at any nesting depth."""
  for region in operation.regions:
    for block in region.blocks:
      for child in block.operations:
        yield child
        yield from _operations(child.operation)


def float_width(mlir_type):
  """Floating width in bits of a tensor's element type, unwrapping complex.

  Read through the MLIR type API rather than parsed out of the printed form:
  the spelling of a shaped complex type is not something to depend on.
  """
  element = ir.ShapedType(mlir_type).element_type
  try:
    element = ir.ComplexType(element).element_type
  except ValueError:
    pass
  width = getattr(element, "width", None)
  # An FFT operand is float or complex-float, so anything else is a signal
  # that this is reading the wrong thing rather than a transform to allow.
  assert width is not None, f"FFT operand of unexpected type {element}"
  return width


def fft_float_widths(modules):
  """Floating width of every FFT operand and result across the modules."""
  widths = []
  for module in modules:
    for operation in _operations(module.operation):
      if operation.operation.name != "stablehlo.fft":
        continue
      for value in [*operation.operands, *operation.results]:
        widths.append(float_width(value.type))
  return widths


def assert_single_precision_transforms(modules):
  assert modules, "no lowering captured; the hook is no longer being called"
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
