"""Digital filter helpers, independently of the metrics that use them."""

import jax.numpy as jnp
import numpy as np
import scipy.signal

from jax_haspi_hasqi import filters

SIZES = ((64, 64), (128, 96), (301, 301))


def _signals(n, m, seed=0):
  rng = np.random.default_rng(seed)
  return rng.standard_normal(n), rng.standard_normal(m)


def test_correlate_full_matches_scipy():
  for n, m in SIZES:
    x, y = _signals(n, m)
    got = np.asarray(filters.correlate_full(jnp.asarray(x), jnp.asarray(y)))
    np.testing.assert_allclose(got, scipy.signal.correlate(x, y, "full"))


def test_the_default_computes_in_the_inputs_dtype():
  """dtype=None must leave existing callers exactly as they were."""
  x, y = _signals(128, 128)
  x, y = jnp.asarray(x), jnp.asarray(y)
  explicit = filters.correlate_full(x, y, dtype=x.dtype)
  np.testing.assert_array_equal(
    np.asarray(filters.correlate_full(x, y)), np.asarray(explicit)
  )


def test_dtype_selects_the_transform_precision_only():
  """The result comes back in the caller's dtype whatever the transform used."""
  x, y = _signals(128, 128)
  x, y = jnp.asarray(x), jnp.asarray(y)
  assert x.dtype == jnp.float64

  single = filters.correlate_full(x, y, dtype=jnp.float32)

  assert single.dtype == jnp.float64
  np.testing.assert_allclose(
    np.asarray(single),
    scipy.signal.correlate(np.asarray(x), np.asarray(y), "full"),
    atol=1e-5,
  )


def test_single_precision_costs_accuracy():
  """Why the default is not float32: the transform loses ~8 digits.

  Callers that resolve a discrete argmax over the result cannot afford this;
  callers that reduce it to a magnitude can.
  """
  x, y = _signals(128, 128)
  x, y = jnp.asarray(x), jnp.asarray(y)
  want = scipy.signal.correlate(np.asarray(x), np.asarray(y), "full")

  double = np.max(np.abs(np.asarray(filters.correlate_full(x, y)) - want))
  single = np.max(
    np.abs(np.asarray(filters.correlate_full(x, y, dtype=jnp.float32)) - want)
  )

  assert double < 1e-12
  assert single > double
