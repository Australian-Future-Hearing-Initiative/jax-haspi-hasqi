"""Gradient behaviour of the aligned ear model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import goldens


@pytest.fixture(scope="module")
def aligned():
  reference = goldens.stage("reference_signal")
  processed = goldens.stage("processed_signal")
  reference, processed, _ = ear_model.input_align(reference, processed)
  return jnp.asarray(reference), jnp.asarray(processed)


def envelope_mean(reference, processed, nchan=32):
  noise = jnp.zeros((nchan, reference.shape[0]))
  levels = jnp.asarray([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])
  outputs = ear_model._ear_model_aligned(
    reference, processed, levels, 0, 65.0, noise, noise, nchan, 1
  )
  return jnp.mean(outputs[2])


def test_gradient_is_finite(aligned):
  """A zero-valued envelope must not send an infinite sqrt derivative back."""
  reference, processed = aligned
  gradient = jax.grad(envelope_mean, argnums=1)(reference, processed)
  assert bool(jnp.all(jnp.isfinite(gradient)))


def test_gradient_is_not_identically_zero(aligned):
  reference, processed = aligned
  gradient = jax.grad(envelope_mean, argnums=1)(reference, processed)
  assert float(jnp.max(jnp.abs(gradient))) > 0


def test_gradient_matches_a_finite_difference(aligned):
  reference, processed = aligned
  gradient = jax.grad(envelope_mean, argnums=1)(reference, processed)
  index = processed.shape[0] // 2
  step = 1e-6
  up = envelope_mean(reference, processed.at[index].add(step))
  down = envelope_mean(reference, processed.at[index].add(-step))
  assert float((up - down) / (2 * step)) == pytest.approx(
    float(gradient[index]), rel=1e-2
  )


def test_safe_sqrt_leaves_values_alone():
  x = jnp.asarray([0.0, 1e-30, 1.0, 4.0])
  np.testing.assert_allclose(
    np.asarray(ear_model._safe_sqrt(x)), np.sqrt(np.asarray(x))
  )


def test_safe_sqrt_has_a_finite_derivative_at_zero():
  assert float(jax.grad(ear_model._safe_sqrt)(0.0)) == 0.0
