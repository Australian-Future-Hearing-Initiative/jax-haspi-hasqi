"""Precision policy: respect caller and input dtype.

The gammatone bank is a 4th-order IIR whose denominator is a *quadruple* pole,
(1 - a z^-1)^4, with a = 0.9911 at the 80 Hz channel. The poles themselves are
well placed and comfortably stable -- the 80 Hz impulse response decays to a
denormal within 200k samples. What float32 could not survive was cancellation:
(1-a)^4 is 6.1e-09 there, and expanding the polynomial to reach it costs about
nine digits against terms of size six. float32 carries seven.

That single mistake appeared at three sites, all now fixed by keeping the
factor intact:

  - the direct-form recursion, 3.0e+18 relative error, replaced by four
    cascaded one-pole sections (`filters.cascaded_one_pole`, 4.2e-06);
  - the filter gain, which cancelled to exactly 0.0, now 2(1-a)^4/(1+2a)^2;
  - the group delay, which divided by that same zero and produced inf, then
    silently became INT_MIN on the cast to an integer shift, now the closed
    form 4a/(1+2a) + 4a/(1-a).

With those in place float32 runs end to end on every golden case: worst
relative error 1.30% on the final score, worst absolute 3.8e-03, no raises.
The one case above 1% is a HASQI score of 0.0763, where 1.30% is 9.9e-04
absolute. On real speech (VCTK, calibrated to 65 dB SPL), float32 drifts
further from float64: up to 1.87e-02 on HASPI and 4.37e-03 on HASQI.

`haspi_v2`, `hasqi_v2` and `ear_model` compute in the floating-point dtype of
their inputs (`jnp.result_type(reference, processed, jnp.float32)`): float32 in
a default JAX process or when given float32 arrays, and float64 when x64 is
enabled and the inputs are float64. `in_float64` enables x64 for the call and
promotes floating-point inputs to float64; `require_x64` asserts that x64 is
enabled.
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np


@contextlib.contextmanager
def _x64():
  """Enable float64 for the duration, restoring the caller's setting after."""
  with jax.enable_x64(True):
    yield


def _promote_float64(x):
  if isinstance(x, (jax.Array, np.ndarray, np.generic)):
    if jnp.issubdtype(x.dtype, jnp.floating):
      return x if x.dtype == np.float64 else x.astype(np.float64)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
      return x if x.dtype == np.complex128 else x.astype(np.complex128)
  return x


def in_float64(fn):
  """Run `fn` under float64 whatever the caller has configured.

  Enables `jax_enable_x64` for the duration of the call, restores the caller's
  setting on exit, and promotes floating-point (`float64`) and complex
  (`complex128`) array/scalar inputs in `*args` and `**kwargs` (including
  nested pytrees) before calling `fn`, even when `jax_enable_x64` is already
  enabled. Non-floating arguments are left unchanged.

  The jit cache survives the toggle, so this costs nothing after the first
  call for a given shape: 2.04 s cold, then 0.18 s and 0.17 s warm, measured
  from a float32 process. When combining with an outer `jax.jit` from a
  float32 process, wrap the jitted callable (`in_float64(jax.jit(fn))`), as
  JAX does not support enabling x64 mid-trace.

  Args:
    fn: The function to wrap.

  Returns:
    The wrapped function.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with _x64():
      args, kwargs = jax.tree_util.tree_map(_promote_float64, (args, kwargs))
      return fn(*args, **kwargs)

  return wrapper


def require_x64():
  """Raise unless JAX is configured for float64.

  Retained for callers who want to assert the process-wide setting rather than
  wrapping calls with `in_float64`. The public entry points do not call this
  and preserve the caller's input dtype instead.

  Raises:
    RuntimeError: If x64 is disabled, naming both ways to enable it.
  """
  if not jax.config.jax_enable_x64:
    raise RuntimeError(
      "jax-haspi-hasqi requires float64. Enable it globally with "
      "jax.config.update('jax_enable_x64', True), or scope it to the call "
      "with `with jax.enable_x64(True):`. float32 does run, but agrees with "
      "the reference only to about 1% rather than 1e-09."
    )
