"""Precision policy: the port reproduces the reference only in float64.

The gammatone filter bank is a 4th-order IIR whose poles sit against the unit
circle -- 0.9913 at the 80 Hz channel, which needs about nine significant
digits in the denominator just to place the filter correctly. float32 carries
seven. The filter bank output then differs from the float64 result by a
relative 5.6e+03, and downstream HASPI raises "Signal below threshold" while
HASQI returns NaN. Computing coefficients more carefully does not rescue it:
float64 coefficients fed to a float32 recursion still drift by 6e-03.

So float64 is not a preference here, and rather than making that the caller's
problem, `in_float64` promotes for the duration of the call and restores the
caller's configuration afterwards. A float32 caller gets a score agreeing with
a native float64 run to about 2e-09.
"""

import contextlib
import functools

import jax


@contextlib.contextmanager
def _x64():
  """Enable float64 for the duration, restoring the caller's setting after."""
  with jax.enable_x64(True):
    yield


def in_float64(fn):
  """Run `fn` under float64 whatever the caller has configured.

  The jit cache survives the toggle, so this costs nothing after the first
  call for a given shape: 2.04 s cold, then 0.18 s and 0.17 s warm, measured
  from a float32 process.

  Args:
    fn: The function to wrap.

  Returns:
    The wrapped function.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    if jax.config.jax_enable_x64:
      return fn(*args, **kwargs)
    with _x64():
      return fn(*args, **kwargs)

  return wrapper


def require_x64():
  """Raise unless JAX is configured for float64.

  Retained for callers who want to assert the process-wide setting rather than
  rely on the promotion in `in_float64`. The public entry points no longer call
  this: they promote instead.

  Raises:
    RuntimeError: If x64 is disabled, naming both ways to enable it.
  """
  if not jax.config.jax_enable_x64:
    raise RuntimeError(
      "jax-haspi-hasqi requires float64. Enable it globally with "
      "jax.config.update('jax_enable_x64', True), or scope it to the call "
      "with `with jax.enable_x64(True):`. In float32 HASPI raises and HASQI "
      "returns NaN."
    )
