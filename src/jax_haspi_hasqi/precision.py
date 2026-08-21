"""Precision policy: float64 is the default; float32 is viable but opt-in.

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
absolute. Before the fixes, HASPI raised "Signal below threshold" on 23 of 24
cases and HASQI returned NaN.

float64 nonetheless remains the default, because the goldens are the only
evidence and they are short and largely synthetic. `in_float64` promotes for
the duration of the call and restores the caller's configuration afterwards,
so a float32 caller gets a score agreeing with a native float64 run to about
2e-09. Callers who want float32 -- for an accelerator that rejects float64, or
for speed -- can call the undecorated `haspi_v2.__wrapped__` or
`hasqi_v2.__wrapped__`; expect agreement at the 1% level, not 1e-09.
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
      "with `with jax.enable_x64(True):`. float32 does run, but agrees with "
      "the reference only to about 1% rather than 1e-09."
    )
