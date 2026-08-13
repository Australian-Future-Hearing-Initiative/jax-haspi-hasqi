"""Precision guard: the port reproduces the reference only in float64."""

import jax


def require_x64():
  """Raise unless JAX is configured for float64.

  In float32 the gammatone recursion carries 7.6e-02 relative error, which is
  not a metric. The failure is worse than inaccurate: HASPI raises
  `ValueError: Signal below threshold` because the dB envelopes drift under
  the cepstral gate, and HASQI returns NaN. Neither says anything about
  precision, so a caller gets a plausible-looking hole in a results table.

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
