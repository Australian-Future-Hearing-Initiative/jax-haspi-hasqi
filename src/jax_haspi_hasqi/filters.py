"""Digital filters in JAX, matching scipy.signal semantics used by the reference."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from jax import lax


def lfilter(b, a, x):
  """scipy.signal.lfilter for a 1-D signal, as a scan in direct form II transposed.

  Args:
    b: Numerator coefficients.
    a: Denominator coefficients; a[0] must be 1, as it is everywhere here.
    x: Signal to filter.
  """
  b = jnp.asarray(b, x.dtype)
  a = jnp.asarray(a, x.dtype)
  order = max(len(b), len(a)) - 1
  b = jnp.pad(b, (0, order + 1 - len(b)))
  a = jnp.pad(a, (0, order + 1 - len(a)))

  def step(state, sample):
    y = b[0] * sample + state[0]
    shifted = jnp.concatenate([state[1:], jnp.zeros(1, state.dtype)])
    return shifted + b[1:] * sample - a[1:] * y, y

  _, out = lax.scan(step, jnp.zeros(order, x.dtype), x)
  return out


def correlate_full(x, y, dtype=None):
  """scipy.signal.correlate(x, y, 'full').

  Via FFT: `jnp.convolve` lowers to a general convolution whose cost explodes
  when both operands are batched by vmap, which is how the filter bank uses
  it. Agreement with the direct form is ~1e-12 relative, far below the port's
  error floor.

  Args:
    x: First input signal.
    y: Second input signal.
    dtype: Precision for the transform only; the result is returned in x's
      dtype. Defaults to x's dtype. Accelerators that reject float64 FFT
      operands need float32 here, at a cost of ~8 significant digits. Callers
      that reduce the result to a magnitude can absorb that; callers that
      resolve a discrete argmax over it cannot, as neighbouring correlation
      peaks can sit closer together than float32 can represent.
  """
  n = x.shape[0] + y.shape[0] - 1
  size = 1 << int(np.ceil(np.log2(n)))
  compute = x.dtype if dtype is None else dtype
  spectrum = jnp.fft.rfft(x.astype(compute), size) * jnp.fft.rfft(
    y[::-1].astype(compute), size
  )
  return jnp.fft.irfft(spectrum, size)[:n].astype(x.dtype)


def correlate_at_lags(x, y, indices):
  """correlate_full(x, y)[indices], evaluated directly rather than transformed.

  Each index costs one dot product, so this is only worth it for a handful of
  lags; in exchange it needs no FFT and so keeps the caller's precision on
  accelerators that reject float64 FFT operands.

  Args:
    x: First input signal.
    y: Second input signal, the same length as x.
    indices: Indices into the length 2*n-1 full correlation.

  Returns:
    The correlation at those indices, in x's dtype.
  """
  n = x.shape[0]
  positions = jnp.arange(n)

  def at(index):
    # correlate_full index k is the lag d = k - (n - 1), so y is read at t - d.
    taps = positions - (index - (n - 1))
    inside = (taps >= 0) & (taps < n)
    return jnp.sum(x * jnp.where(inside, y[jnp.clip(taps, 0, n - 1)], 0))

  return jax.vmap(at)(indices)


def group_delay_at_dc(b, a):
  """Group delay at w=1 rad/sample, as scipy.signal.group_delay((b, a), w=1)[1][0].

  The closed form avoids scipy's FFT path; verified to agree to 6e-10 on the
  gammatone coefficients this is used with.
  """
  b = jnp.asarray(b)
  a = jnp.asarray(a)
  kb = jnp.arange(len(b), dtype=b.dtype)
  ka = jnp.arange(len(a), dtype=a.dtype)
  return jnp.sum(kb * b) / jnp.sum(b) - jnp.sum(ka * a) / jnp.sum(a)


def upfirdn(h, x, up, down):
  """scipy.signal.upfirdn: zero-stuff by up, FIR filter, keep every down-th sample."""
  n = x.shape[0]
  stuffed = jnp.zeros((n - 1) * up + 1, x.dtype).at[::up].set(x)
  return jnp.convolve(stuffed, jnp.asarray(h, x.dtype), "full")[::down]


def _output_len(len_h, in_len, up, down):
  """Number of samples upfirdn returns, as scipy computes it."""
  return (((in_len - 1) * up + len_h) - 1) // down + 1


def resample_poly_taps(up, down):
  """The FIR taps scipy.signal.resample_poly designs, zero-padded as it pads them.

  Static in the sampling rates, so this stays in numpy and out of the graph.

  Returns:
    The padded taps, and the number of leading output samples to discard.
  """
  max_rate = max(up, down)
  half_len = 10 * max_rate
  taps = (
    scipy.signal.firwin(
      2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)
    )
    * up
  )

  n_pre_pad = down - half_len % down
  n_pre_remove = (half_len + n_pre_pad) // down
  n_post_pad = 0
  return taps, n_pre_pad, n_post_pad, n_pre_remove


def resample_poly(x, up, down, n_in=None):
  """scipy.signal.resample_poly with the default kaiser(5.0) window.

  Args:
    x: Signal to resample.
    up: Upsampling factor, reduced by the gcd with `down` internally.
    down: Downsampling factor.
    n_in: Length of x, if it is not statically known.
  """
  factor = math.gcd(int(up), int(down))
  up = int(up) // factor
  down = int(down) // factor
  if up == down == 1:
    return x

  n_in = x.shape[0] if n_in is None else n_in
  n_out = n_in * up
  n_out = n_out // down + bool(n_out % down)

  taps, n_pre_pad, n_post_pad, n_pre_remove = resample_poly_taps(up, down)
  while (
    _output_len(len(taps) + n_pre_pad + n_post_pad, n_in, up, down)
    < n_out + n_pre_remove
  ):
    n_post_pad += 1
  taps = np.concatenate((np.zeros(n_pre_pad), taps, np.zeros(n_post_pad)))

  return upfirdn(taps, x, up, down)[n_pre_remove : n_pre_remove + n_out]


def shift_with_zeros(x, delay):
  """Advance x by `delay` samples, zero-filling; negative delay retards it.

  Reproduces the reference's two-branch concatenate with a single dynamic
  slice, so `delay` may be a traced value.
  """
  npts = x.shape[0]
  padded = jnp.concatenate(
    [jnp.zeros(npts, x.dtype), x, jnp.zeros(npts, x.dtype)]
  )
  return lax.dynamic_slice(padded, (npts + delay,), (npts,))
