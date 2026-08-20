"""NAL-R hearing-aid equalisation, ported from clarity.enhancer.nalr.

The prescription formula -- the 0.31 slope factor, the six bias terms, and the
two-branch x_ave -- is published in Byrne D, Dillon H (1986), "The National
Acoustic Laboratories' (NAL) new procedure for selecting the gain and
frequency response of a hearing aid", Ear and Hearing 7(4):257-265. NAL-R is
the linear rule from that paper, and is not NAL-NL1 or NAL-NL2, which are
separately licensed. See NOTICE.
"""

import functools

import jax.numpy as jnp
import numpy as np
import scipy.signal.windows

NALR_FREQUENCIES = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0])
_BIAS = np.array([-17.0, -8.0, 1.0, -1.0, -2.0, -2.0])


@functools.lru_cache(maxsize=None)
def _inverse_transform(length, grid_len):
  """The linear map _fir2's inverse transform applies, as a real matrix.

  The spectrum _fir2 builds is Hermitian, and only its first `length` real
  samples survive, so the inverse transform reduces to a cosine sum over the
  half spectrum. Evaluating that directly rather than as a transform keeps the
  design in the caller's precision on accelerators that reject float64 FFT
  operands, and discards nothing it paid for: the transform computes
  2*(grid_len-1) samples to keep `length` of them.

  Returns a NumPy array: the cache outlives any one trace, so converting
  here would cache a tracer from whichever call traced first.
  """
  delay = 0.5 * (length - 1)
  size = 2 * (grid_len - 1)
  bins = np.arange(grid_len)
  angle = 2 * np.pi * np.outer(np.arange(length), bins) / size - (
    delay * np.pi * bins / (grid_len - 1)
  )
  # Every bin but DC and Nyquist appears twice in the Hermitian spectrum.
  weight = np.full(grid_len, 2.0)
  weight[0] = weight[-1] = 1.0
  return np.cos(angle) * weight / size


def _fir2(order, frequencies, gains, n_interpolate=512):
  """MATLAB fir2, as pyclarity reimplements it in msbg_utils.

  The breakpoint frequencies are fixed by the caller, so the interpolation
  index arithmetic is static and only the gains flow through the graph.
  """
  length = order + 1
  window = jnp.asarray(scipy.signal.windows.hamming(length))
  gains = jnp.asarray(gains)

  frequencies = np.asarray(frequencies, dtype=np.float64).copy()
  frequencies[0] = 0.0
  frequencies[-1] = 1.0
  lap = int(np.fix(n_interpolate / 25))
  grid_len = n_interpolate + 1

  response = jnp.zeros(grid_len, gains.dtype).at[0].set(gains[0])
  start = 0
  for i in range(len(frequencies) - 1):
    if frequencies[i + 1] == frequencies[i]:
      start = int(np.ceil(start - lap / 2))
      end = start + lap - 1
    else:
      end = int(np.fix(frequencies[i + 1] * grid_len)) - 1
    index = np.arange(start, end + 1)
    ramp = (
      np.zeros(len(index)) if start == end else (index - start) / (end - start)
    )
    response = response.at[index].set(
      ramp * gains[i + 1] + (1 - ramp) * gains[i]
    )
    start = end + 1

  transform = jnp.asarray(_inverse_transform(length, grid_len), response.dtype)

  return (transform @ response) * window


def build(hearing_loss, n_fir=140, sample_rate=24000.0):
  """NAL-R FIR filter for one audiogram, plus the matching pure-delay filter.

  Args:
    hearing_loss: Levels in dB HL at [250, 500, 1000, 2000, 4000, 6000] Hz.
    n_fir: Filter order.
    sample_rate: Sampling rate in Hz.
  """
  hearing_loss = jnp.asarray(hearing_loss)
  nyquist = 0.5 * sample_rate
  delay = jnp.zeros(n_fir + 1).at[n_fir // 2].set(1.0)

  critical_loss = hearing_loss[1] + hearing_loss[2] + hearing_loss[3]
  x_ave = jnp.where(
    critical_loss <= 180,
    0.05 * critical_loss,
    9.0 + 0.116 * (critical_loss - 180),
  )
  gain_db = jnp.clip(x_ave + 0.31 * hearing_loss + _BIAS, 0.0, None)

  # Interpolate the gains onto a uniform grid, extending flat past the ends.
  grid = np.linspace(0, n_fir, n_fir + 1) / n_fir
  knots = np.concatenate(([0.0], NALR_FREQUENCIES, [nyquist]))
  knot_gains = jnp.concatenate((gain_db[:1], gain_db, gain_db[-1:]))
  interpolated = jnp.interp(nyquist * grid, jnp.asarray(knots), knot_gains)
  nalr = _fir2(n_fir, grid, jnp.power(10.0, interpolated / 20.0))

  # A flat audiogram needs no gain, and the reference returns the delay itself.
  return jnp.where(jnp.max(hearing_loss) > 0, nalr, delay), delay


def apply(nalr, signal):
  """Convolve a signal with a built NAL-R filter."""
  return jnp.convolve(signal, nalr)
