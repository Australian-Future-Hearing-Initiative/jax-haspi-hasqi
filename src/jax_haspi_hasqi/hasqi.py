"""HASQI v2 quality index, ported from clarity.evaluator.hasqi and eb.py."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import filters
from jax_haspi_hasqi import precision

_SMALL = 1e-30
_SILENCE_THRESHOLD = 2.5
_SEGMENT_COVARIANCE = 16
_CEPSTRAL_COEFFICIENTS = 6
_LOUDNESS_WEIGHT = 0.579
_SLOPE_WEIGHT = 0.421
_LOUDNESS_SCALE = 2.5
_SYNC_FILTER_ORDERS = np.array([1, 3, 5, 5, 5, 5])
_SYNC_CUTOFFS = 1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def _segment_layout(npts, nwin):
  """Start offsets of the reference's 50 %-overlap segmentation.

  The first and last segments are half-windows, so they are handled apart from
  the full ones.

  Returns:
    Starts of the full segments, and the total segment count.
  """
  nhalf = nwin // 2
  nseg = int(1 + np.floor(npts / nwin) + np.floor((npts - nwin / 2) / nwin))
  starts = nhalf * np.arange(1, nseg - 1)
  return starts, nseg


def _gather_segments(signal, starts, length):
  """Slice each channel at every start, zero-padding reads past the end.

  Args:
    signal: Shape (nchan, npts).
    starts: Segment start offsets.
    length: Samples per segment.

  Returns:
    Shape (nchan, nseg, length).
  """
  padded = jnp.pad(signal, ((0, 0), (0, length)))
  index = jnp.asarray(starts)[:, None] + np.arange(length)[None, :]
  return padded[:, index]


def env_smooth(
  envelopes, segment_size=_SEGMENT_COVARIANCE, sample_rate=24000.0
):
  """Window and subsample the envelopes to a 125 Hz frame rate.

  Args:
    envelopes: Envelope per band, shape (nchan, nsamp).
    segment_size: Averaging segment in ms.
    sample_rate: Envelope sampling rate in Hz.
  """
  nwin = int(np.around(segment_size * (0.001 * sample_rate)))
  nwin += nwin % 2
  nhalf = nwin // 2

  window = np.hanning(nwin)
  window_sum = np.sum(window)
  half_sum = np.sum(window[nhalf:nwin])

  starts, nseg = _segment_layout(envelopes.shape[1], nwin)

  first = jnp.sum(envelopes[:, :nhalf] * window[nhalf:nwin], axis=1) / half_sum
  middle = (
    jnp.sum(_gather_segments(envelopes, starts, nwin) * window, axis=2)
    / window_sum
  )
  last = (
    jnp.sum(
      _gather_segments(envelopes, np.array([nhalf * (nseg - 1)]), nhalf)[:, 0]
      * window[:nhalf],
      axis=1,
    )
    / half_sum
  )
  return jnp.concatenate([first[:, None], middle, last[:, None]], axis=1)


def _mel_cepstrum_basis(nbands, nbasis=_CEPSTRAL_COEFFICIENTS):
  basis = np.cos(
    np.outer(np.arange(nbands), np.arange(nbasis)) * np.pi / (nbands - 1)
  )
  return basis / np.linalg.norm(basis, axis=0)


def mel_cepstrum_correlation(
  reference, distorted, threshold=_SILENCE_THRESHOLD, noise=None
):
  """Cross-covariance of the mel cepstra, averaged over basis functions 2-6.

  Args:
    reference: Smoothed reference envelope in dB SL, shape (nbands, nseg).
    distorted: Smoothed processed envelope, same shape.
    threshold: Loudness in dB above which a segment is included.
    noise: Additive envelope noise. HASQI passes zero, so this defaults to
      none; the reference still draws it from the unseeded global RNG.

  Returns:
    The average cepstral correlation and the six individual ones.
  """
  nbands = reference.shape[0]
  basis = jnp.asarray(_mel_cepstrum_basis(nbands))

  loudness = jnp.sum(jnp.power(10.0, reference / 20), axis=0) / nbands
  loudness = 20 * jnp.log10(loudness)
  keep = loudness > threshold
  nsamp = jnp.sum(keep)

  if noise is not None:
    reference = reference + noise[0]
    distorted = distorted + noise[1]

  def cepstra(envelope):
    coefficients = basis.T @ jnp.where(keep, envelope, 0.0)
    mean = jnp.sum(coefficients, axis=1, keepdims=True) / jnp.maximum(nsamp, 1)
    return jnp.where(keep, coefficients - mean, 0.0)

  reference_cep = cepstra(reference)
  distorted_cep = cepstra(distorted)

  reference_sum = jnp.sum(reference_cep**2, axis=1)
  distorted_sum = jnp.sum(distorted_cep**2, axis=1)
  valid = (reference_sum >= _SMALL) & (distorted_sum >= _SMALL)
  correlations = jnp.where(
    valid,
    jnp.abs(jnp.sum(reference_cep * distorted_cep, axis=1))
    / jnp.sqrt(jnp.where(valid, reference_sum * distorted_sum, 1.0)),
    0.0,
  )

  average = jnp.sum(correlations[1:]) / (_CEPSTRAL_COEFFICIENTS - 1)
  # The reference bails out and returns zeros when nothing clears threshold.
  enough = nsamp > 1
  return jnp.where(enough, average, 0.0), jnp.where(enough, correlations, 0.0)


def spectrum_diff(reference_sl, processed_sl):
  """Loudness-normalised differences in the long-term spectrum and its slope.

  Returns:
    [sum abs, std, max] for the spectrum, the normalised spectrum and the slope.
  """
  nbands = reference_sl.shape[0]
  reference = jnp.power(10.0, reference_sl / 20)
  processed = jnp.power(10.0, processed_sl / 20)
  reference = reference / jnp.sum(reference)
  processed = processed / jnp.sum(processed)

  def statistics(difference):
    return jnp.stack(
      [
        jnp.sum(jnp.abs(difference)),
        nbands * jnp.std(difference),
        jnp.max(jnp.abs(difference)),
      ]
    )

  d_loud = statistics(reference - processed)
  d_norm = statistics((reference - processed) / (reference + processed))
  d_slope = statistics(jnp.diff(reference) - jnp.diff(processed))
  return d_loud, d_norm, d_slope


def bm_covary(
  reference_bm,
  processed_bm,
  segment_size=_SEGMENT_COVARIANCE,
  sample_rate=24000.0,
):
  """Normalised cross-covariance of the BM motion, per band and 16 ms segment.

  Returns:
    The cross-covariance and the two mean-square levels, each (nchan, nseg).
  """
  max_lag = int(np.around(1.0 * (0.001 * sample_rate)))
  nwin = int(np.around(segment_size * (0.001 * sample_rate)))
  nwin += nwin % 2 == 1
  nhalf = nwin // 2

  window = np.hanning(nwin)
  half_window = window[nhalf:nwin]

  def inverse_window_correlation(values):
    correlation = np.correlate(values, values, "full")
    start = len(values) - 1 - max_lag
    if start < 0:
      raise ValueError("segment size too small")
    return jnp.asarray(1 / correlation[start : max_lag + len(values)])

  window_correlation = inverse_window_correlation(window)
  half_correlation = inverse_window_correlation(half_window)
  window_power = 1.0 / np.sum(window**2)
  half_power = 1.0 / np.sum(half_window**2)

  npts = reference_bm.shape[1]
  starts, nseg = _segment_layout(npts, nwin)

  def covary(reference_seg, processed_seg, power, correction):
    """Cross-covariance of one windowed segment, over the trailing axis."""
    reference_seg = reference_seg - jnp.mean(
      reference_seg, axis=-1, keepdims=True
    )
    processed_seg = processed_seg - jnp.mean(
      processed_seg, axis=-1, keepdims=True
    )
    reference_ms = jnp.sum(reference_seg**2, axis=-1) * power
    processed_ms = jnp.sum(processed_seg**2, axis=-1) * power

    length = reference_seg.shape[-1]
    correlation = filters.correlate_full(reference_seg, processed_seg)
    correlation = correlation[length - 1 - max_lag : max_lag + length]
    peak = jnp.max(jnp.abs(correlation * correction))

    valid = (reference_ms > _SMALL) & (processed_ms > _SMALL)
    covariance = jnp.where(
      valid,
      peak / jnp.sqrt(jnp.where(valid, reference_ms * processed_ms, 1.0)),
      0.0,
    )
    return covariance, reference_ms, processed_ms

  over_channels = jax.vmap(covary, in_axes=(0, 0, None, None))
  over_segments = jax.vmap(
    jax.vmap(covary, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None)
  )

  first = over_channels(
    reference_bm[:, :nhalf] * window[nhalf:nwin],
    processed_bm[:, :nhalf] * window[nhalf:nwin],
    half_power,
    half_correlation,
  )
  middle = over_segments(
    _gather_segments(reference_bm, starts, nwin) * window,
    _gather_segments(processed_bm, starts, nwin) * window,
    window_power,
    window_correlation,
  )
  last_start = np.array([nhalf * (nseg - 1)])
  last = over_channels(
    _gather_segments(reference_bm, last_start, nhalf)[:, 0] * window[:nhalf],
    _gather_segments(processed_bm, last_start, nhalf)[:, 0] * window[:nhalf],
    half_power,
    half_correlation,
  )

  def stack(index):
    return jnp.concatenate(
      [first[index][:, None], middle[index], last[index][:, None]], axis=1
    )

  covariance = jnp.clip(stack(0), 0, 1)
  # The reference doubles the MS levels to match the dB SL envelope scaling.
  return covariance, 2.0 * stack(1), 2.0 * stack(2)


def ave_covary2(
  cross_covariance, reference_mean_square, threshold=_SILENCE_THRESHOLD
):
  """Average the cross-covariance over tiles above threshold.

  Returns:
    The unweighted average, and six averages weighted for progressive loss of
    IHC synchronisation at high frequencies.
  """
  n_channels = cross_covariance.shape[0]
  centre_freq = ear_model.center_frequency(n_channels)

  cutoff = (
    np.atleast_2d(_SYNC_CUTOFFS ** (2 * _SYNC_FILTER_ORDERS))
    .repeat(n_channels, axis=0)
    .T
  )
  frequency = centre_freq ** (
    2 * np.atleast_2d(_SYNC_FILTER_ORDERS).repeat(n_channels, axis=0).T
  )
  sync = jnp.asarray(np.sqrt(cutoff / (cutoff + frequency)))

  signal_rms = jnp.sqrt(reference_mean_square)
  loudness = jnp.sum(jnp.power(10.0, signal_rms / 20), axis=0) / n_channels
  loudness = 20 * jnp.log10(loudness)
  segment_keep = loudness > threshold
  nseg = jnp.sum(segment_keep)

  tile_keep = (signal_rms > threshold) & segment_keep
  weight = jnp.where(tile_keep, 1.0, 0.0)
  sync_weight = jnp.where(tile_keep[None, :, :], sync[:, :, None], 0.0)

  weighted_sum = jnp.sum(weight * cross_covariance)
  weight_sum = jnp.sum(weight)
  average = jnp.where(
    weight_sum >= 1, weighted_sum / jnp.maximum(weight_sum, 1), 0.0
  )

  sync_covariance = jnp.sum(
    sync_weight * cross_covariance[None, :, :], axis=(1, 2)
  ) / jnp.sum(sync_weight, axis=(1, 2))

  enough = nseg > 1
  return (
    jnp.where(enough, average, 0.0),
    jnp.where(enough, sync_covariance, jnp.zeros(6)),
  )


@functools.partial(
  jax.jit,
  static_argnames=(
    "silence_threshold",
    "segment_covariance",
    "sample_rate",
  ),
)
def _hasqi_backend(
  reference_db,
  processed_db,
  reference_bm,
  processed_bm,
  reference_sl,
  processed_sl,
  silence_threshold,
  segment_covariance,
  sample_rate,
):
  """Everything after the ear model, as one compiled region.

  Compiling the stages separately cost ~4.6 s per distinct shape and 95 ms of
  arithmetic. The stage boundaries are still functions; only the jit is
  merged.

  Returns:
    The cepstral correlation, BM sync5, and the two linear terms.
  """
  reference_smooth = env_smooth(reference_db, segment_covariance, sample_rate)
  processed_smooth = env_smooth(processed_db, segment_covariance, sample_rate)
  cepstral_correlation, _ = mel_cepstrum_correlation(
    reference_smooth, processed_smooth, silence_threshold
  )

  d_loud_stats, _, d_slope_stats = spectrum_diff(reference_sl, processed_sl)

  cross_covariance, reference_mean_square, _ = bm_covary(
    reference_bm, processed_bm, segment_covariance, sample_rate
  )
  _, sync_covariance = ave_covary2(
    cross_covariance, reference_mean_square, _SILENCE_THRESHOLD
  )

  d_loud = jnp.clip(1.0 - d_loud_stats[1] / _LOUDNESS_SCALE, 0.0, 1.0)
  d_slope = jnp.clip(1.0 - d_slope_stats[1], 0.0, 1.0)
  return cepstral_correlation, sync_covariance[4], d_loud, d_slope


@precision.in_float64
def hasqi_v2(
  reference,
  reference_rate,
  processed,
  processed_rate,
  hearing_loss,
  equalisation=1,
  level1=65.0,
  silence_threshold=_SILENCE_THRESHOLD,
  segment_covariance=_SEGMENT_COVARIANCE,
  reference_noise=None,
  processed_noise=None,
):
  """HASQI version 2 quality index.

  Runs in float64 regardless of the caller's JAX configuration, and restores
  that configuration afterwards; see jax_haspi_hasqi.precision for why the
  filter bank leaves no choice. Passing float32 inputs is fine and costs about
  2e-09 against a native float64 run.

  Args:
    reference: Clean reference signal.
    reference_rate: Its sampling rate in Hz.
    processed: Processed signal.
    processed_rate: Its sampling rate in Hz.
    hearing_loss: Levels in dB at [250, 500, 1000, 2000, 4000, 6000] Hz.
    equalisation: 1 to apply NAL-R to the reference here, 2 if it already has it.
    level1: dB SPL corresponding to an RMS of 1.
    silence_threshold: dB SL below which a time-frequency tile is ignored.
    segment_covariance: Segment size in ms.
    reference_noise: Threshold noise for the reference's BM motion.
    processed_noise: As above, for the processed signal.

  Returns:
    The combined score, its nonlinear and linear parts, and the four raw terms
    [cepstral correlation, BM sync5, d_loud, d_slope].
  """
  (
    reference_db,
    reference_bm,
    processed_db,
    processed_bm,
    reference_sl,
    processed_sl,
    sample_rate,
  ) = ear_model.ear_model(
    reference,
    reference_rate,
    processed,
    processed_rate,
    hearing_loss,
    equalisation,
    level1,
    reference_noise=reference_noise,
    processed_noise=processed_noise,
  )

  cepstral_correlation, bm_sync5, d_loud, d_slope = _hasqi_backend(
    reference_db,
    processed_db,
    reference_bm,
    processed_bm,
    reference_sl,
    processed_sl,
    silence_threshold=silence_threshold,
    segment_covariance=segment_covariance,
    sample_rate=sample_rate,
  )

  nonlinear = (cepstral_correlation**2) * bm_sync5
  linear = _LOUDNESS_WEIGHT * d_loud + _SLOPE_WEIGHT * d_slope
  return (
    nonlinear * linear,
    nonlinear,
    linear,
    jnp.stack([cepstral_correlation, bm_sync5, d_loud, d_slope]),
  )
