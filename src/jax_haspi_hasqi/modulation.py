"""HASPI v2 modulation front end, ported from clarity.evaluator.haspi.ebm."""

import jax
import jax.numpy as jnp
import numpy as np

_MODULATION_CENTRES = np.array([2, 6, 10, 16, 25, 40, 64, 100, 160, 256])
_FILTER_LENGTH_SECONDS = 0.24
_SMALL = 1e-30


def env_filter(
  reference_db, processed_db, filter_cutoff, freq_sub_sample, freq_samp
):
  """Low-pass and subsample the dB SL envelopes.

  Args:
    reference_db: Envelope per band, shape (nsamp, nbands) after transposing.
    processed_db: Same, for the processed signal.
    filter_cutoff: LP cutoff in Hz.
    freq_sub_sample: Output sampling rate in Hz.
    freq_samp: Input sampling rate in Hz.

  Returns:
    Both envelopes, low-passed and subsampled, shape (nsub, nbands).
  """
  if freq_sub_sample > freq_samp:
    raise ValueError("upsampling rate too high.")
  if filter_cutoff > 0.5 * freq_sub_sample:
    raise ValueError("LP cutoff frequency too high.")

  # The reference transposes whenever there are more columns than rows, so a
  # band-major input arrives here already oriented sample-major.
  if reference_db.shape[1] > reference_db.shape[0]:
    reference_db = reference_db.T
    processed_db = processed_db.T
  nsamp = reference_db.shape[0]

  # 0.7 is the reference's empirical adjustment to the -3 dB length.
  nfilt = round(0.001 * 0.7 * 1000 * (1 / filter_cutoff) * freq_samp)
  nhalf = nfilt // 2
  nfilt = 2 * nhalf

  # MATLAB's hanning() drops the two zero-weighted end samples, unlike numpy's.
  window = 0.5 * (
    1 - np.cos(2 * np.pi * np.arange(1, nfilt / 2 + 1) / (nfilt + 1))
  )
  taps = np.concatenate((window, np.flip(window)))
  taps = jnp.asarray(taps / np.sum(taps))

  def low_pass(envelope):
    padded = jnp.pad(envelope, ((len(taps) - 1, len(taps) - 1), (0, 0)))
    convolved = jnp.apply_along_axis(
      lambda column: jnp.convolve(column, taps, "valid"), 0, padded
    )
    return convolved[nhalf : nhalf + nsamp]

  index = np.arange(0, nsamp, int(freq_samp // freq_sub_sample))
  return low_pass(reference_db)[index], low_pass(processed_db)[index]


def _cepstrum_basis(nbands, nbasis):
  basis = np.cos(
    np.outer(np.arange(nbands), np.arange(nbasis)) * np.pi / (nbands - 1)
  )
  return basis / np.sqrt(np.sum(basis**2, axis=0))


def cepstral_correlation_coef(
  reference_db, processed_db, thresh_cep, thresh_nerve, nbasis, noise=None
):
  """Mel cepstra of the samples that clear the silence threshold.

  Numpy, not JAX: the gate keeps a data-dependent number of *time samples*,
  and the modulation filterbank then convolves along that axis, so a mask is
  not equivalent to the reference's selection.

  Args:
    reference_db: Subsampled reference envelope, shape (nsamp, nbands).
    processed_db: Same, for the processed signal.
    thresh_cep: Loudness in dB above which a sample is kept.
    thresh_nerve: RMS of the IHC firing dither, in dB.
    nbasis: Number of cepstral basis functions.
    noise: Dither for the two gated envelopes, each (nkept, nbands). The
      reference draws it from the unseeded global RNG after the gate, so it is
      shaped by the gate; None means no dither.

  Returns:
    Mean-removed cepstral coefficients for both signals, (nkept, nbasis).
  """
  reference_db = np.asarray(reference_db)
  processed_db = np.asarray(processed_db)
  nbands = reference_db.shape[1]
  basis = _cepstrum_basis(nbands, nbasis)

  loudness = np.sum(10 ** (reference_db / 20), axis=1) / nbands
  loudness = 20 * np.log10(loudness)
  keep = np.where(loudness > thresh_cep)[0]
  if len(keep) <= 1:
    raise ValueError("Signal below threshold")

  reference_db = reference_db[keep, :]
  processed_db = processed_db[keep, :]
  if noise is not None:
    reference_db = reference_db + thresh_nerve * np.asarray(noise[0])
    processed_db = processed_db + thresh_nerve * np.asarray(noise[1])

  reference_cep = reference_db @ basis
  processed_cep = processed_db @ basis
  reference_cep -= np.mean(reference_cep, axis=0, keepdims=True)
  processed_cep -= np.mean(processed_cep, axis=0, keepdims=True)
  return reference_cep, processed_cep


def _modulation_filters(freq_sub_sampling, centre_frequencies):
  """Band edges, filter taps and transient lengths for the modulation bank."""
  nmod = len(centre_frequencies)
  edge = np.zeros(nmod + 1)
  edge[0:3] = [0, 4, 8]
  for k in range(3, nmod + 1):
    edge[k] = (centre_frequencies[k - 1] ** 2) / edge[k - 1]

  nyquist = 0.5 * freq_sub_sampling
  edge = edge[edge < nyquist]
  nmod = len(edge) - 1
  centre_frequencies = centre_frequencies[:nmod]

  # Constant-Q above 10 Hz; the two lowest bands share the base length.
  lengths = np.full(nmod, _FILTER_LENGTH_SECONDS)
  lengths[2:nmod] = (
    _FILTER_LENGTH_SECONDS * centre_frequencies[2] / centre_frequencies[2:nmod]
  )
  nfir = 2 * np.floor(lengths * freq_sub_sampling / 2).astype(int)

  taps = []
  for length in nfir:
    coefficients = np.hanning(length + 1)
    taps.append(coefficients / np.sum(coefficients))
  return centre_frequencies, taps, nfir // 2, nyquist


def fir_modulation_filter(
  reference_envelope,
  processed_envelope,
  freq_sub_sampling,
  center_frequencies=None,
):
  """Split each cepstral coefficient into modulation-rate bands.

  Args:
    reference_envelope: Cepstral coefficients, shape (nsamp, nchan).
    processed_envelope: Same, for the processed signal.
    freq_sub_sampling: Envelope sampling rate in Hz.
    center_frequencies: Modulation band centres in Hz.

  Returns:
    Both filtered signals, shape (nchan, nmod, nsamp), and the band centres.
  """
  if center_frequencies is None:
    center_frequencies = _MODULATION_CENTRES
  nsamp, nchan = reference_envelope.shape
  center_frequencies, taps, transients, nyquist = _modulation_filters(
    freq_sub_sampling, center_frequencies
  )

  reference = jnp.asarray(reference_envelope)
  processed = jnp.asarray(processed_envelope)
  sample = np.arange(1, nsamp + 1)

  def band(envelope, cosine, sine, coefficients, start):
    """Demodulate to baseband, low-pass, then modulate back up."""
    demodulated = envelope * (cosine - 1j * sine)[:, None]
    filtered = jax.vmap(
      lambda column: jnp.convolve(column, coefficients, "full"),
      in_axes=1,
      out_axes=1,
    )(demodulated)
    filtered = filtered[start : start + nsamp]
    return (
      jnp.real(filtered) * cosine[:, None] - jnp.imag(filtered) * sine[:, None]
    )

  reference_bands = []
  processed_bands = []
  for index, centre in enumerate(center_frequencies):
    if index == 0:
      cosine = jnp.ones(nsamp)
      sine = jnp.zeros(nsamp)
    else:
      phase = np.pi * sample * centre / nyquist
      cosine = jnp.asarray(np.sqrt(2) * np.cos(phase))
      sine = jnp.asarray(np.sqrt(2) * np.sin(phase))

    coefficients = jnp.asarray(taps[index])
    reference_bands.append(
      band(reference, cosine, sine, coefficients, transients[index])
    )
    processed_bands.append(
      band(processed, cosine, sine, coefficients, transients[index])
    )

  return (
    jnp.stack(reference_bands, axis=1).transpose(2, 1, 0),
    jnp.stack(processed_bands, axis=1).transpose(2, 1, 0),
    center_frequencies,
  )


def modulation_cross_correlation(reference_modulation, processed_modulation):
  """Normalised cross-covariance per band, averaged over basis functions 2-6.

  Args:
    reference_modulation: Shape (nchan, nmod, nsamp).
    processed_modulation: Same shape.

  Returns:
    One correlation per modulation band.
  """
  reference = reference_modulation - jnp.mean(
    reference_modulation, axis=2, keepdims=True
  )
  processed = processed_modulation - jnp.mean(
    processed_modulation, axis=2, keepdims=True
  )

  reference_sum = jnp.sum(reference**2, axis=2)
  processed_sum = jnp.sum(processed**2, axis=2)
  valid = (reference_sum >= _SMALL) & (processed_sum >= _SMALL)
  covariance = jnp.where(
    valid,
    jnp.abs(jnp.sum(reference * processed, axis=2))
    / jnp.sqrt(jnp.where(valid, reference_sum * processed_sum, 1.0)),
    0.0,
  )
  return jnp.mean(covariance[1:6], axis=0)
