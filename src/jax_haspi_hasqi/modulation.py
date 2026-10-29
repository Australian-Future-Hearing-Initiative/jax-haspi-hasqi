"""HASPI v2 modulation front end, ported from clarity.evaluator.haspi.ebm."""

import jax
import jax.numpy as jnp
import numpy as np

_MODULATION_CENTRES = np.array([2, 6, 10, 16, 25, 40, 64, 100, 160, 256])
_FILTER_LENGTH_SECONDS = 0.24
_SMALL = 1e-30


def env_filter(
  reference_db,
  processed_db,
  filter_cutoff,
  freq_sub_sample,
  freq_samp,
  valid_len=None,
):
  """Low-pass and subsample the dB SL envelopes.

  Args:
    reference_db: Envelope per band, shape (nsamp, nbands) after transposing.
    processed_db: Same, for the processed signal.
    filter_cutoff: LP cutoff in Hz.
    freq_sub_sample: Output sampling rate in Hz.
    freq_samp: Input sampling rate in Hz.
    valid_len: Optional active sample count before subsampling.

  Returns:
    Both envelopes, low-passed and subsampled, shape (nsub, nbands).
  """
  if freq_sub_sample > freq_samp:
    raise ValueError("upsampling rate too high.")
  if filter_cutoff > 0.5 * freq_sub_sample:
    raise ValueError("LP cutoff frequency too high.")

  dtype = jnp.result_type(reference_db, processed_db, jnp.float32)
  reference_db = jnp.asarray(reference_db, dtype=dtype)
  processed_db = jnp.asarray(processed_db, dtype=dtype)

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
  taps = jnp.asarray(taps / np.sum(taps), dtype=dtype)

  def low_pass(envelope):
    padded = jnp.pad(envelope, ((len(taps) - 1, len(taps) - 1), (0, 0)))
    convolved = jax.vmap(
      lambda column: jnp.convolve(column, taps, "valid"),
      in_axes=1,
      out_axes=1,
    )(padded)
    return convolved[nhalf : nhalf + nsamp]

  step = int(freq_samp // freq_sub_sample)
  index = jnp.arange(0, nsamp, step)
  reference_lp = low_pass(reference_db)[index]
  processed_lp = low_pass(processed_db)[index]
  if valid_len is not None:
    valid_sub = (index < valid_len)[:, None]
    zero = jnp.zeros((), dtype=dtype)
    reference_lp = jnp.where(valid_sub, reference_lp, zero)
    processed_lp = jnp.where(valid_sub, processed_lp, zero)
  return reference_lp, processed_lp


def _cepstrum_basis(nbands, nbasis):
  basis = np.cos(
    np.outer(np.arange(nbands), np.arange(nbasis)) * np.pi / (nbands - 1)
  )
  return basis / np.sqrt(np.sum(basis**2, axis=0))


def cepstral_correlation_coef(
  reference_db, processed_db, thresh_cep, thresh_nerve, nbasis, noise=None
):
  """Mel cepstra of the samples that clear the silence threshold.

  Compacts the ``nkept`` active frames into ``[:nkept]`` and zero-pads the
  remaining ``[nkept:nsamp]`` slots so the output shape stays fixed across
  clips for XLA compilation.

  Args:
    reference_db: Subsampled reference envelope, shape (nsamp, nbands).
    processed_db: Same, for the processed signal.
    thresh_cep: Loudness in dB above which a sample is kept.
    thresh_nerve: RMS of the IHC firing dither, in dB.
    nbasis: Number of cepstral basis functions.
    noise: Dither for the two gated envelopes, each (nkept, nbands) or
      (nsamp, nbands); None means no dither.

  Returns:
    Mean-removed cepstral coefficients for both signals, each (nsamp, nbasis),
    and ``nkept`` (the number of active frames in ``[:nkept]``).
  """
  dtype = jnp.result_type(reference_db, processed_db, jnp.float32)
  reference_db = jnp.asarray(reference_db, dtype=dtype)
  processed_db = jnp.asarray(processed_db, dtype=dtype)
  nsamp, nbands = reference_db.shape
  basis = jnp.asarray(_cepstrum_basis(nbands, nbasis), dtype=dtype)

  loudness = (
    jnp.sum(jnp.power(dtype.type(10.0), reference_db / 20), axis=1) / nbands
  )
  loudness = 20 * jnp.log10(jnp.maximum(loudness, dtype.type(_SMALL)))
  keep_mask = loudness > thresh_cep
  nkept = jnp.sum(keep_mask)
  enough = nkept > 1
  active = jnp.where(enough, nkept, 0)

  order = jnp.argsort(~keep_mask, stable=True)
  valid_pos = (jnp.arange(nsamp) < active)[:, None]
  zero = jnp.zeros((), dtype=dtype)
  reference_db = jnp.where(valid_pos, reference_db[order], zero)
  processed_db = jnp.where(valid_pos, processed_db[order], zero)

  if noise is not None:

    def _pad_noise(n):
      n = jnp.asarray(n, dtype=dtype)
      pad = max(0, nsamp - n.shape[0])
      return jnp.pad(n[:nsamp], ((0, pad), (0, 0)))

    dither = dtype.type(thresh_nerve)
    reference_db = reference_db + jnp.where(
      valid_pos, dither * _pad_noise(noise[0]), zero
    )
    processed_db = processed_db + jnp.where(
      valid_pos, dither * _pad_noise(noise[1]), zero
    )

  denom = jnp.maximum(active, 1).astype(dtype)
  reference_cep = reference_db @ basis
  processed_cep = processed_db @ basis
  ref_mean = jnp.sum(reference_cep, axis=0, keepdims=True) / denom
  proc_mean = jnp.sum(processed_cep, axis=0, keepdims=True) / denom
  reference_cep = jnp.where(valid_pos, reference_cep - ref_mean, zero)
  processed_cep = jnp.where(valid_pos, processed_cep - proc_mean, zero)
  return reference_cep, processed_cep, active


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

  dtype = jnp.result_type(reference_envelope, processed_envelope, jnp.float32)
  reference = jnp.asarray(reference_envelope, dtype=dtype)
  processed = jnp.asarray(processed_envelope, dtype=dtype)
  sample = np.arange(1, nsamp + 1)

  def band(envelope, cosine, sine, coefficients, start):
    """Demodulate to baseband, low-pass, then modulate back up."""
    demod_real = envelope * cosine[:, None]
    demod_imag = -envelope * sine[:, None]
    convolve = jax.vmap(
      lambda column: jnp.convolve(column, coefficients, "full"),
      in_axes=1,
      out_axes=1,
    )
    filt_real = convolve(demod_real)[start : start + nsamp]
    filt_imag = convolve(demod_imag)[start : start + nsamp]
    return filt_real * cosine[:, None] - filt_imag * sine[:, None]

  reference_bands = []
  processed_bands = []
  for index, centre in enumerate(center_frequencies):
    if index == 0:
      cosine = jnp.ones(nsamp, dtype=dtype)
      sine = jnp.zeros(nsamp, dtype=dtype)
    else:
      phase = np.pi * sample * centre / nyquist
      cosine = jnp.asarray(np.sqrt(2) * np.cos(phase), dtype=dtype)
      sine = jnp.asarray(np.sqrt(2) * np.sin(phase), dtype=dtype)

    coefficients = jnp.asarray(taps[index], dtype=dtype)
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


def modulation_cross_correlation(
  reference_modulation, processed_modulation, nkept=None
):
  """Normalised cross-covariance per band, averaged over basis functions 2-6.

  Args:
    reference_modulation: Shape (nchan, nmod, nsamp).
    processed_modulation: Same shape.
    nkept: Optional active frame count in ``[0, nsamp]``.

  Returns:
    One correlation per modulation band.
  """
  dtype = jnp.result_type(
    reference_modulation, processed_modulation, jnp.float32
  )
  reference_modulation = jnp.asarray(reference_modulation, dtype=dtype)
  processed_modulation = jnp.asarray(processed_modulation, dtype=dtype)
  nsamp = reference_modulation.shape[2]
  if nkept is None:
    nkept = nsamp
  enough = nkept > 1
  active = jnp.where(enough, nkept, 0)
  denom = jnp.maximum(active, 1).astype(dtype)
  mask = (jnp.arange(nsamp) < active)[None, None, :]
  zero = jnp.zeros((), dtype=dtype)

  ref_masked = jnp.where(mask, reference_modulation, zero)
  proc_masked = jnp.where(mask, processed_modulation, zero)
  ref_mean = jnp.sum(ref_masked, axis=2, keepdims=True) / denom
  proc_mean = jnp.sum(proc_masked, axis=2, keepdims=True) / denom
  reference = jnp.where(mask, ref_masked - ref_mean, zero)
  processed = jnp.where(mask, proc_masked - proc_mean, zero)

  reference_sum = jnp.sum(reference**2, axis=2)
  processed_sum = jnp.sum(processed**2, axis=2)
  valid = (reference_sum >= _SMALL) & (processed_sum >= _SMALL)
  one = jnp.ones((), dtype=dtype)
  covariance = jnp.where(
    valid,
    jnp.abs(jnp.sum(reference * processed, axis=2))
    / jnp.sqrt(jnp.where(valid, reference_sum * processed_sum, one)),
    zero,
  )
  return jnp.where(enough, jnp.mean(covariance[1:6], axis=0), zero)
