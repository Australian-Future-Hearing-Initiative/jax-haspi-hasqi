"""Cochlear model shared by HASPI and HASQI, ported from clarity.evaluator.haspi.eb."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from jax import lax

from jax_haspi_hasqi import filters
from jax_haspi_hasqi import nalr

_EAR_Q = 9.26449
_MIN_BANDWIDTH = 24.7
_SAMPLE_RATE = 24000.0
_SMALL = 1e-30
_THRESHOLD_HIGH = 100.0
_IHC_THRESHOLD = -10.0
_IHC_OVERSHOOT = 2.0
_NALR_TAPS = 140
_AUDIOMETRIC_FREQUENCIES = np.array(
  [250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0]
)


def center_frequency(nchan=32, shift=None, low_freq=80.0, high_freq=8000.0):
  """ERB-spaced gammatone centre frequencies, from Slaney's Apple TR #35.

  Args:
    nchan: Number of filters.
    shift: Fractional basal shift along the basilar membrane. HASPI passes
      None; see the note in ear_model.
    low_freq: Lowest centre frequency in Hz.
    high_freq: Highest centre frequency in Hz.
  """
  if shift is not None:
    k = 1
    a = 2.1
    scale = 165.4
    x_low = (1 / a) * np.log10(k + (low_freq / scale)) * (1 + shift)
    x_high = (1 / a) * np.log10(k + (high_freq / scale)) * (1 + shift)
    low_freq = scale * (10 ** (a * x_low) - k)
    high_freq = scale * (10 ** (a * x_high) - k)

  overall = _EAR_Q * _MIN_BANDWIDTH
  spacing = (
    np.arange(1, nchan)
    * (-np.log(high_freq + overall) + np.log(low_freq + overall))
    / (nchan - 1)
  )
  frequencies = -overall + np.exp(spacing) * (high_freq + overall)
  return np.flip(np.insert(frequencies, 0, high_freq))


def loss_parameters(hearing_loss, centre_freq):
  """Split the audiogram between outer and inner hair cells.

  Args:
    hearing_loss: Levels in dB at the 6 audiometric frequencies.
    centre_freq: Gammatone centre frequencies, low to high.

  Returns:
    OHC attenuation, filter bandwidth, compression kneepoint, compression
    ratio and IHC attenuation, one per channel.
  """
  hearing_loss = jnp.asarray(hearing_loss)
  nfilt = len(centre_freq)

  knots = np.concatenate(
    ([centre_freq[0]], _AUDIOMETRIC_FREQUENCIES, [centre_freq[-1]])
  )
  knot_loss = jnp.concatenate(
    (hearing_loss[:1], hearing_loss, hearing_loss[-1:])
  )
  loss = jnp.maximum(
    jnp.interp(jnp.asarray(centre_freq), jnp.asarray(knots), knot_loss), 0.0
  )

  compression_ratio = 1.25 + 2.25 * np.arange(nfilt) / (nfilt - 1)
  max_ohc = 70 * (1 - (1 / compression_ratio))
  theoretical_ohc = jnp.asarray(1.25 * max_ohc)

  severe = loss >= theoretical_ohc
  attn_ohc = jnp.where(severe, 0.8 * theoretical_ohc, 0.8 * loss)
  attn_ihc = jnp.where(
    severe, 0.2 * theoretical_ohc + (loss - theoretical_ohc), 0.2 * loss
  )

  bandwidth = 1.0 + (attn_ohc / 50.0) + 2.0 * (attn_ohc / 50.0) ** 6
  low_knee = attn_ohc + 30
  upamp = 30 + (70 / compression_ratio)
  compression_ratio = (100 - low_knee) / (upamp + attn_ohc - low_knee)
  return attn_ohc, bandwidth, low_knee, compression_ratio, attn_ihc


def resample_24khz(signal, sample_rate):
  """Resample to 24 kHz, matching RMS in the retained band.

  The reference rounds both rates to the nearest kHz before comparing them, so
  22050 Hz is treated as 22 kHz.
  """
  target_khz = int(np.round(_SAMPLE_RATE / 1000))
  source_khz = int(np.round(sample_rate / 1000))
  if source_khz == target_khz:
    return signal

  resampled = filters.resample_poly(signal, target_khz, source_khz)
  if source_khz < target_khz:
    ratio = jnp.sqrt(jnp.mean(signal**2)) / jnp.sqrt(jnp.mean(resampled**2))
    return ratio * resampled

  # Downsampling: compare RMS inside the 21 kHz band both filters pass.
  order, attenuation = 7, 30
  source_b, source_a = scipy.signal.cheby2(order, attenuation, 21 / source_khz)
  target_b, target_a = scipy.signal.cheby2(order, attenuation, 21 / target_khz)
  source_band = filters.lfilter(source_b, source_a, signal)
  target_band = filters.lfilter(target_b, target_a, resampled)
  ratio = jnp.sqrt(jnp.mean(source_band**2)) / jnp.sqrt(
    jnp.mean(target_band**2)
  )
  return ratio * resampled


def input_align(reference, processed):
  """Align the processed signal to the reference and prune leading silence.

  Numpy rather than JAX: the output length depends on where the reference
  crosses its own amplitude threshold, so the shape is data-dependent.

  Returns:
    The aligned pair, and the (start, stop, delay) the alignment chose.
  """
  reference = np.asarray(reference)
  processed = np.asarray(processed)
  processed_n = len(processed)
  min_length = min(len(reference), processed_n)

  head_reference = reference[:min_length] - np.mean(reference[:min_length])
  head_processed = processed[:min_length] - np.mean(processed[:min_length])
  correlation = scipy.signal.correlate(head_reference, head_processed, "full")
  delay = min_length - np.argmax(np.abs(correlation)) - 1
  # Back up 2 ms to allow for dispersion.
  delay = int(np.rint(delay - 2 * _SAMPLE_RATE / 1000.0))

  if delay > 0:
    processed = np.concatenate((processed[delay:processed_n], np.zeros(delay)))
  else:
    processed = np.concatenate(
      (np.zeros(-delay), processed[: processed_n + delay])
    )

  above = np.where(np.abs(reference) > 0.001 * np.max(np.abs(reference)))[0]
  start = above[0]
  stop = min(above[-1], processed_n)
  return (
    reference[start : stop + 1],
    processed[start : stop + 1],
    (start, stop, delay),
  )


def middle_ear(signal, sample_rate=_SAMPLE_RATE):
  """2-pole 350 Hz high-pass in series with a 1-pole 5 kHz low-pass."""
  low_b, low_a = scipy.signal.butter(1, 5000 / (0.5 * sample_rate))
  high_b, high_a = scipy.signal.butter(2, 350 / (0.5 * sample_rate), "high")
  return filters.lfilter(high_b, high_a, filters.lfilter(low_b, low_a, signal))


def _gammatone_coefficients(bandwidth, centre_freq, sample_rate):
  erb = _MIN_BANDWIDTH + (centre_freq / _EAR_Q)
  tpt = 2 * np.pi / sample_rate
  a = jnp.exp(-bandwidth * tpt * erb * 1.019)
  a_1, a_2, a_3, a_4, a_5 = (
    4.0 * a,
    -6.0 * a * a,
    4.0 * a**3,
    -(a**4),
    4.0 * a * a,
  )
  gain = 2.0 * (1 - a_1 - a_2 - a_3 - a_4) / (1 + a_1 + a_5)
  numerator = jnp.stack([jnp.ones_like(a), a_1, a_5])
  denominator = jnp.stack([jnp.ones_like(a), -a_1, -a_2, -a_3, -a_4])
  return numerator, denominator, gain


def _carrier(npts, centre_freq, sample_rate):
  """Demodulation carrier.

  The reference generates this by iterated 2-D rotation; evaluating the
  trigonometry directly agrees to ~1e-11 over the lengths used here and is
  what the vectorised form wants.
  """
  theta = (2 * np.pi / sample_rate) * centre_freq * jnp.arange(npts)
  return -jnp.sin(theta), jnp.cos(theta)


def _safe_sqrt(x):
  """sqrt with a zero, rather than NaN, derivative at zero.

  A silent sample gives an exactly zero envelope, where sqrt's derivative is
  infinite. The value is unchanged; only the gradient differs.
  """
  positive = x > 0
  return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def gammatone_basilar_membrane(
  reference,
  reference_bandwidth,
  processed,
  processed_bandwidth,
  centre_freq,
  sample_rate=_SAMPLE_RATE,
):
  """4th-order gammatone filter applied to both signals at one centre frequency.

  Returns:
    Envelope and basilar-membrane motion for each of the two signals.
  """
  sincf, coscf = _carrier(reference.shape[0], centre_freq, sample_rate)

  def one(signal, bandwidth):
    numerator, denominator, gain = _gammatone_coefficients(
      bandwidth, centre_freq, sample_rate
    )
    real = filters.lfilter(numerator, denominator, signal * coscf)
    imaginary = filters.lfilter(numerator, denominator, signal * sincf)
    motion = gain * (real * coscf + imaginary * sincf)
    envelope = gain * _safe_sqrt(real * real + imaginary * imaginary)
    return envelope, motion

  reference_envelope, reference_motion = one(reference, reference_bandwidth)
  processed_envelope, processed_motion = one(processed, processed_bandwidth)
  return (
    reference_envelope,
    reference_motion,
    processed_envelope,
    processed_motion,
  )


def bandwidth_adjust(control, bandwidth_min, bandwidth_max, level1):
  """Widen the auditory filter at high signal levels.

  The reference branches at 50 and 100 dB SPL; the clip is the same function.
  """
  control_db = 20 * jnp.log10(jnp.sqrt(jnp.mean(control**2))) + level1
  fraction = jnp.clip((control_db - 50) / 50, 0.0, 1.0)
  return bandwidth_min + fraction * (bandwidth_max - bandwidth_min)


def env_compress_basilar_membrane(
  envelope,
  motion,
  control,
  attn_ohc,
  threshold_low,
  compression_ratio,
  level1,
  sample_rate=_SAMPLE_RATE,
):
  """Apply cochlear compression to one band's envelope and BM motion."""
  log_envelope = level1 + 20 * jnp.log10(jnp.maximum(control, _SMALL))
  log_envelope = jnp.maximum(
    jnp.minimum(log_envelope, _THRESHOLD_HIGH), threshold_low
  )
  gain = -attn_ohc - (log_envelope - threshold_low) * (
    1 - (1 / compression_ratio)
  )

  gain = jnp.power(10.0, gain / 20)
  low_b, low_a = scipy.signal.butter(1, 800 / (0.5 * sample_rate))
  gain = filters.lfilter(low_b, low_a, gain)
  return gain * envelope, gain * motion


def envelope_align(reference, output, sample_rate=_SAMPLE_RATE, corr_range=100):
  """Shift the output to the reference's envelope peak, over a +/-100 ms window."""
  npts = reference.shape[0]
  lags = min(int(np.rint(0.001 * corr_range * sample_rate)), npts)

  correlation = filters.correlate_full(reference, output)
  # The reference slices past the end when lags == npts; numpy clips, so do
  # the same rather than reading out of bounds.
  start, stop = npts - lags, min(npts + lags, 2 * npts - 1)
  location = jnp.argmax(
    lax.dynamic_slice(correlation, (start,), (stop - start,))
  )
  return filters.shift_with_zeros(output, lags - location - 1)


def envelope_sl(envelope, motion, attn_ihc, level1):
  """Convert a compressed envelope to dB SL, scaling the BM motion to match."""
  in_db = jnp.maximum(level1 - attn_ihc + 20 * jnp.log10(envelope + _SMALL), 0)
  return in_db, ((in_db + _SMALL) / (envelope + _SMALL)) * motion


def inner_hair_cell_adaptation(
  envelope_db, motion, delta=_IHC_OVERSHOOT, sample_rate=_SAMPLE_RATE
):
  """Rapid and short-term IHC adaptation, as a two-state RC circuit.

  The reference iterates sample by sample; the recursion is linear and
  time-invariant, so a scan reproduces it exactly.
  """
  delta = max(delta, 1.0001)
  tau1, tau2 = 0.002, 0.060
  r_1 = 1 / delta
  r_2 = r_3 = 0.5 * (1 - r_1)
  c_1 = tau1 * (r_1 + r_2) / (r_1 * r_2)
  c_2 = tau2 / ((r_1 + r_2) * r_3)

  a11 = r_1 + r_2 + r_1 * r_2 * c_1 * sample_rate
  a12, a21 = -r_1, -r_3
  a22 = r_2 + r_3 + r_2 * r_3 * c_2 * sample_rate
  denominator = 1 / ((a11 * a22) - (a21 * a12))
  product_1 = r_1 * r_2 * c_1 * sample_rate
  product_2 = r_2 * r_3 * c_2 * sample_rate

  def step(state, v_0):
    v_1, v_2 = state
    b_1 = v_0 * r_2 + product_1 * v_1
    b_2 = product_2 * v_2
    v_1 = denominator * (a22 * b_1 - a12 * b_2)
    v_2 = denominator * (-a21 * b_1 + a11 * b_2)
    return (v_1, v_2), (v_0 - v_1) / r_1

  zero = jnp.zeros((), envelope_db.dtype)
  _, out = lax.scan(step, (zero, zero), envelope_db)
  out = jnp.maximum(out, 0)
  gain = (out + _SMALL) / (envelope_db + _SMALL)
  return out, gain * motion


def basilar_membrane_add_noise(motion, noise, level1, threshold=_IHC_THRESHOLD):
  """Add the threshold noise to the BM motion.

  Args:
    motion: BM motion per channel.
    noise: Unit-variance noise of the same shape. The reference draws this
      from the unseeded global RNG; here the caller supplies it.
    level1: dB SPL corresponding to RMS 1.
    threshold: Noise level in dB re auditory threshold.
  """
  return motion + jnp.power(10.0, (threshold - level1) / 20) * noise


def group_delay_compensate(
  signal, bandwidths, centre_freq, sample_rate=_SAMPLE_RATE
):
  """Align the bands to a common group delay."""
  erb = _MIN_BANDWIDTH + (jnp.asarray(centre_freq) / _EAR_Q)
  tpt = 2 * np.pi / sample_rate
  a = jnp.exp(-tpt * 1.019 * bandwidths * erb)
  a_1, a_2, a_3, a_4, a_5 = (
    4.0 * a,
    -6.0 * a * a,
    4.0 * a**3,
    -(a**4),
    4.0 * a * a,
  )

  numerator = jnp.stack([jnp.ones_like(a), a_1, a_5], axis=-1)
  denominator = jnp.stack([jnp.ones_like(a), -a_1, -a_2, -a_3, -a_4], axis=-1)
  delays = jax.vmap(filters.group_delay_at_dc)(numerator, denominator)
  delays = jnp.round(delays)
  delays = delays - jnp.min(delays)
  correct = (jnp.max(delays) - delays).astype(jnp.int32)
  return jax.vmap(lambda row, shift: filters.shift_with_zeros(row, -shift))(
    signal, correct
  )


def convert_rms_to_sl(
  average, control, attn_ohc, threshold_low, compression_ratio, attn_ihc, level1
):
  """Convert a band's RMS gammatone output to dB SL."""
  control_db = level1 + 20 * jnp.log10(jnp.maximum(control, _SMALL))
  control_db = jnp.maximum(
    jnp.minimum(control_db, _THRESHOLD_HIGH), threshold_low
  )
  gain = -attn_ohc - (control_db - threshold_low) * (
    1 - (1 / compression_ratio)
  )

  signal_db = jnp.maximum(
    level1 + 20 * jnp.log10(jnp.maximum(average, _SMALL)), 0
  )
  return jnp.maximum(signal_db + gain - attn_ihc, 0)


@functools.partial(jax.jit, static_argnames=("nchan", "itype", "m_delay"))
def _ear_model_aligned(
  reference,
  processed,
  hearing_loss,
  itype,
  level1,
  reference_noise,
  processed_noise,
  nchan,
  m_delay,
):
  """The fixed-shape part of ear_model, once alignment has been resolved."""
  centre_freq = center_frequency(nchan)
  attn_ohc_y, bandwidth_min_y, low_knee_y, ratio_y, attn_ihc_y = (
    loss_parameters(hearing_loss, centre_freq)
  )
  # Intelligibility scores the reference ear as normal-hearing; quality does not.
  loss_x = jnp.zeros_like(hearing_loss) if itype == 0 else hearing_loss
  attn_ohc_x, bandwidth_min_x, low_knee_x, ratio_x, attn_ihc_x = (
    loss_parameters(loss_x, centre_freq)
  )

  # HASPI passes shift=None: the 0.02 basal shift its own docstring describes
  # is never applied. Preserved deliberately; see docs/porting_notes.md.
  centre_freq_control = center_frequency(nchan)
  _, bandwidth_1, _, _, _ = loss_parameters(
    np.full(6, 100.0), centre_freq_control
  )
  centre_freq = jnp.asarray(centre_freq)
  centre_freq_control = jnp.asarray(centre_freq_control)

  nsamp = reference.shape[0]
  if itype == 1:
    fir, _ = nalr.build(hearing_loss, _NALR_TAPS, _SAMPLE_RATE)
    reference = nalr.apply(fir, reference)[_NALR_TAPS : _NALR_TAPS + nsamp]

  reference_mid = middle_ear(reference)
  processed_mid = middle_ear(processed)

  def channel(index):
    reference_control, _, processed_control, _ = gammatone_basilar_membrane(
      reference_mid,
      bandwidth_1[index],
      processed_mid,
      bandwidth_1[index],
      centre_freq_control[index],
    )
    reference_bandwidth = bandwidth_adjust(
      reference_control, bandwidth_min_x[index], bandwidth_1[index], level1
    )
    processed_bandwidth = bandwidth_adjust(
      processed_control, bandwidth_min_y[index], bandwidth_1[index], level1
    )

    xenv, xbm, yenv, ybm = gammatone_basilar_membrane(
      reference_mid,
      reference_bandwidth,
      processed_mid,
      processed_bandwidth,
      centre_freq[index],
    )

    reference_compressed, reference_motion = env_compress_basilar_membrane(
      xenv,
      xbm,
      reference_control,
      attn_ohc_x[index],
      low_knee_x[index],
      ratio_x[index],
      level1,
    )
    processed_compressed, processed_motion = env_compress_basilar_membrane(
      yenv,
      ybm,
      processed_control,
      attn_ohc_y[index],
      low_knee_y[index],
      ratio_y[index],
      level1,
    )
    processed_compressed = envelope_align(
      reference_compressed, processed_compressed
    )
    processed_motion = envelope_align(reference_motion, processed_motion)

    reference_compressed, reference_motion = envelope_sl(
      reference_compressed, reference_motion, attn_ihc_x[index], level1
    )
    processed_compressed, processed_motion = envelope_sl(
      processed_compressed, processed_motion, attn_ihc_y[index], level1
    )

    reference_db, reference_motion = inner_hair_cell_adaptation(
      reference_compressed, reference_motion
    )
    processed_db, processed_motion = inner_hair_cell_adaptation(
      processed_compressed, processed_motion
    )
    return (
      reference_db,
      reference_motion,
      processed_db,
      processed_motion,
      jnp.sqrt(jnp.mean(xenv**2)),
      jnp.sqrt(jnp.mean(yenv**2)),
      jnp.sqrt(jnp.mean(reference_control**2)),
      jnp.sqrt(jnp.mean(processed_control**2)),
      reference_bandwidth,
      processed_bandwidth,
    )

  (
    reference_db,
    reference_b,
    processed_db,
    processed_b,
    reference_average,
    processed_average,
    reference_control_average,
    processed_control_average,
    reference_bandwidth,
    _,
  ) = jax.vmap(channel)(jnp.arange(nchan))

  reference_basilar_membrane = basilar_membrane_add_noise(
    reference_b, reference_noise, level1
  )
  processed_basilar_membrane = basilar_membrane_add_noise(
    processed_b, processed_noise, level1
  )

  if m_delay > 0:
    # The reference compensates the processed signal with the *reference*
    # bandwidths. Preserved; see docs/porting_notes.md.
    compensate = functools.partial(
      group_delay_compensate,
      bandwidths=reference_bandwidth,
      centre_freq=centre_freq,
    )
    reference_db = compensate(reference_db)
    processed_db = compensate(processed_db)
    reference_basilar_membrane = compensate(reference_basilar_membrane)
    processed_basilar_membrane = compensate(processed_basilar_membrane)

  reference_sl = convert_rms_to_sl(
    reference_average,
    reference_control_average,
    attn_ohc_x,
    low_knee_x,
    ratio_x,
    attn_ihc_x,
    level1,
  )
  processed_sl = convert_rms_to_sl(
    processed_average,
    processed_control_average,
    attn_ohc_y,
    low_knee_y,
    ratio_y,
    attn_ihc_y,
    level1,
  )
  return (
    reference_db,
    reference_basilar_membrane,
    processed_db,
    processed_basilar_membrane,
    reference_sl,
    processed_sl,
  )


def ear_model(
  reference,
  reference_rate,
  processed,
  processed_rate,
  hearing_loss,
  itype,
  level1,
  nchan=32,
  m_delay=1,
  reference_noise=None,
  processed_noise=None,
):
  """Middle ear, gammatone filter bank, OHC compression and IHC attenuation.

  Args:
    reference: Reference signal.
    reference_rate: Its sampling rate in Hz.
    processed: Processed signal, assumed to have equal or greater group delay.
    processed_rate: Its sampling rate in Hz.
    hearing_loss: Levels in dB at [250, 500, 1000, 2000, 4000, 6000] Hz.
    itype: 0 intelligibility, 1 quality with NAL-R added here, 2 quality with
      NAL-R already applied to the reference.
    level1: dB SPL corresponding to an RMS of 1.
    nchan: Number of auditory bands.
    m_delay: Compensate the filter bank's inter-channel group delay.
    reference_noise: Unit-variance threshold noise, shape (nchan, nsamp).
      Defaults to zeros, which makes the model deterministic; the reference
      draws from the unseeded global RNG instead.
    processed_noise: As above, for the processed signal.

  Returns:
    Envelope and BM motion per band for each signal, the long-term dB SL
    spectra, and the model sampling rate.
  """
  # Cast rather than asarray: enabling x64 does not promote an array that is
  # already float32, so a float32 caller would otherwise run the filter bank
  # in single precision and get NaN out of HASQI. The cast costs ~4e-08 on the
  # score, which is what float32 input data is worth; computing in float32
  # costs everything.
  reference_24k = resample_24khz(
    jnp.asarray(reference, jnp.float64), reference_rate
  )
  processed_24k = resample_24khz(
    jnp.asarray(processed, jnp.float64), processed_rate
  )

  shortest = min(reference_24k.shape[0], processed_24k.shape[0])
  reference_24k, processed_24k, _ = input_align(
    reference_24k[:shortest], processed_24k[:shortest]
  )
  reference_24k = jnp.asarray(reference_24k, jnp.float64)
  processed_24k = jnp.asarray(processed_24k, jnp.float64)

  shape = (nchan, reference_24k.shape[0])
  if reference_noise is None:
    reference_noise = jnp.zeros(shape)
  if processed_noise is None:
    processed_noise = jnp.zeros(shape)

  outputs = _ear_model_aligned(
    reference_24k,
    processed_24k,
    jnp.asarray(hearing_loss, jnp.float64),
    itype,
    level1,
    jnp.asarray(reference_noise),
    jnp.asarray(processed_noise),
    nchan,
    m_delay,
  )
  return (*outputs, _SAMPLE_RATE)
