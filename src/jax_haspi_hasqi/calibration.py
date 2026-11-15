"""Opt-in RMS-to-SPL calibration and 24 kHz NAL-R equalisation helpers.

``haspi_v2`` and ``hasqi_v2`` follow ``pyclarity==0.9.0``: an RMS of 1.0 in the
reference signal maps to ``level1`` dB SPL (default 65 dB SPL), and
``hasqi_v2(..., equalisation=1)`` applies 24 kHz NAL-R equalisation only to the
reference. Standard ``[-1, 1]`` speech sits around -20 to -25 dBFS (RMS
~0.06-0.10), which silently evaluates at ~40-45 dB SPL unless calibrated.

These helpers leave the core metric signatures and default outputs unchanged
while providing composable preprocessing for calibrated unaided and aided
evaluations.
"""

import jax.numpy as jnp

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import nalr

MODEL_SAMPLE_RATE = ear_model._SAMPLE_RATE
_NALR_TAPS = ear_model._NALR_TAPS


def scale_to_spl(reference, processed, target_spl=65.0, level1=65.0):
  """Jointly scale ``(reference, processed)`` to a target reference SPL.

  Multiplies both signals by the same scalar gain so ``rms(reference)`` maps to
  ``target_spl`` dB SPL under the ear model's ``level1`` convention (where RMS
  1.0 equals ``level1`` dB SPL). At the default ``target_spl == level1``,
  ``rms(reference) == 1.0``. Relative gain and SNR between ``processed`` and
  ``reference`` are preserved.

  If ``reference`` is silent (RMS == 0) or non-finite, both signals are
  returned unchanged (cast to the resolved float dtype) without NaN or Inf.

  Args:
    reference: Clean reference waveform.
    processed: Processed waveform (need not match ``reference``'s length).
    target_spl: Desired presentation level of ``reference`` in dB SPL.
    level1: dB SPL corresponding to an RMS of 1.0 in the downstream metric call.

  Returns:
    ``(scaled_reference, scaled_processed)`` in the resolved floating-point
    dtype (``float32`` by default, or ``float64`` under ``jax_enable_x64`` with
    ``float64`` inputs).
  """
  reference = jnp.asarray(reference)
  processed = jnp.asarray(processed)
  dtype = jnp.result_type(reference, processed, jnp.float32)
  reference = reference.astype(dtype)
  processed = processed.astype(dtype)

  rms = ear_model._safe_sqrt(jnp.mean(reference**2))
  target_rms = jnp.power(
    jnp.asarray(10.0, dtype=dtype),
    (jnp.asarray(target_spl, dtype=dtype) - jnp.asarray(level1, dtype=dtype))
    / jnp.asarray(20.0, dtype=dtype),
  )
  valid = (rms > 0) & jnp.isfinite(rms)
  safe_rms = jnp.where(valid, rms, jnp.ones((), dtype=dtype))
  scale = jnp.where(valid, target_rms / safe_rms, jnp.ones((), dtype=dtype))
  return reference * scale, processed * scale


def apply_nalr_24khz(signal, sample_rate, hearing_loss):
  """Resample ``signal`` to 24 kHz and apply zero-delay 140-tap NAL-R gain.

  Matches the 24 kHz, 140-tap linear NAL-R prescription that ``ear_model``
  applies to ``reference`` when ``equalisation=1``, enabling aided
  hearing-impaired evaluation of speech enhancers that do not apply hearing-aid
  gain themselves. Unlike ``ear_model``'s internal ``[140 : 140 + nsamp]``
  slice, the symmetric FIR output is centred at its group delay
  (``[70 : 70 + nsamp]``) so that output length matches the 24 kHz input, a
  flat-zero audiogram is an exact identity, and no leading samples are dropped
  or negatively delayed before ``input_align``.

  Pass ``24000`` (``calibration.MODEL_SAMPLE_RATE``) as ``processed_rate`` when
  scoring the returned signal with ``haspi_v2`` or ``hasqi_v2``.

  Args:
    signal: Input waveform to equalise (typically a calibrated processed clip).
    sample_rate: Sampling rate of ``signal`` in Hz.
    hearing_loss: Audiometric thresholds in dB HL at [250, 500, 1000, 2000,
      4000, 6000] Hz.

  Returns:
    Equalised waveform at 24 kHz, in the floating-point dtype of ``signal``.
  """
  signal = jnp.asarray(signal)
  dtype = jnp.result_type(signal, jnp.float32)
  signal = signal.astype(dtype)
  hearing_loss = jnp.asarray(hearing_loss, dtype=dtype)

  signal_24k = ear_model.resample_24khz(signal, sample_rate)
  fir, _ = nalr.build(
    hearing_loss, n_fir=_NALR_TAPS, sample_rate=MODEL_SAMPLE_RATE
  )
  half = _NALR_TAPS // 2
  return nalr.apply(fir, signal_24k)[half : half + signal_24k.shape[0]]
