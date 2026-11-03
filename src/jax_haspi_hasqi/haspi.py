"""HASPI v2 intelligibility index, ported from clarity.evaluator.haspi.haspi."""

import functools

import jax
import jax.numpy as jnp

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import modulation
from jax_haspi_hasqi import neural_net

_LOW_PASS_CUTOFF = 320.0
_CEPSTRAL_BASIS = 6
_SILENCE_THRESHOLD = 2.5
_NERVE_DITHER = 0.1


@functools.partial(jax.jit, static_argnames=("f_lp", "sample_rate"))
def _haspi_backend(
  reference_db,
  processed_db,
  valid_len,
  f_lp,
  sample_rate,
  cepstral_noise=None,
):
  """Everything downstream of the ear model, as one compiled region.

  Returns:
    The intelligibility estimate and the ten modulation-band correlations.
  """
  freq_sub_sample = 8.0 * f_lp
  reference_lp, processed_lp = modulation.env_filter(
    reference_db,
    processed_db,
    f_lp,
    freq_sub_sample,
    sample_rate,
    valid_len=valid_len,
  )
  reference_cep, processed_cep, nkept = modulation.cepstral_correlation_coef(
    reference_lp,
    processed_lp,
    _SILENCE_THRESHOLD,
    _NERVE_DITHER,
    _CEPSTRAL_BASIS,
    cepstral_noise,
  )
  reference_mod, processed_mod, _ = modulation.fir_modulation_filter(
    reference_cep, processed_cep, freq_sub_sample
  )
  correlations = modulation.modulation_cross_correlation(
    reference_mod, processed_mod, nkept=nkept
  )
  score = jnp.where(
    nkept > 1,
    neural_net.intelligibility(correlations),
    jnp.zeros((), dtype=correlations.dtype),
  )
  return score, correlations


def haspi_v2(
  reference,
  reference_rate,
  processed,
  processed_rate,
  hearing_loss,
  level1=65.0,
  f_lp=_LOW_PASS_CUTOFF,
  itype=0,
  reference_noise=None,
  processed_noise=None,
  cepstral_noise=None,
):
  """HASPI version 2 intelligibility index.

  Runs in the floating-point precision of ``reference`` and ``processed``
  (``float32`` by default, or ``float64`` when ``jax_enable_x64`` is enabled
  and inputs are ``float64``). Returns ``0.0`` when the reference fails to
  clear the silence threshold.

  When citing scores from this function, use:
    "HASPI v2 (pyclarity 0.9.0 algorithm, deterministic noise-free variant)"

  Args:
    reference: Clean reference signal, without amplification.
    reference_rate: Its sampling rate in Hz.
    processed: Processed signal.
    processed_rate: Its sampling rate in Hz.
    hearing_loss: Audiometric thresholds in dB HL at [250, 500, 1000, 2000,
      4000, 6000] Hz. The caller must resample/interpolate other frequencies to
      these six values before calling (e.g. log-frequency interpolation).
    level1: dB SPL corresponding to an RMS of 1.
    f_lp: Envelope low-pass cutoff in Hz.
    itype: Passed to the ear model; 0 for intelligibility.
    reference_noise: Threshold noise for the reference's BM motion.
    processed_noise: As above, for the processed signal.
    cepstral_noise: IHC firing dither for the two subsampled envelopes, in the
      post-silence-gate shape the reference draws it in.

  Returns:
    The intelligibility estimate and the ten modulation-band correlations.
  """
  # shift=None is not passed on: the ear model already omits the 0.02 basal
  # shift HASPI's own docstring describes, which is the published defect.
  reference_db, _, processed_db, _, _, _, sample_rate, valid_len = (
    ear_model._run_ear_model(
      reference,
      reference_rate,
      processed,
      processed_rate,
      hearing_loss,
      itype,
      level1,
      reference_noise=reference_noise,
      processed_noise=processed_noise,
    )
  )
  return _haspi_backend(
    reference_db,
    processed_db,
    valid_len,
    f_lp,
    sample_rate,
    cepstral_noise=cepstral_noise,
  )


def haspi_v2_better_ear(
  reference_left,
  reference_right,
  processed_left,
  processed_right,
  sample_rate,
  hearing_loss_left,
  hearing_loss_right,
  level=100.0,
):
  """HASPI for the better of the two ears."""
  left, _ = haspi_v2(
    reference_left,
    sample_rate,
    processed_left,
    sample_rate,
    hearing_loss_left,
    level,
  )
  right, _ = haspi_v2(
    reference_right,
    sample_rate,
    processed_right,
    sample_rate,
    hearing_loss_right,
    level,
  )
  return jnp.maximum(left, right)
