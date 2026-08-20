"""HASQI against the recorded pyclarity 0.9.0 values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import hasqi

# The ear model contributes ~2e-08 (see docs/porting_notes.md); the back end
# itself agrees with the reference to machine precision.
TOLERANCE = 1e-7

RAISES = {"both_silent", "reference_silent"}


def score(case):
  return float(
    hasqi.hasqi_v2(
      case.reference,
      case.reference_sample_rate,
      case.processed,
      case.processed_sample_rate,
      case.levels,
      equalisation=case.equalisation,
      level1=case.level1,
    )[0]
  )


def scorable():
  return [c for c in goldens.cases() if c.name not in RAISES]


@pytest.mark.parametrize("case", scorable(), ids=lambda c: c.name)
def test_matches_the_reference(case):
  assert score(case) == pytest.approx(
    case.score("hasqi"), rel=TOLERANCE, abs=1e-12
  )


def test_identical_pair_scores_below_one():
  """NAL-R and the impaired ear are applied to the reference too, so a
  bit-identical pair is not identical by the time it is scored."""
  assert score(goldens.case("identical")) == pytest.approx(
    0.778153809, abs=1e-7
  )


def test_identical_without_equalisation_is_the_ceiling():
  """itype=2 skips NAL-R, which is as close to 1.0 as HASQI gets."""
  assert score(goldens.case("identical_no_eq")) == pytest.approx(
    0.924515205, abs=1e-7
  )


def test_silence_scores_zero():
  for name in ("processed_silent", "very_quiet", "audiogram_severe"):
    assert score(goldens.case(name)) == 0.0


def test_quality_falls_as_snr_falls():
  scores = [score(goldens.case(f"tone_snr{snr}")) for snr in (30, 20, 10, 0)]
  assert scores == sorted(scores, reverse=True)


def test_raw_terms_match_the_reference():
  case = goldens.case("audiogram_clinical")
  _, _, _, raw = hasqi.hasqi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    equalisation=case.equalisation,
    level1=case.level1,
  )
  want = goldens.raw(f"case{case.index:02d}__hasqi__noise_free__raw")
  np.testing.assert_allclose(np.asarray(raw), want, rtol=TOLERANCE, atol=1e-12)


def test_combined_is_the_product_of_its_parts():
  case = goldens.case("tone_snr20")
  combined, nonlinear, linear, _ = hasqi.hasqi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    equalisation=case.equalisation,
    level1=case.level1,
  )
  assert float(combined) == pytest.approx(float(nonlinear) * float(linear))


def test_env_smooth_subsamples_to_the_expected_frame_count():
  envelope = jnp.ones((32, 6000))
  smoothed = hasqi.env_smooth(envelope)
  assert smoothed.shape[0] == 32
  assert smoothed.shape[1] == 1 + 6000 // 384 + (6000 - 192) // 384


def test_mel_cepstrum_is_one_for_identical_input():
  envelope = jnp.asarray(
    np.abs(np.random.default_rng(0).standard_normal((32, 40))) * 40
  )
  average, individual = hasqi.mel_cepstrum_correlation(envelope, envelope)
  assert float(average) == pytest.approx(1.0)
  assert np.allclose(np.asarray(individual)[1:], 1.0)


def test_mel_cepstrum_is_zero_below_threshold():
  """The reference bails out when too few segments clear the silence gate."""
  quiet = jnp.full((32, 40), -100.0)
  average, individual = hasqi.mel_cepstrum_correlation(quiet, quiet)
  assert float(average) == 0.0
  assert not np.any(np.asarray(individual))


def test_spectrum_diff_is_zero_for_identical_spectra():
  spectrum = jnp.asarray(np.linspace(10.0, 40.0, 32))
  d_loud, d_norm, d_slope = hasqi.spectrum_diff(spectrum, spectrum)
  for statistics in (d_loud, d_norm, d_slope):
    np.testing.assert_allclose(np.asarray(statistics), 0.0, atol=1e-15)


def test_bm_covary_is_one_for_identical_motion():
  motion = jnp.asarray(np.random.default_rng(0).standard_normal((32, 6000)))
  covariance, reference_ms, processed_ms = hasqi.bm_covary(motion, motion)
  assert np.allclose(np.asarray(covariance), 1.0, atol=1e-9)
  np.testing.assert_allclose(
    np.asarray(reference_ms), np.asarray(processed_ms), rtol=1e-12
  )


def test_ave_covary_ignores_tiles_below_threshold():
  covariance = jnp.ones((32, 10))
  loud = jnp.full((32, 10), 100.0)
  average, sync = hasqi.ave_covary2(covariance, loud)
  assert float(average) == pytest.approx(1.0)
  assert np.all(np.asarray(sync) > 0)

  quiet = jnp.zeros((32, 10))
  average, sync = hasqi.ave_covary2(covariance, quiet)
  assert float(average) == 0.0
  assert not np.any(np.asarray(sync))


def test_back_end_stages_are_jittable():
  envelope = jnp.asarray(
    np.abs(np.random.default_rng(0).standard_normal((32, 6000))) * 40
  )
  motion = jnp.asarray(np.random.default_rng(1).standard_normal((32, 6000)))
  smoothed = jax.jit(hasqi.env_smooth)(envelope)
  assert isinstance(smoothed, jax.Array)
  assert isinstance(
    jax.jit(hasqi.mel_cepstrum_correlation)(smoothed, smoothed)[0], jax.Array
  )
  assert isinstance(jax.jit(hasqi.bm_covary)(motion, motion)[0], jax.Array)
