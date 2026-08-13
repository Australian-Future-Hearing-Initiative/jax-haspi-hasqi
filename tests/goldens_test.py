"""Checks that the checked-in goldens are complete and self-consistent."""

import numpy as np
import pytest

from jax_haspi_hasqi import goldens

EXPECTED_AUDIOGRAMS = {
  "flat_zero",
  "mild",
  "clinical",
  "moderate",
  "severe",
  "sloping_steep",
}

# Cases whose signal is degenerate enough that the reference's dither is the
# only thing left to correlate, so noise-free and seeded genuinely diverge.
DITHER_DOMINATED = {"processed_silent", "audiogram_severe"}

# Largest observed |noise_free - seeded|, rounded up: 1.14e-02 (HASPI,
# scaled_half) and 1.39e-02 (HASQI, speech_snr5).
MAX_NOISE_FREE_GAP = {"haspi": 1.2e-02, "hasqi": 1.4e-02}

# Above this the gap is the reference's noise, not float error.
SIGNIFICANT_GAP = 1e-04


def test_manifest_pins_the_reference_version():
  assert goldens.manifest()["pyclarity_version"] == "0.9.0"


def test_every_case_has_its_inputs():
  for case in goldens.cases():
    assert case.reference.size > 0
    assert case.processed.size > 0
    assert case.levels.shape == (6,)
    assert np.isfinite(case.reference).all()
    assert np.isfinite(case.processed).all()


def test_scores_are_finite_or_the_reference_raised():
  for case in goldens.cases():
    for metric in ("haspi", "hasqi"):
      score = case.score(metric)
      if score is None:
        assert case.raises, case.name
      else:
        assert np.isfinite(score), (case.name, metric)


def test_noise_free_and_seeded_agree_except_where_the_signal_is_degenerate():
  """HASQI's gap is one-sided; HASPI's is not. See the README.

  Where the processed BM motion is exactly zero, bm_covary scores the tile 0
  but still counts it, so the reference's noise can only raise HASQI. HASPI
  never sees that noise, and its cepstral dither is unbiased.
  """
  for case in goldens.cases():
    if case.name in DITHER_DOMINATED:
      continue
    for metric in ("haspi", "hasqi"):
      free = case.score(metric)
      seeded = case.score(metric, "seeded")
      if free is None or seeded is None:
        continue
      gap = free - seeded
      assert abs(gap) < MAX_NOISE_FREE_GAP[metric], (case.name, metric, gap)
      if metric == "hasqi" and abs(gap) > SIGNIFICANT_GAP:
        assert gap < 0, (case.name, gap)


@pytest.mark.parametrize("name", sorted(DITHER_DOMINATED))
def test_dither_dominates_only_where_expected(name):
  """A silent or fully masked signal scores 0 noise-free but not seeded."""
  case = goldens.case(name)
  assert case.score("haspi") == pytest.approx(0.0039, abs=1e-3)
  assert case.score("haspi", "seeded") > 0.05


def test_seeded_variant_records_its_noise_draws():
  """Four draws per call, or two when the reference exits early."""
  for case in goldens.cases():
    for metric in ("haspi", "hasqi"):
      if case.score(metric, "seeded") is None:
        continue
      shapes = getattr(case, metric)["noise_shapes"]
      assert len(shapes) in (2, 4), (case.name, metric, len(shapes))
      assert all(len(shape) == 2 for shape in shapes)


def test_identical_inputs_score_near_the_top():
  """itype=2 skips NAL-R, so an identical pair is the real ceiling.

  HASQI still falls short of 1.0 because the impaired ear model is applied to
  the reference as well, and its OHC compression is not level-preserving.
  """
  case = goldens.case("identical_no_eq")
  assert case.score("haspi") > 0.999
  assert case.score("hasqi") > 0.92


def test_nalr_on_the_reference_lowers_an_identical_pair():
  """itype=0/1 equalise the reference, so the 'identical' pair stops being so."""
  equalised = goldens.case("identical").score("hasqi")
  assert equalised < goldens.case("identical_no_eq").score("hasqi")


def test_near_identical_matches_identical():
  """A 1e-6 perturbation must not move the score by more than ~1e-5."""
  free = goldens.case("identical").score("hasqi")
  near = goldens.case("near_identical").score("hasqi")
  assert near == pytest.approx(free, abs=1e-5)


def test_degradation_is_monotonic_in_snr():
  scores = [
    goldens.case(f"tone_snr{snr}").score("hasqi") for snr in (30, 20, 10, 0)
  ]
  assert scores == sorted(scores, reverse=True), scores


def test_silence_cases_record_the_reference_raising():
  for name in ("both_silent", "reference_silent"):
    assert goldens.case(name).raises == "IndexError"


def test_very_quiet_splits_the_two_metrics():
  """HASPI raises below the cepstral threshold; HASQI returns zero."""
  case = goldens.case("very_quiet")
  assert case.haspi["raises"] == "ValueError"
  assert case.score("hasqi") == 0.0


@pytest.mark.parametrize("itype", [0, 1, 2])
def test_every_itype_branch_is_covered(itype):
  assert goldens.case(f"itype{itype}").score("haspi") is not None


def test_every_audiogram_is_covered():
  covered = {case.audiogram for case in goldens.cases()}
  assert EXPECTED_AUDIOGRAMS <= covered


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 32000, 44100])
def test_every_resampling_branch_is_covered(rate):
  rates = {case.reference_sample_rate for case in goldens.cases()}
  assert rate in rates


def test_mismatched_sample_rates_are_covered():
  case = goldens.case("rate_mismatch_24k_16k")
  assert case.reference_sample_rate != case.processed_sample_rate
  assert case.score("haspi") is not None


def test_unequal_lengths_are_covered():
  case = goldens.case("unequal_length")
  assert len(case.processed) < len(case.reference)
  assert case.score("haspi") is not None


def test_nalr_is_a_pure_delay_when_there_is_no_loss():
  fir = goldens.stage("nalr_fir__flat_zero")
  delay = goldens.stage("nalr_delay__flat_zero")
  np.testing.assert_array_equal(fir, delay)


def test_stage_goldens_cover_the_ear_model_outputs():
  for tag in ("flat_zero__itype0", "clinical__itype1"):
    for key in ("reference_db", "processed_db", "reference_bm", "reference_sl"):
      assert goldens.stage(f"ear_model__{tag}__{key}").size > 0


def test_ear_model_gives_32_channels():
  envelope = goldens.stage("ear_model__clinical__itype0__reference_db")
  assert envelope.shape[0] == 32


def test_centre_frequencies_span_the_documented_range():
  centre = goldens.stage("center_frequency_32")
  assert centre.shape == (32,)
  assert centre[0] == pytest.approx(80.0)
  assert centre[-1] == pytest.approx(8000.0)


def test_the_unapplied_shift_is_visible_in_the_goldens():
  """shift=0.02 moves the bank; HASPI's defect is that it never asks for it."""
  plain = goldens.stage("center_frequency_32")
  shifted = goldens.stage("center_frequency_32_shift02")
  assert not np.allclose(plain, shifted)


def test_loss_parameters_are_recorded_for_both_extremes():
  for name in ("flat_zero", "clinical"):
    for key in ("attn_ohc", "bandwidth", "low_knee", "compression_ratio"):
      assert goldens.stage(f"loss_parameters__{name}__{key}").shape == (32,)
