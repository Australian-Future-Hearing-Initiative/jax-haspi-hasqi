"""HASPI against the recorded pyclarity 0.9.0 values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import modulation
from jax_haspi_hasqi import neural_net

# Inherited from ear_model; the HASPI head itself is exact to 1e-15.
TOLERANCE = 1e-7

RAISES = {
  "both_silent": IndexError,
  "reference_silent": IndexError,
  "very_quiet": ValueError,
}

# The signal is fully masked, so the reference's dither is the only thing left
# to correlate. Noise-free the ten raw correlations are exactly zero and the
# network floor shows through; seeded, the reference scores 0.056 on noise.
DITHER_DOMINATED = {"processed_silent", "audiogram_severe"}


def score(case):
  return float(
    haspi.haspi_v2(
      case.reference,
      case.reference_sample_rate,
      case.processed,
      case.processed_sample_rate,
      case.levels,
      level1=case.level1,
      itype=case.itype,
    )[0]
  )


def scorable():
  return [c for c in goldens.cases() if c.name not in RAISES]


@pytest.mark.parametrize("case", scorable(), ids=lambda c: c.name)
def test_matches_the_reference(case):
  assert score(case) == pytest.approx(
    case.score("haspi"), rel=TOLERANCE, abs=1e-12
  )


@pytest.mark.parametrize("name", sorted(RAISES))
def test_raises_where_the_reference_raises(name):
  case = goldens.case(name)
  with pytest.raises(RAISES[name]):
    score(case)


@pytest.mark.parametrize("name", sorted(DITHER_DOMINATED))
def test_dither_dominated_cases_land_on_the_noise_free_value(name):
  """Without the reference's unseeded dither there is nothing to correlate."""
  case = goldens.case(name)
  assert score(case) == pytest.approx(case.score("haspi"), rel=1e-9)
  assert score(case) == pytest.approx(0.003924696, abs=1e-9)
  assert case.score("haspi", "seeded") > 0.05


def test_raw_correlations_are_zero_when_the_signal_is_masked():
  case = goldens.case("processed_silent")
  _, correlations = haspi.haspi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    level1=case.level1,
    itype=case.itype,
  )
  np.testing.assert_array_equal(np.asarray(correlations), np.zeros(10))


def test_raw_correlations_match_the_reference():
  case = goldens.case("audiogram_clinical")
  _, correlations = haspi.haspi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    level1=case.level1,
    itype=case.itype,
  )
  want = goldens.raw(f"case{case.index:02d}__haspi__noise_free__raw")
  np.testing.assert_allclose(np.asarray(correlations), want, rtol=TOLERANCE)


def test_identical_pair_scores_near_one():
  """Unlike HASQI, HASPI's reference ear is normal-hearing, so it nearly saturates."""
  assert score(goldens.case("identical")) == pytest.approx(
    0.998708858, abs=1e-7
  )
  assert score(goldens.case("identical_no_eq")) == pytest.approx(
    0.999981409, abs=1e-7
  )


def test_intelligibility_falls_as_snr_falls():
  scores = [score(goldens.case(f"tone_snr{snr}")) for snr in (30, 20, 10, 0)]
  assert scores == sorted(scores, reverse=True)


def test_the_normalisation_puts_the_ceiling_at_one():
  """0.9508 is chosen so the ensemble's maximum output is exactly 1."""
  assert float(neural_net.intelligibility(jnp.ones(10))) == pytest.approx(
    1.0, abs=1e-4
  )


def test_the_network_floor_is_the_dither_dominated_value():
  """Zero correlations give 0.0039 -- the score a fully masked signal gets."""
  assert float(neural_net.intelligibility(jnp.zeros(10))) == pytest.approx(
    0.003924696, abs=1e-9
  )


def test_the_network_is_a_ten_member_ensemble():
  features = jnp.linspace(0.1, 0.9, 10)
  ensemble = float(neural_net.nn_feed_forward_ensemble(features))
  assert 0.0 < ensemble < 1.0
  assert float(neural_net.intelligibility(features)) == pytest.approx(
    ensemble / 0.9508
  )


def test_modulation_bands_are_dropped_above_nyquist():
  """Only bands whose upper edge clears the subsampling Nyquist survive."""
  envelope = jnp.asarray(
    np.abs(np.random.default_rng(0).standard_normal((200, 6))) * 40
  )
  _, _, centres = modulation.fir_modulation_filter(envelope, envelope, 2560.0)
  assert len(centres) == 10
  _, _, narrow = modulation.fir_modulation_filter(envelope, envelope, 200.0)
  assert len(narrow) < 10


def test_cepstral_gate_raises_below_threshold():
  quiet = np.full((100, 32), -100.0)
  with pytest.raises(ValueError, match="below threshold"):
    modulation.cepstral_correlation_coef(quiet, quiet, 2.5, 0.1, 6)


def test_cepstral_gate_drops_silent_samples():
  """The kept-sample count is data-dependent, which is why this stays numpy."""
  envelope = np.full((100, 32), 40.0)
  envelope[:40] = -100.0
  reference_cep, processed_cep = modulation.cepstral_correlation_coef(
    envelope, envelope, 2.5, 0.1, 6
  )
  assert reference_cep.shape == (60, 6)
  assert processed_cep.shape == (60, 6)


def test_cross_correlation_is_one_for_identical_input():
  bands = jnp.asarray(np.random.default_rng(0).standard_normal((6, 10, 200)))
  correlations = modulation.modulation_cross_correlation(bands, bands)
  np.testing.assert_allclose(np.asarray(correlations), 1.0, rtol=1e-12)


def test_filterbank_and_network_are_jittable():
  envelope = jnp.asarray(
    np.abs(np.random.default_rng(0).standard_normal((200, 6))) * 40
  )
  reference, processed, _ = modulation.fir_modulation_filter(
    envelope, envelope, 2560.0
  )
  correlate = jax.jit(modulation.modulation_cross_correlation)
  assert isinstance(correlate(reference, processed), jax.Array)
  assert isinstance(
    jax.jit(neural_net.intelligibility)(jnp.ones(10)), jax.Array
  )


def test_network_is_differentiable():
  gradient = jax.grad(neural_net.intelligibility)(jnp.full(10, 0.5))
  assert bool(jnp.all(jnp.isfinite(gradient)))
  assert float(jnp.max(jnp.abs(gradient))) > 0
