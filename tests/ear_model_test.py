"""Ear model against the recorded pyclarity 0.9.0 per-stage values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import nalr

AUDIOGRAMS = ("flat_zero", "clinical")
ITYPES = (0, 1)
OUTPUTS = (
  "reference_db",
  "reference_bm",
  "processed_db",
  "processed_bm",
  "reference_sl",
  "processed_sl",
)

# The scan-based IIR accumulates a little differently from scipy's, and the
# gammatone recursion is 4th-order over thousands of samples. See
# docs/porting_notes.md for the measurement.
TOLERANCE = 1e-7


def levels(name):
  return np.asarray(goldens.manifest()["audiograms"][name])


def relative_error(got, want):
  got = np.asarray(got)
  want = np.asarray(want)
  assert got.shape == want.shape
  return np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30)


@pytest.fixture(scope="module")
def signals():
  return goldens.stage("reference_signal"), goldens.stage("processed_signal")


@pytest.fixture(scope="module")
def outputs(signals):
  reference, processed = signals
  return {
    (name, itype): ear_model.ear_model(
      reference, 24000, processed, 24000, levels(name), itype, 65.0
    )
    for name in AUDIOGRAMS
    for itype in ITYPES
  }


def test_center_frequency_matches():
  np.testing.assert_allclose(
    ear_model.center_frequency(32), goldens.stage("center_frequency_32"), rtol=0
  )


def test_center_frequency_shift_matches():
  """The shift is exercised even though HASPI never asks for it.

  Not bit-exact, unlike the unshifted case above. The shifted path is the only
  one that routes through np.log10 and 10**, and libm's last-bit rounding for
  those is not fixed across numpy builds: numpy 2.4.6 and 2.5.2 disagree by one
  ULP on channel 11 (792.7504771345069 vs 792.750477134507, 1.4e-16 relative).
  Asserting bit-equality there tests the C library, not the port. A few ULP is
  still thirteen orders below the 1e-08 the goldens pin overall.
  """
  np.testing.assert_allclose(
    ear_model.center_frequency(32, 0.02),
    goldens.stage("center_frequency_32_shift02"),
    rtol=1e-15,
    atol=0,
  )


def test_middle_ear_matches(signals):
  got = ear_model.middle_ear(jnp.asarray(signals[0]))
  assert relative_error(got, goldens.stage("middle_ear")) < 1e-12


@pytest.mark.parametrize("name", AUDIOGRAMS)
def test_loss_parameters_match(name):
  keys = ("attn_ohc", "bandwidth", "low_knee", "compression_ratio", "attn_ihc")
  got = ear_model.loss_parameters(levels(name), ear_model.center_frequency(32))
  for key, value in zip(keys, got):
    want = goldens.stage(f"loss_parameters__{name}__{key}")
    assert relative_error(value, want) < 1e-15, key


# The design sums a cosine basis where the reference ran an FFT, and a radix-2
# transform accumulates more accurately than any direct evaluation: computing
# both the basis and the sum in extended precision still lands at 2.8e-15, so
# this is a floor rather than something to tune away. Only the audiograms that
# actually exercise it get the looser bound.
NALR_TAP_TOLERANCE = {"clinical": 5e-15}


@pytest.mark.parametrize("name", AUDIOGRAMS)
def test_nalr_filter_matches(name):
  """Taps match the reference design.

  Held at 1e-15 except where noted in NALR_TAP_TOLERANCE. For context on the
  size of that exception: before the design stopped using an FFT, clinical sat
  at 2.6e-17 and flat_zero at exactly 0. clinical now measures 2.9e-15. The
  taps are worth far less than that downstream -- the change moves HASQI by at
  most 3e-10, against an ear model that already differs from the reference by
  2.8e-08 -- but the step is two orders of magnitude, so it is named here
  rather than absorbed into a blanket bound.
  """
  got, _ = nalr.build(levels(name))
  tolerance = NALR_TAP_TOLERANCE.get(name, 1e-15)
  assert relative_error(got, goldens.stage(f"nalr_fir__{name}")) < tolerance


def test_nalr_is_a_pure_delay_without_loss():
  built, delay = nalr.build(levels("flat_zero"))
  np.testing.assert_array_equal(np.asarray(built), np.asarray(delay))


@pytest.mark.parametrize("name", AUDIOGRAMS)
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("key", OUTPUTS)
def test_ear_model_output_matches(outputs, name, itype, key):
  got = outputs[(name, itype)][OUTPUTS.index(key)]
  want = goldens.stage(f"ear_model__{name}__itype{itype}__{key}")
  assert relative_error(got, want) < TOLERANCE


def test_ear_model_reports_the_model_sample_rate(outputs):
  assert outputs[("clinical", 0)][6] == 24000.0


def test_ear_model_gives_one_row_per_channel(outputs):
  for key in ("reference_db", "processed_bm"):
    assert outputs[("clinical", 0)][OUTPUTS.index(key)].shape[0] == 32


def test_intelligibility_gives_the_reference_a_normal_ear(outputs):
  """itype=0 zeroes the reference's loss, so its envelope ignores the audiogram."""
  flat = outputs[("flat_zero", 0)][0]
  clinical = outputs[("clinical", 0)][0]
  np.testing.assert_allclose(np.asarray(flat), np.asarray(clinical), rtol=0)


def test_quality_applies_the_loss_to_both_ears(outputs):
  """itype=1 does not, which is the whole difference between the two modes."""
  flat = outputs[("flat_zero", 1)][0]
  clinical = outputs[("clinical", 1)][0]
  assert not np.allclose(np.asarray(flat), np.asarray(clinical))


def test_bandwidth_adjust_matches_the_reference_branches():
  """Below 50 dB SPL the bandwidth floors, above 100 it saturates."""
  quiet = jnp.full(100, 1e-4)
  loud = jnp.full(100, 1e4)
  assert ear_model.bandwidth_adjust(quiet, 1.0, 3.0, 65.0) == pytest.approx(1.0)
  assert ear_model.bandwidth_adjust(loud, 1.0, 3.0, 65.0) == pytest.approx(3.0)


def test_threshold_noise_defaults_to_silence(signals):
  """The reference draws unseeded noise here; the port defaults to none."""
  reference, processed = signals
  quiet = ear_model.ear_model(
    reference, 24000, processed, 24000, levels("clinical"), 0, 65.0
  )
  noisy = ear_model.ear_model(
    reference,
    24000,
    processed,
    24000,
    levels("clinical"),
    0,
    65.0,
    reference_noise=jnp.ones((32, len(goldens.stage("middle_ear")) - 1)),
    processed_noise=jnp.zeros((32, len(goldens.stage("middle_ear")) - 1)),
  )
  assert not np.allclose(np.asarray(quiet[1]), np.asarray(noisy[1]))


def test_the_aligned_model_is_jittable(signals):
  """input_align is resolved outside the graph; everything after it compiles."""
  reference, processed = signals
  aligned = ear_model.ear_model(
    reference, 24000, processed, 24000, levels("clinical"), 0, 65.0
  )
  assert isinstance(aligned[0], jax.Array)


def test_resample_is_a_no_op_at_the_model_rate(signals):
  got = ear_model.resample_24khz(jnp.asarray(signals[0]), 24000)
  np.testing.assert_array_equal(np.asarray(got), signals[0])


def test_resample_rounds_the_input_rate_to_the_nearest_khz(signals):
  """22050 Hz is treated as 22 kHz, as the reference does."""
  at_22050 = ear_model.resample_24khz(jnp.asarray(signals[0]), 22050)
  at_22000 = ear_model.resample_24khz(jnp.asarray(signals[0]), 22000)
  np.testing.assert_array_equal(np.asarray(at_22050), np.asarray(at_22000))


def test_input_align_prunes_leading_silence():
  signal = np.concatenate((np.zeros(500), np.ones(1000)))
  reference, processed, (start, _, _) = ear_model.input_align(signal, signal)
  assert start == 500
  assert len(reference) == len(processed) == 1000


def test_input_align_rejects_a_silent_reference():
  """Upstream indexes an empty array here; the cause is named instead."""
  silent = np.zeros(2400)
  with pytest.raises(ValueError, match="silence threshold"):
    ear_model.input_align(silent, silent)


def test_input_align_rejects_a_reference_with_no_finite_peak():
  """NaN fails the > comparison too, so a non-zero max is not the condition."""
  reference = np.full(2400, np.nan)
  with pytest.raises(ValueError, match="silence threshold"):
    ear_model.input_align(reference, np.ones(2400))


def test_input_align_is_bit_identical_on_a_scorable_pair():
  """The guard must not perturb the alignment it does not reject."""
  rng = np.random.default_rng(11)
  reference = np.concatenate((np.zeros(300), rng.standard_normal(4000)))
  processed = np.concatenate((np.zeros(280), rng.standard_normal(4000)))
  got_reference, got_processed, meta = ear_model.input_align(
    reference, processed
  )

  start, stop, delay = meta
  assert (start, stop, delay) == (300, 4280, -1203)
  shifted = np.concatenate(
    (np.zeros(-delay), processed[: len(processed) + delay])
  )
  np.testing.assert_array_equal(got_reference, reference[start : stop + 1])
  np.testing.assert_array_equal(got_processed, shifted[start : stop + 1])


def test_envelope_align_recovers_a_known_shift():
  """Shifting back restores the overlap; the vacated tail stays zero."""
  rng = np.random.default_rng(0)
  reference = rng.standard_normal(2000)
  shift = 37
  shifted = np.concatenate((np.zeros(shift), reference[:-shift]))
  got = np.asarray(
    ear_model.envelope_align(jnp.asarray(reference), jnp.asarray(shifted))
  )
  np.testing.assert_allclose(got[:-shift], reference[:-shift], atol=1e-12)
  np.testing.assert_array_equal(got[-shift:], np.zeros(shift))


def test_top_k_breaks_ties_towards_the_lower_lag():
  """Pins the tie rule that keeps a fully silent case on its usual lag.

  A silent input correlates to exactly zero everywhere, so nothing
  distinguishes the lags and the tie rule alone decides. top_k resolves ties
  to the lower index, matching argmax; argpartition returns the highest lags
  instead and would shift the silent golden case.

  This pins the property rather than guarding the substitution: a silent
  window maps every lag to the same all-zero output, so no assertion made
  through envelope_align can observe the wrong choice. It fails only if top_k
  itself changes, and exists so the requirement stays written down.
  """
  window = jnp.zeros(799)

  assert np.array_equal(np.asarray(lax.top_k(window, 8)[1]), np.arange(8))
  assert int(jnp.argmax(window)) == 0

  # The behaviour being ruled out, recorded so the risk stays visible.
  assert np.argpartition(-np.asarray(window), 7)[:8].min() > 8


def test_envelope_align_is_insensitive_to_the_shortlist_size():
  """Any shortlist deep enough to hold the peak must give the same answer."""
  rng = np.random.default_rng(1)
  reference = jnp.asarray(rng.standard_normal(1500))
  output = jnp.asarray(rng.standard_normal(1500))
  want = np.asarray(ear_model.envelope_align(reference, output, candidates=2))
  for candidates in (4, 8, 16):
    got = ear_model.envelope_align(reference, output, candidates=candidates)
    np.testing.assert_array_equal(np.asarray(got), want)
