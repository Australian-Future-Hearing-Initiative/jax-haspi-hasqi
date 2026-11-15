"""RMS-to-SPL calibration and 24 kHz NAL-R equalisation helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import calibration
from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi

SR = 24000


def _levels(name):
  return np.asarray(goldens.manifest()["audiograms"][name], dtype=np.float64)


def _speech_like(sample_rate=SR, duration=1.0, pad=0.05, seed=42):
  """Smoothly tapered multi-harmonic signal (150-4350 Hz) at ~-24 dBFS."""
  rng = np.random.default_rng(seed)
  n = int(round(duration * sample_rate))
  t = np.arange(n) / sample_rate
  taper = np.sin(np.pi * np.arange(n) / n) ** 2
  envelope = (
    taper
    * 0.5
    * (1.0 + np.sin(2.0 * np.pi * 3.5 * t))
    * (0.5 + 0.5 * np.sin(2.0 * np.pi * 1.3 * t + 0.7))
  )
  signal = (
    sum(
      (1.0 / k)
      * np.sin(2.0 * np.pi * 150.0 * k * t + rng.uniform(0, 2 * np.pi))
      for k in range(1, 30)
    )
    * envelope
  )
  zeros = np.zeros(int(round(pad * sample_rate)))
  padded = np.concatenate((zeros, signal, zeros))
  rms = np.sqrt(np.mean(padded**2))
  return padded / rms * (10.0 ** (-24.0 / 20.0))


@pytest.mark.parametrize("dbfs", (-40.0, -24.0, -6.0, 0.0, 12.0))
def test_scale_to_spl_normalises_reference_rms_to_one_at_level1(dbfs):
  raw = _speech_like(duration=0.25, pad=0.0)
  raw = raw / np.sqrt(np.mean(raw**2)) * (10.0 ** (dbfs / 20.0))
  processed = raw * 0.75

  ref_cal, proc_cal = calibration.scale_to_spl(raw, processed)
  assert float(jnp.sqrt(jnp.mean(ref_cal**2))) == pytest.approx(1.0, rel=1e-12)
  assert float(jnp.sqrt(jnp.mean(proc_cal**2))) == pytest.approx(
    0.75, rel=1e-12
  )


def test_scale_to_spl_maps_target_spl_relative_to_level1():
  raw = _speech_like(duration=0.25, pad=0.0)

  ref_75, _ = calibration.scale_to_spl(raw, raw, target_spl=75.0, level1=65.0)
  assert float(jnp.sqrt(jnp.mean(ref_75**2))) == pytest.approx(
    10.0 ** (10.0 / 20.0), rel=1e-12
  )

  # Clarity Challenge convention: level=100 dB SPL for RMS=1 -> 65 dB is -35 dB.
  ref_100, _ = calibration.scale_to_spl(raw, raw, target_spl=65.0, level1=100.0)
  assert float(jnp.sqrt(jnp.mean(ref_100**2))) == pytest.approx(
    10.0 ** (-35.0 / 20.0), rel=1e-12
  )


def test_scale_to_spl_target_matches_equivalent_level1_in_metrics():
  """Scaling to 75 dB re level1=65 matches RMS=1 evaluated at level1=75."""
  ref = _speech_like()
  rng = np.random.default_rng(1)
  proc = ref + 0.1 * rng.standard_normal(len(ref)) * np.sqrt(np.mean(ref**2))
  levels = _levels("clinical")

  ref_a, proc_a = calibration.scale_to_spl(
    ref, proc, target_spl=75.0, level1=65.0
  )
  ref_b, proc_b = calibration.scale_to_spl(
    ref, proc, target_spl=65.0, level1=65.0
  )

  haspi_a, _ = haspi.haspi_v2(ref_a, SR, proc_a, SR, levels, level1=65.0)
  haspi_b, _ = haspi.haspi_v2(ref_b, SR, proc_b, SR, levels, level1=75.0)
  hasqi_a, *_ = hasqi.hasqi_v2(
    ref_a, SR, proc_a, SR, levels, equalisation=2, level1=65.0
  )
  hasqi_b, *_ = hasqi.hasqi_v2(
    ref_b, SR, proc_b, SR, levels, equalisation=2, level1=75.0
  )

  assert float(haspi_a) == pytest.approx(float(haspi_b), rel=1e-10)
  assert float(hasqi_a) == pytest.approx(float(hasqi_b), rel=1e-10)


def test_scale_to_spl_preserves_snr_and_unequal_lengths():
  rng = np.random.default_rng(2)
  ref = _speech_like(duration=0.5, pad=0.0)
  noise = rng.standard_normal(len(ref)) * np.sqrt(np.mean(ref**2)) * 0.25
  proc = ref + noise

  ref_cal, proc_cal = calibration.scale_to_spl(ref, proc)
  snr_before = np.sum(ref**2) / np.sum((proc - ref) ** 2)
  snr_after = float(jnp.sum(ref_cal**2) / jnp.sum((proc_cal - ref_cal) ** 2))
  assert snr_after == pytest.approx(snr_before, rel=1e-12)

  # Processed shorter than reference keeps its own length and relative RMS.
  short_proc = proc[: len(proc) // 2]
  ref_u, proc_u = calibration.scale_to_spl(ref, short_proc)
  assert ref_u.shape == ref.shape
  assert proc_u.shape == short_proc.shape
  assert float(jnp.sqrt(jnp.mean(proc_u**2))) == pytest.approx(
    np.sqrt(np.mean(short_proc**2)) / np.sqrt(np.mean(ref**2)), rel=1e-12
  )


def test_metrics_are_scale_invariant_after_scale_to_spl():
  ref = _speech_like()
  rng = np.random.default_rng(3)
  proc = ref + 0.2 * rng.standard_normal(len(ref)) * np.sqrt(np.mean(ref**2))
  levels = _levels("clinical")

  scores = []
  for gain in (1e-3, 0.05, 1.0, 50.0):
    rc, pc = calibration.scale_to_spl(ref * gain, proc * gain)
    hp = float(haspi.haspi_v2(rc, SR, pc, SR, levels)[0])
    hq = float(hasqi.hasqi_v2(rc, SR, pc, SR, levels, equalisation=2)[0])
    scores.append((hp, hq))

  base_hp, base_hq = scores[0]
  for hp, hq in scores[1:]:
    assert hp == pytest.approx(base_hp, rel=1e-10)
    assert hq == pytest.approx(base_hq, rel=1e-10)


def test_model_sample_rate_is_int_and_shares_jit_cache_with_24000():
  assert isinstance(calibration.MODEL_SAMPLE_RATE, int)
  assert calibration.MODEL_SAMPLE_RATE == 24000

  signal = jnp.asarray(_speech_like(duration=0.1, pad=0.0))
  levels = jnp.asarray(_levels("clinical"))
  jitted = jax.jit(
    calibration.apply_nalr_24khz, static_argnames=("sample_rate",)
  )
  jitted(signal, 24000, levels)
  jitted(signal, calibration.MODEL_SAMPLE_RATE, levels)
  assert jitted._cache_size() == 1


def test_scale_to_spl_rejects_non_1d_but_supports_vmap():
  batch_ref = jnp.ones((2, 100), dtype=jnp.float64)
  batch_proc = jnp.ones((2, 100), dtype=jnp.float64) * 0.5

  with pytest.raises(ValueError, match="1-D"):
    calibration.scale_to_spl(batch_ref, batch_proc)
  with pytest.raises(ValueError, match="1-D"):
    calibration.scale_to_spl(jnp.ones(100), batch_proc)
  with pytest.raises(ValueError, match="1-D"):
    calibration.scale_to_spl(jnp.asarray(1.0), jnp.ones(10))

  r_batched, p_batched = jax.vmap(calibration.scale_to_spl)(
    batch_ref * jnp.array([[0.1], [2.0]]), batch_proc
  )
  assert r_batched.shape == (2, 100)
  np.testing.assert_allclose(
    np.asarray(jnp.sqrt(jnp.mean(r_batched**2, axis=1))), 1.0, rtol=1e-12
  )
  np.testing.assert_allclose(
    np.asarray(jnp.sqrt(jnp.mean(p_batched**2, axis=1))),
    np.array([5.0, 0.25]),
    rtol=1e-12,
  )


def test_scale_to_spl_handles_silent_empty_nonfinite_and_extreme_inputs():
  silent = jnp.zeros(1000, dtype=jnp.float64)
  proc = jnp.ones(1000, dtype=jnp.float64) * 0.25

  ref_s, proc_s = calibration.scale_to_spl(silent, proc)
  assert bool(jnp.all(jnp.isfinite(ref_s)))
  assert bool(jnp.all(jnp.isfinite(proc_s)))
  np.testing.assert_array_equal(np.asarray(ref_s), np.zeros(1000))
  np.testing.assert_array_equal(np.asarray(proc_s), np.asarray(proc))

  # Empty reference returns empty reference and unchanged processed.
  empty = jnp.zeros(0, dtype=jnp.float64)
  ref_e, proc_e = calibration.scale_to_spl(empty, proc)
  assert ref_e.shape == (0,)
  np.testing.assert_array_equal(np.asarray(proc_e), np.asarray(proc))

  # Float32 subnormal square underflows to zero without producing NaN or Inf.
  tiny = jnp.full(100, 1e-25, dtype=jnp.float32)
  ref_t, proc_t = calibration.scale_to_spl(tiny, jnp.ones(100, jnp.float32))
  assert ref_t.dtype == proc_t.dtype == jnp.float32
  assert bool(jnp.all(jnp.isfinite(ref_t)))
  assert bool(jnp.all(jnp.isfinite(proc_t)))

  # Inf and NaN references leave finite processed unpolluted.
  fin_proc = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64)
  for bad_val in (jnp.inf, jnp.nan):
    bad_ref = jnp.array([bad_val, 1.0, 2.0], dtype=jnp.float64)
    _, proc_out = calibration.scale_to_spl(bad_ref, fin_proc)
    np.testing.assert_array_equal(np.asarray(proc_out), np.asarray(fin_proc))

  # Extreme or NaN target_spl that would overflow leaves inputs finite.
  unit = jnp.ones(16, dtype=jnp.float32) * 0.25
  for bad_spl in (1000.0, jnp.nan):
    r_ex, p_ex = calibration.scale_to_spl(unit, unit, target_spl=bad_spl)
    np.testing.assert_array_equal(np.asarray(r_ex), np.asarray(unit))
    np.testing.assert_array_equal(np.asarray(p_ex), np.asarray(unit))

  # Quotient overflow (finite target_rms / tiny finite rms) falls back safely.
  small32 = jnp.full(16, 1e-12, dtype=jnp.float32)
  r_ov, p_ov = calibration.scale_to_spl(small32, unit, target_spl=665.0)
  np.testing.assert_array_equal(np.asarray(r_ov), np.asarray(small32))
  np.testing.assert_array_equal(np.asarray(p_ov), np.asarray(unit))


def test_scale_to_spl_preserves_caller_dtype():
  ref = _speech_like(duration=0.1, pad=0.0)
  proc = ref * 0.8

  r32, p32 = calibration.scale_to_spl(jnp.float32(ref), jnp.float32(proc))
  assert r32.dtype == p32.dtype == jnp.float32

  # Non-default Python float and float64 level args must not promote float32.
  r32_py, p32_py = calibration.scale_to_spl(
    jnp.float32(ref), jnp.float32(proc), target_spl=72.5, level1=100.0
  )
  assert r32_py.dtype == p32_py.dtype == jnp.float32
  r32_64, p32_64 = calibration.scale_to_spl(
    jnp.float32(ref),
    jnp.float32(proc),
    target_spl=jnp.float64(72.5),
    level1=jnp.float64(100.0),
  )
  assert r32_64.dtype == p32_64.dtype == jnp.float32

  # int16 PCM waveforms promote to float32 without int16 square overflow.
  pcm = np.array([-16384, 0, 16384, 8192], dtype=np.int16)
  rpcm, ppcm = calibration.scale_to_spl(pcm, pcm)
  assert rpcm.dtype == ppcm.dtype == jnp.float32
  assert float(jnp.sqrt(jnp.mean(rpcm**2))) == pytest.approx(1.0, rel=1e-6)

  r64, p64 = calibration.scale_to_spl(jnp.float64(ref), jnp.float64(proc))
  assert r64.dtype == p64.dtype == jnp.float64

  # Mixed float32/float64 resolves to float64 under x64.
  rmix, pmix = calibration.scale_to_spl(jnp.float32(ref), jnp.float64(proc))
  assert rmix.dtype == pmix.dtype == jnp.float64

  # Python lists are accepted without TypeError.
  rl, pl = calibration.scale_to_spl([0.1, -0.2, 0.3], [0.2, -0.1, 0.0])
  assert rl.dtype == pl.dtype == jnp.float64

  with jax.enable_x64(False):
    rf, pf = calibration.scale_to_spl(np.float64(ref), np.float64(proc))
    assert rf.dtype == pf.dtype == jnp.float32


def test_scale_to_spl_is_jittable_and_differentiable():
  ref = jnp.asarray(_speech_like(duration=0.1, pad=0.0))
  proc = ref * 0.6

  jitted = jax.jit(calibration.scale_to_spl)
  r_jit, p_jit = jitted(ref, proc, 70.0, 65.0)
  r_eager, p_eager = calibration.scale_to_spl(ref, proc, 70.0, 65.0)
  np.testing.assert_allclose(np.asarray(r_jit), np.asarray(r_eager), rtol=1e-12)
  np.testing.assert_allclose(np.asarray(p_jit), np.asarray(p_eager), rtol=1e-12)

  def loss(r, p, spl):
    rs, ps = calibration.scale_to_spl(r, p, target_spl=spl)
    return jnp.sum(rs * ps)

  grad_r, grad_p, grad_spl = jax.grad(loss, argnums=(0, 1, 2))(ref, proc, 70.0)
  assert bool(jnp.all(jnp.isfinite(grad_r)))
  assert bool(jnp.all(jnp.isfinite(grad_p)))
  assert bool(jnp.isfinite(grad_spl))

  eps = 1e-6
  fd_spl = (loss(ref, proc, 70.0 + eps) - loss(ref, proc, 70.0 - eps)) / (
    2 * eps
  )
  assert float(grad_spl) == pytest.approx(float(fd_spl), rel=1e-5)

  # Silent, inf, nan, and overflowing inputs must keep gradients finite.
  def proc_loss(r, p, spl):
    _, ps = calibration.scale_to_spl(r, p, target_spl=spl)
    return jnp.sum(ps)

  for bad_ref, spl in (
    (jnp.zeros_like(ref), 65.0),
    (ref.at[0].set(jnp.inf), 65.0),
    (ref.at[0].set(jnp.nan), 65.0),
    (ref, 1e4),
  ):
    gr, gp, gs = jax.grad(proc_loss, argnums=(0, 1, 2))(bad_ref, proc, spl)
    assert bool(jnp.all(jnp.isfinite(gr)))
    assert bool(jnp.all(jnp.isfinite(gp)))
    assert bool(jnp.isfinite(gs))

  # Empty reference gradient stays finite.
  ge_r, ge_p = jax.grad(lambda r, p: proc_loss(r, p, 65.0), argnums=(0, 1))(
    jnp.zeros(0, dtype=jnp.float64), proc
  )
  assert ge_r.shape == (0,)
  assert bool(jnp.all(jnp.isfinite(ge_p)))

  # Float32 quotient-overflow gradient stays finite.
  g32_r, g32_p, g32_s = jax.grad(proc_loss, argnums=(0, 1, 2))(
    jnp.full(16, 1e-12, dtype=jnp.float32),
    jnp.ones(16, dtype=jnp.float32),
    jnp.float32(665.0),
  )
  assert bool(jnp.all(jnp.isfinite(g32_r)))
  assert bool(jnp.all(jnp.isfinite(g32_p)))
  assert bool(jnp.isfinite(g32_s))


def test_apply_nalr_24khz_rejects_empty_and_non_1d_but_supports_vmap():
  levels = _levels("clinical")
  with pytest.raises(ValueError, match="non-empty 1-D"):
    calibration.apply_nalr_24khz(jnp.zeros((2, 100)), SR, levels)
  with pytest.raises(ValueError, match="non-empty 1-D"):
    calibration.apply_nalr_24khz(jnp.zeros(0), SR, levels)
  with pytest.raises(ValueError, match="non-empty 1-D"):
    calibration.apply_nalr_24khz(jnp.asarray(1.0), SR, levels)

  batch = jnp.stack(
    [
      jnp.asarray(_speech_like(duration=0.1, pad=0.0, seed=1)),
      jnp.asarray(_speech_like(duration=0.1, pad=0.0, seed=2)),
    ]
  )
  out = jax.vmap(calibration.apply_nalr_24khz, in_axes=(0, None, None))(
    batch, SR, levels
  )
  assert out.shape == batch.shape
  np.testing.assert_allclose(
    np.asarray(out[0]),
    np.asarray(calibration.apply_nalr_24khz(batch[0], SR, levels)),
    rtol=1e-12,
  )


def test_apply_nalr_24khz_is_identity_for_normal_hearing():
  signal = jnp.asarray(_speech_like(duration=0.25, pad=0.0))
  flat = _levels("flat_zero")

  out = calibration.apply_nalr_24khz(signal, SR, flat)
  assert out.shape == signal.shape
  np.testing.assert_array_equal(np.asarray(out), np.asarray(signal))

  # Better-than-zero dB HL audiogram also applies unity gain.
  negative = np.array([-5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
  out_neg = calibration.apply_nalr_24khz(signal, SR, negative)
  np.testing.assert_array_equal(np.asarray(out_neg), np.asarray(signal))


def test_apply_nalr_24khz_preserves_zero_group_delay_and_shapes_gain():
  rng = np.random.default_rng(10)
  noise = jnp.asarray(rng.standard_normal(SR))
  levels = _levels("sloping_steep")

  equalised = calibration.apply_nalr_24khz(noise, SR, levels)
  assert equalised.shape == noise.shape

  # Cross-correlation peak sits at lag 0 (zero group delay).
  corr = np.correlate(np.asarray(equalised), np.asarray(noise), mode="full")
  lag = int(np.argmax(corr)) - (len(noise) - 1)
  assert lag == 0

  # Verify per-band gain matches Byrne & Dillon (1986) prescription.
  t = np.arange(SR) / SR
  hl_250, hl_500, hl_1k, hl_2k, hl_4k, _ = levels
  x_ave = 0.05 * (hl_500 + hl_1k + hl_2k)
  want_250 = max(x_ave + 0.31 * hl_250 - 17.0, 0.0)
  want_4k = max(x_ave + 0.31 * hl_4k - 2.0, 0.0)

  for freq, want_db in ((250.0, want_250), (4000.0, want_4k)):
    tone = jnp.asarray(np.sin(2.0 * np.pi * freq * t))
    out = np.asarray(calibration.apply_nalr_24khz(tone, SR, levels))
    # Measure steady-state gain away from FIR edges.
    gain_db = 20.0 * np.log10(
      np.sqrt(np.mean(out[500:-500] ** 2))
      / np.sqrt(np.mean(np.asarray(tone)[500:-500] ** 2))
    )
    assert gain_db == pytest.approx(want_db, abs=0.5)


@pytest.mark.parametrize("sample_rate", (8000, 16000, 22050, 44100, 48000))
def test_apply_nalr_24khz_resamples_non_24khz_inputs(sample_rate):
  signal = _speech_like(sample_rate=sample_rate, duration=0.25, pad=0.05)
  levels = _levels("clinical")

  out = calibration.apply_nalr_24khz(signal, sample_rate, levels)
  expected_len = ear_model.resample_24khz(signal, sample_rate).shape[0]
  assert out.shape == (expected_len,)
  assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.parametrize("sample_rate", (16000, 24000, 48000))
def test_apply_nalr_24khz_preserves_signal_dtype(sample_rate):
  """Float64 audiogram arrays must not promote float32 waveforms."""
  signal = _speech_like(sample_rate=sample_rate, duration=0.1, pad=0.0)
  levels64 = _levels("clinical")

  out32 = calibration.apply_nalr_24khz(
    jnp.float32(signal), sample_rate, levels64
  )
  assert out32.dtype == jnp.float32

  out64 = calibration.apply_nalr_24khz(
    jnp.float64(signal), sample_rate, levels64
  )
  assert out64.dtype == jnp.float64


def test_apply_nalr_24khz_is_jittable_and_differentiable():
  signal = jnp.asarray(_speech_like(duration=0.1, pad=0.0))
  levels = jnp.asarray(_levels("clinical"))

  jitted = jax.jit(
    calibration.apply_nalr_24khz, static_argnames=("sample_rate",)
  )
  np.testing.assert_allclose(
    np.asarray(jitted(signal, SR, levels)),
    np.asarray(calibration.apply_nalr_24khz(signal, SR, levels)),
    rtol=1e-12,
  )

  sig16 = jnp.asarray(_speech_like(sample_rate=16000, duration=0.1, pad=0.0))
  np.testing.assert_allclose(
    np.asarray(jitted(sig16, 16000, levels)),
    np.asarray(calibration.apply_nalr_24khz(sig16, 16000, levels)),
    rtol=1e-12,
  )

  grad_sig, grad_hl = jax.grad(
    lambda s, hl: jnp.sum(calibration.apply_nalr_24khz(s, SR, hl) ** 2),
    argnums=(0, 1),
  )(signal, levels)
  assert bool(jnp.all(jnp.isfinite(grad_sig)))
  assert bool(jnp.all(jnp.isfinite(grad_hl)))
  assert float(jnp.max(jnp.abs(grad_sig))) > 0
  assert float(jnp.max(jnp.abs(grad_hl))) > 0

  # Silent input gradient stays finite across resampling.
  grad_zero = jax.grad(
    lambda s: jnp.sum(calibration.apply_nalr_24khz(s, 16000, levels) ** 2)
  )(jnp.zeros(1600, dtype=jnp.float64))
  assert bool(jnp.all(jnp.isfinite(grad_zero)))


@pytest.mark.parametrize(
  "audiogram", ("mild", "clinical", "moderate", "severe", "sloping_steep")
)
def test_aided_nalr_restores_impaired_haspi_and_hasqi_ceiling(audiogram):
  """Aided clean speech scores ~1.0; unaided equalisation=1 drops sharply."""
  clean = _speech_like()
  ref, proc = calibration.scale_to_spl(clean, clean)
  levels = _levels(audiogram)

  proc_aided = calibration.apply_nalr_24khz(proc, SR, levels)

  haspi_aided = float(haspi.haspi_v2(ref, SR, proc_aided, SR, levels)[0])
  hasqi_aided = float(
    hasqi.hasqi_v2(ref, SR, proc_aided, SR, levels, equalisation=1)[0]
  )
  hasqi_unaided_eq1 = float(
    hasqi.hasqi_v2(ref, SR, proc, SR, levels, equalisation=1)[0]
  )
  hasqi_unaided_eq2 = float(
    hasqi.hasqi_v2(ref, SR, proc, SR, levels, equalisation=2)[0]
  )

  assert haspi_aided > 0.95
  assert hasqi_aided > 0.99
  assert hasqi_unaided_eq2 > 0.99
  assert hasqi_unaided_eq1 < 0.65
  if audiogram == "severe":
    haspi_unaided = float(haspi.haspi_v2(ref, SR, proc, SR, levels)[0])
    assert haspi_aided > haspi_unaided + 0.05


def test_aided_nalr_works_end_to_end_at_48khz_and_16khz():
  levels = _levels("sloping_steep")
  for sr in (16000, 48000):
    clean = _speech_like(sample_rate=sr)
    ref, proc = calibration.scale_to_spl(clean, clean)
    proc_aided = calibration.apply_nalr_24khz(proc, sr, levels)

    hp = float(
      haspi.haspi_v2(
        ref, sr, proc_aided, calibration.MODEL_SAMPLE_RATE, levels
      )[0]
    )
    hq = float(
      hasqi.hasqi_v2(
        ref,
        sr,
        proc_aided,
        calibration.MODEL_SAMPLE_RATE,
        levels,
        equalisation=1,
      )[0]
    )
    hq_unaided = float(
      hasqi.hasqi_v2(ref, sr, proc, sr, levels, equalisation=1)[0]
    )
    assert hp > 0.99
    assert hq > 0.99
    assert hq_unaided < 0.5


def test_aided_impaired_scores_fall_monotonically_with_snr():
  clean = _speech_like()
  rng = np.random.default_rng(7)
  noise = rng.standard_normal(len(clean))
  noise = noise / np.sqrt(np.mean(noise**2)) * np.sqrt(np.mean(clean**2))
  levels = _levels("severe")

  haspi_scores = []
  hasqi_scores = []
  for snr_db in (None, 10.0, 5.0, -5.0):
    proc = clean if snr_db is None else clean + noise * (10.0 ** (-snr_db / 20))
    ref_c, proc_c = calibration.scale_to_spl(clean, proc)
    proc_a = calibration.apply_nalr_24khz(proc_c, SR, levels)
    haspi_scores.append(float(haspi.haspi_v2(ref_c, SR, proc_a, SR, levels)[0]))
    hasqi_scores.append(
      float(hasqi.hasqi_v2(ref_c, SR, proc_a, SR, levels, equalisation=1)[0])
    )

  assert haspi_scores == sorted(haspi_scores, reverse=True)
  assert hasqi_scores == sorted(hasqi_scores, reverse=True)
  assert haspi_scores[0] - haspi_scores[-1] > 0.5
  assert hasqi_scores[0] - hasqi_scores[-1] > 0.5
