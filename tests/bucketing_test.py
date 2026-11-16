"""Opt-in length bucketing: parity, compilation reuse, transforms, and edges."""

import contextlib
import inspect
import pathlib
import re
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi

LEVELS = np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0])
_FFT_LINE = re.compile(r"^\s*%\S+ = stablehlo\.fft .*?:(?P<types>.*)$")
_FLOAT_WIDTH = re.compile(r"\bf(?P<width>\d+)\b")


def _signal_pair(n, sample_rate, seed=0):
  rng = np.random.default_rng(seed)
  t = np.arange(n) / sample_rate
  ref = (
    0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
  )
  proc = ref + 0.02 * rng.standard_normal(n)
  return ref, proc


@contextlib.contextmanager
def _captured_ir_dumps():
  with tempfile.TemporaryDirectory() as directory:
    jax.config.update("jax_dump_ir_to", directory)
    modules = []
    try:
      yield modules
    finally:
      jax.config.update("jax_dump_ir_to", None)
    modules.extend(
      path.read_text() for path in sorted(pathlib.Path(directory).iterdir())
    )


def _cache_snapshot():
  return (
    ear_model._resample_24khz_jitted._cache_size(),
    ear_model.input_align._cache_size(),
    ear_model._ear_model_aligned._cache_size(),
    haspi._haspi_backend._cache_size(),
    hasqi._hasqi_backend._cache_size(),
  )


def test_multiple_buckets_compile_once_per_bucket():
  """Six clips spanning three 0.25 s buckets compile exactly three binaries."""
  jax.clear_caches()
  durations = (0.15, 0.25, 0.30, 0.48, 0.55, 0.70)
  for i, sec in enumerate(durations):
    ref, proc = _signal_pair(int(round(24000 * sec)), 24000, seed=i)
    jax.block_until_ready(
      haspi.haspi_v2(ref, 24000, proc, 24000, LEVELS, bucket_seconds=0.25)
    )

  assert ear_model.input_align._cache_size() == 3
  assert ear_model._ear_model_aligned._cache_size() == 3
  assert haspi._haspi_backend._cache_size() == 3


@pytest.mark.parametrize("sample_rate", [24000, 16000, 48000])
def test_single_bucket_compiles_once_across_lengths_and_rates(sample_rate):
  """Clips in the same bucket share one compiled binary at every stage."""
  jax.clear_caches()
  fractions = (0.20, 0.31, 0.42, 0.50)
  clips = [
    _signal_pair(int(round(sample_rate * frac)), sample_rate, seed=i)
    for i, frac in enumerate(fractions)
  ]
  jnp_clips = [(jnp.asarray(r), jnp.asarray(p)) for r, p in clips[1:]]

  # Warm the 0.5 s bucket on clip 0.
  before_resample = ear_model._resample_24khz_jitted._cache_size()
  jax.block_until_ready(
    haspi.haspi_v2(
      clips[0][0],
      sample_rate,
      clips[0][1],
      sample_rate,
      LEVELS,
      bucket_seconds=0.5,
    )
  )
  jax.block_until_ready(
    hasqi.hasqi_v2(
      clips[0][0],
      sample_rate,
      clips[0][1],
      sample_rate,
      LEVELS,
      bucket_seconds=0.5,
    )
  )
  expected_resample_delta = 0 if sample_rate == 24000 else 1
  assert (
    ear_model._resample_24khz_jitted._cache_size() - before_resample
    == expected_resample_delta
  )

  # Subsequent varying lengths (NumPy and jax.Array) must emit zero lowerings.
  warm_caches = _cache_snapshot()
  with _captured_ir_dumps() as modules:
    for ref, proc in (*clips[1:], *jnp_clips):
      jax.block_until_ready(
        haspi.haspi_v2(
          ref, sample_rate, proc, sample_rate, LEVELS, bucket_seconds=0.5
        )
      )
      jax.block_until_ready(
        hasqi.hasqi_v2(
          ref, sample_rate, proc, sample_rate, LEVELS, bucket_seconds=0.5
        )
      )

  assert modules == []
  assert _cache_snapshot() == warm_caches


def test_seeded_noise_does_not_recompile_within_bucket():
  """Varying-length BM and cepstral noise share one bucket compilation."""
  rng = np.random.default_rng(19)
  lengths = (6000, 8500, 11000)
  inputs = []
  for i, n in enumerate(lengths):
    ref, proc = _signal_pair(n, 24000, seed=i)
    bm_r = rng.standard_normal((32, n))
    bm_p = rng.standard_normal((32, n))
    c_len = max(n // 16, 10)
    cep = (rng.standard_normal((c_len, 32)), rng.standard_normal((c_len, 32)))
    inputs.append((ref, proc, bm_r, bm_p, cep))

  ref0, proc0, bm_r0, bm_p0, cep0 = inputs[0]
  jax.block_until_ready(
    haspi.haspi_v2(
      ref0,
      24000,
      proc0,
      24000,
      LEVELS,
      reference_noise=bm_r0,
      processed_noise=bm_p0,
      cepstral_noise=cep0,
      bucket_seconds=0.5,
    )
  )

  warm_caches = _cache_snapshot()
  with _captured_ir_dumps() as modules:
    for ref, proc, bm_r, bm_p, cep in inputs[1:]:
      got, _ = haspi.haspi_v2(
        ref,
        24000,
        proc,
        24000,
        LEVELS,
        reference_noise=bm_r,
        processed_noise=bm_p,
        cepstral_noise=cep,
        bucket_seconds=0.5,
      )
      jax.block_until_ready(got)

  assert modules == []
  assert _cache_snapshot() == warm_caches

  ref1, proc1, bm_r1, bm_p1, cep1 = inputs[1]
  want, _ = haspi.haspi_v2(
    ref1,
    24000,
    proc1,
    24000,
    LEVELS,
    reference_noise=bm_r1,
    processed_noise=bm_p1,
    cepstral_noise=cep1,
  )
  got, _ = haspi.haspi_v2(
    ref1,
    24000,
    proc1,
    24000,
    LEVELS,
    reference_noise=bm_r1,
    processed_noise=bm_p1,
    cepstral_noise=cep1,
    bucket_seconds=0.5,
  )
  assert abs(float(got) - float(want)) <= 1e-12


@pytest.mark.parametrize(
  ("sample_rate", "n_samples", "bucket_seconds"),
  [
    # 8 kHz with step=1000 gives L_out=6999 < 7000, covering post-resample pad.
    (8000, 2100, 1000 / 24000),
    (16000, 4000, 0.5),
    (22050, 5500, 0.5),
    (24000, 6000, 0.5),
    (32000, 8000, 0.5),
    (44100, 11025, 0.5),
    (48000, 12000, 0.5),
  ],
)
def test_haspi_and_hasqi_match_unbucketed_across_sample_rates(
  sample_rate, n_samples, bucket_seconds
):
  """Bucketing matches unbucketed scores and raw terms across sample rates."""
  ref, proc = _signal_pair(n_samples, sample_rate, seed=sample_rate)

  h_base, c_base = haspi.haspi_v2(ref, sample_rate, proc, sample_rate, LEVELS)
  h_bkt, c_bkt = haspi.haspi_v2(
    ref, sample_rate, proc, sample_rate, LEVELS, bucket_seconds=bucket_seconds
  )
  assert abs(float(h_base) - float(h_bkt)) <= 1e-12
  np.testing.assert_allclose(np.asarray(c_bkt), np.asarray(c_base), atol=1e-12)

  q_base, nl_base, lin_base, raw_base = hasqi.hasqi_v2(
    ref, sample_rate, proc, sample_rate, LEVELS
  )
  q_bkt, nl_bkt, lin_bkt, raw_bkt = hasqi.hasqi_v2(
    ref, sample_rate, proc, sample_rate, LEVELS, bucket_seconds=bucket_seconds
  )
  assert abs(float(q_base) - float(q_bkt)) <= 1e-12
  assert abs(float(nl_base) - float(nl_bkt)) <= 1e-12
  assert abs(float(lin_base) - float(lin_bkt)) <= 1e-12
  np.testing.assert_allclose(
    np.asarray(raw_bkt), np.asarray(raw_base), atol=1e-12
  )


@pytest.mark.parametrize(
  "name",
  [
    "speech_snr20",
    "audiogram_clinical",
    "rate_16000",
    "rate_44100",
    "rate_mismatch_24k_16k",
    "unequal_length",
    "duration_0.05s",
    "itype0",
    "itype1",
    "itype2",
    "both_silent",
    "reference_silent",
    "very_quiet",
    "processed_silent",
  ],
)
def test_golden_cases_match_unbucketed_with_bucketing(name):
  """Representative golden cases agree to <= 1e-12 when bucketed."""
  case = goldens.case(name)
  h_base, _ = haspi.haspi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    level1=case.level1,
    itype=case.itype,
  )
  h_bkt, _ = haspi.haspi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    level1=case.level1,
    itype=case.itype,
    bucket_seconds=0.5,
  )
  q_base, *_ = hasqi.hasqi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    equalisation=case.equalisation,
    level1=case.level1,
  )
  q_bkt, *_ = hasqi.hasqi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    equalisation=case.equalisation,
    level1=case.level1,
    bucket_seconds=0.5,
  )
  assert abs(float(h_base) - float(h_bkt)) <= 1e-12
  assert abs(float(q_base) - float(q_bkt)) <= 1e-12


def test_unequal_lengths_across_different_buckets_match_unbucketed():
  """Signals landing in different resampling buckets still match unbucketed."""
  rng = np.random.default_rng(9)
  t_long = np.arange(36000) / 48000
  t_short = np.arange(12000) / 48000
  ref = 0.3 * np.sin(2 * np.pi * 220 * t_long)
  proc = 0.3 * np.sin(2 * np.pi * 220 * t_short) + 0.02 * rng.standard_normal(
    len(t_short)
  )

  h_base, _ = haspi.haspi_v2(ref, 48000, proc, 48000, LEVELS)
  h_bkt, _ = haspi.haspi_v2(ref, 48000, proc, 48000, LEVELS, bucket_seconds=0.5)
  q_base, *_ = hasqi.hasqi_v2(ref, 48000, proc, 48000, LEVELS)
  q_bkt, *_ = hasqi.hasqi_v2(
    ref, 48000, proc, 48000, LEVELS, bucket_seconds=0.5
  )
  assert abs(float(h_base) - float(h_bkt)) <= 1e-12
  assert abs(float(q_base) - float(q_bkt)) <= 1e-12


@pytest.mark.parametrize("sample_rate", [24000, 16000, 48000])
def test_float32_matches_unbucketed_with_bucketing(sample_rate):
  """float32 inputs keep float32 dtype and float32-grade parity when bucketed."""
  ref, proc = _signal_pair(int(sample_rate * 0.25), sample_rate, seed=5)
  ref32, proc32, lv32 = np.float32(ref), np.float32(proc), np.float32(LEVELS)

  h_base, _ = haspi.haspi_v2(ref32, sample_rate, proc32, sample_rate, lv32)
  h_bkt, _ = haspi.haspi_v2(
    ref32, sample_rate, proc32, sample_rate, lv32, bucket_seconds=0.5
  )
  q_base, *_ = hasqi.hasqi_v2(ref32, sample_rate, proc32, sample_rate, lv32)
  q_bkt, *_ = hasqi.hasqi_v2(
    ref32, sample_rate, proc32, sample_rate, lv32, bucket_seconds=0.5
  )
  assert h_bkt.dtype == jnp.float32
  assert q_bkt.dtype == jnp.float32
  # 1-ULP float32 sum-order diffs can flip a 1-sample lag tie in envelope_align.
  assert abs(float(h_base) - float(h_bkt)) <= 5e-3
  assert abs(float(q_base) - float(q_bkt)) <= 5e-3


@pytest.mark.parametrize("itype", [0, 1, 2])
def test_run_ear_model_bucketed_shape_and_values(itype):
  """_run_ear_model pads 2-D outputs to bucket multiples with exact zero tails."""
  assert (
    "bucket_seconds" not in inspect.signature(ear_model.ear_model).parameters
  )
  ref, proc = _signal_pair(6000, 24000, seed=itype)
  base = ear_model._run_ear_model(ref, 24000, proc, 24000, LEVELS, itype, 65.0)
  bkt = ear_model._run_ear_model(
    ref, 24000, proc, 24000, LEVELS, itype, 65.0, bucket_seconds=0.5
  )

  for idx in range(4):
    assert base[idx].shape == (32, 6000)
    assert bkt[idx].shape == (32, 12000)
    np.testing.assert_allclose(
      np.asarray(bkt[idx][:, :6000]), np.asarray(base[idx]), atol=1e-12
    )
    np.testing.assert_array_equal(np.asarray(bkt[idx][:, 6000:]), 0.0)
  for idx in (4, 5):
    np.testing.assert_allclose(
      np.asarray(bkt[idx]), np.asarray(base[idx]), atol=1e-12
    )
  assert bkt[6] == 24000.0
  assert int(bkt[7]) == int(base[7])


def test_clip_length_boundary_and_rollover():
  """Exact bucket boundaries stay in bucket; +1 sample rolls to next multiple."""
  exact_ref, exact_proc = _signal_pair(6000, 24000, seed=1)
  over_ref, over_proc = _signal_pair(6001, 24000, seed=2)

  exact_out = ear_model._run_ear_model(
    exact_ref, 24000, exact_proc, 24000, LEVELS, 0, 65.0, bucket_seconds=0.25
  )
  over_out = ear_model._run_ear_model(
    over_ref, 24000, over_proc, 24000, LEVELS, 0, 65.0, bucket_seconds=0.25
  )
  assert exact_out[0].shape == (32, 6000)
  assert over_out[0].shape == (32, 12000)


def test_haspi_v2_better_ear_bucketed_matches_unbucketed():
  """haspi_v2_better_ear forwards bucket_seconds and shares compilations."""
  warm_l, warm_pl = _signal_pair(5000, 24000, seed=12)
  warm_r, warm_pr = _signal_pair(7000, 24000, seed=13)
  jax.block_until_ready(
    haspi.haspi_v2_better_ear(
      warm_l,
      warm_r,
      warm_pl,
      warm_pr,
      24000,
      LEVELS,
      LEVELS * 0.5,
      bucket_seconds=0.5,
    )
  )

  ref_l, proc_l = _signal_pair(6000, 24000, seed=10)
  ref_r, proc_r = _signal_pair(9000, 24000, seed=11)
  want = haspi.haspi_v2_better_ear(
    ref_l, ref_r, proc_l, proc_r, 24000, LEVELS, LEVELS * 0.5
  )
  warm_caches = _cache_snapshot()
  with _captured_ir_dumps() as modules:
    got = haspi.haspi_v2_better_ear(
      ref_l,
      ref_r,
      proc_l,
      proc_r,
      24000,
      LEVELS,
      LEVELS * 0.5,
      bucket_seconds=0.5,
    )
    jax.block_until_ready(got)
  assert abs(float(got) - float(want)) <= 1e-12
  assert modules == []
  assert _cache_snapshot() == warm_caches


@pytest.mark.parametrize(
  "bad_value",
  [
    0,
    -1.0,
    float("nan"),
    float("inf"),
    -float("inf"),
    True,
    False,
    "1.0",
    1e-9,
  ],
)
def test_invalid_bucket_seconds_raises(bad_value):
  ref, proc = _signal_pair(1200, 24000, seed=0)
  with pytest.raises(ValueError, match="bucket_seconds"):
    haspi.haspi_v2(ref, 24000, proc, 24000, LEVELS, bucket_seconds=bad_value)
  with pytest.raises(ValueError, match="bucket_seconds"):
    hasqi.hasqi_v2(ref, 24000, proc, 24000, LEVELS, bucket_seconds=bad_value)
  with pytest.raises(ValueError, match="bucket_seconds"):
    haspi.haspi_v2_better_ear(
      ref, ref, proc, proc, 24000, LEVELS, LEVELS, bucket_seconds=bad_value
    )


def test_short_and_degenerate_clips_with_bucketing():
  """Short (250, 350, 400), low-threshold, silent, and NaN clips match."""
  rng = np.random.default_rng(7)
  cases = (
    (250, LEVELS, 65.0),
    (350, np.zeros(6), 100.0),
    (400, LEVELS, 65.0),
  )
  for n, levels, level1 in cases:
    ref = rng.standard_normal(n) * 0.2
    proc = ref + 0.01 * rng.standard_normal(n)
    h_base, _ = haspi.haspi_v2(ref, 24000, proc, 24000, levels, level1=level1)
    h_bkt, _ = haspi.haspi_v2(
      ref, 24000, proc, 24000, levels, level1=level1, bucket_seconds=0.5
    )
    assert abs(float(h_base) - float(h_bkt)) <= 1e-12

    q_base, nl_base, lin_base, raw_base = hasqi.hasqi_v2(
      ref, 24000, proc, 24000, levels, level1=level1
    )
    q_bkt, nl_bkt, lin_bkt, raw_bkt = hasqi.hasqi_v2(
      ref, 24000, proc, 24000, levels, level1=level1, bucket_seconds=0.5
    )
    assert abs(float(q_base) - float(q_bkt)) <= 1e-12
    assert abs(float(nl_base) - float(nl_bkt)) <= 1e-12
    assert abs(float(lin_base) - float(lin_bkt)) <= 1e-12
    np.testing.assert_allclose(
      np.asarray(raw_bkt), np.asarray(raw_base), atol=1e-12
    )

  # Low silence_threshold below -100 dB must still exclude padded frames.
  r_low, p_low = _signal_pair(6000, 24000, seed=14)
  q_low_base, *_ = hasqi.hasqi_v2(
    r_low, 24000, p_low, 24000, LEVELS, silence_threshold=-200.0
  )
  q_low_bkt, *_ = hasqi.hasqi_v2(
    r_low,
    24000,
    p_low,
    24000,
    LEVELS,
    silence_threshold=-200.0,
    bucket_seconds=0.5,
  )
  assert abs(float(q_low_base) - float(q_low_bkt)) <= 1e-12

  nan_ref = np.full(2400, np.nan)
  ones = np.ones(2400)
  assert (
    float(
      haspi.haspi_v2(nan_ref, 24000, ones, 24000, LEVELS, bucket_seconds=0.5)[0]
    )
    == 0.0
  )
  assert (
    float(
      hasqi.hasqi_v2(nan_ref, 24000, ones, 24000, LEVELS, bucket_seconds=0.5)[0]
    )
    == 0.0
  )


@pytest.mark.parametrize("sample_rate", [48000, 96000])
def test_short_high_rate_resampling_does_not_crash(sample_rate):
  """Short downsampled clips with up==1 and n < len(taps) run cleanly."""
  x = np.linspace(-0.2, 0.2, 10)
  out = ear_model.resample_24khz(x, sample_rate)
  assert np.all(np.isfinite(np.asarray(out)))

  h_base, _ = haspi.haspi_v2(x, sample_rate, x, sample_rate, LEVELS)
  h_bkt, _ = haspi.haspi_v2(
    x, sample_rate, x, sample_rate, LEVELS, bucket_seconds=0.01
  )
  q_bkt, *_ = hasqi.hasqi_v2(
    x, sample_rate, x, sample_rate, LEVELS, bucket_seconds=0.01
  )
  assert abs(float(h_base) - float(h_bkt)) <= 1e-12
  assert np.isfinite(float(q_bkt))


@pytest.mark.parametrize("sample_rate", [16000, 24000, 48000])
def test_empty_clip_with_single_sample_bucket(sample_rate):
  """Empty clips with 1-sample buckets slice resampled output to target_len."""
  empty = np.zeros(0)
  out = ear_model._run_ear_model(
    empty,
    sample_rate,
    empty,
    sample_rate,
    LEVELS,
    0,
    65.0,
    bucket_seconds=1 / 24000,
  )
  assert out[0].shape == (32, 1)
  assert int(out[7]) == 0

  h, _ = haspi.haspi_v2(
    empty, sample_rate, empty, sample_rate, LEVELS, bucket_seconds=0.05
  )
  q, *_ = hasqi.hasqi_v2(
    empty, sample_rate, empty, sample_rate, LEVELS, bucket_seconds=0.05
  )
  assert float(h) == 0.0
  assert float(q) == 0.0


def test_input_align_and_envelope_align_valid_len_edge_cases():
  """valid_len isolates active prefix from tail values, NaNs, and tiny lengths."""
  r3 = jnp.array([1.0, -2.0, 0.5])
  p3 = jnp.array([-0.5, 1.0, -2.0])
  r3_pad = jnp.concatenate([r3, jnp.full(200, np.nan)])
  p3_pad = jnp.concatenate([p3, jnp.full(200, 99.0)])

  a1, b1, m1 = ear_model.input_align(r3, p3)
  a2, b2, m2 = ear_model.input_align(r3_pad, p3_pad, valid_len=jnp.int32(3))
  assert [int(v) for v in m1] == [int(v) for v in m2]
  np.testing.assert_array_equal(np.asarray(a2[:3]), np.asarray(a1))
  np.testing.assert_array_equal(np.asarray(b2[:3]), np.asarray(b1))
  np.testing.assert_array_equal(np.asarray(a2[3:]), 0.0)
  np.testing.assert_array_equal(np.asarray(b2[3:]), 0.0)

  # _resample_24khz_jitted masks non-zero/NaN tails before filtering.
  sig = jnp.linspace(-0.2, 0.2, 40)
  sig_dirty = jnp.concatenate([sig, jnp.full(40, np.nan)])
  res_clean = ear_model.resample_24khz(sig, 16000)
  res_masked = ear_model._resample_24khz_jitted(
    sig_dirty,
    16,
    valid_in=jnp.int32(40),
    valid_out=jnp.int32(60),
    target_len=120,
  )
  np.testing.assert_allclose(
    np.asarray(res_masked[:60]), np.asarray(res_clean), atol=1e-12
  )
  np.testing.assert_array_equal(np.asarray(res_masked[60:]), 0.0)

  # valid_len=0 returns empty alignment even when tail is non-zero.
  a0, b0, (s0, e0, d0) = ear_model.input_align(
    jnp.ones(64), jnp.ones(64), valid_len=jnp.int32(0)
  )
  assert (int(s0), int(e0), int(d0)) == (0, -1, 0)
  np.testing.assert_array_equal(np.asarray(a0), 0.0)
  np.testing.assert_array_equal(np.asarray(b0), 0.0)

  # envelope_align ignores out-of-range lags for tiny valid_len.
  e1 = ear_model.envelope_align(r3, p3)
  e2 = ear_model.envelope_align(
    jnp.pad(r3, (0, 200)), jnp.pad(p3, (0, 200)), valid_len=jnp.int32(3)
  )
  np.testing.assert_allclose(np.asarray(e2[:3]), np.asarray(e1), atol=1e-12)
  np.testing.assert_array_equal(np.asarray(e2[3:]), 0.0)


@pytest.mark.parametrize("sample_rate", [24000, 16000])
def test_gradients_match_unbucketed(sample_rate):
  """jax.grad through bucketed metrics is finite and matches unbucketed."""
  ref, proc = _signal_pair(int(sample_rate * 0.25), sample_rate, seed=3)
  ref_j, proc_j = jnp.asarray(ref), jnp.asarray(proc)

  gh_base = jax.grad(
    lambda p: haspi.haspi_v2(ref_j, sample_rate, p, sample_rate, LEVELS)[0]
  )(proc_j)
  gh_bkt = jax.grad(
    lambda p: haspi.haspi_v2(
      ref_j, sample_rate, p, sample_rate, LEVELS, bucket_seconds=0.5
    )[0]
  )(proc_j)
  assert bool(jnp.all(jnp.isfinite(gh_bkt)))
  assert float(jnp.max(jnp.abs(gh_bkt))) > 0
  np.testing.assert_allclose(
    np.asarray(gh_bkt), np.asarray(gh_base), atol=1e-12
  )

  gq_base = jax.grad(
    lambda p: hasqi.hasqi_v2(ref_j, sample_rate, p, sample_rate, LEVELS)[0]
  )(proc_j)
  gq_bkt = jax.grad(
    lambda p: hasqi.hasqi_v2(
      ref_j, sample_rate, p, sample_rate, LEVELS, bucket_seconds=0.5
    )[0]
  )(proc_j)
  assert bool(jnp.all(jnp.isfinite(gq_bkt)))
  assert float(jnp.max(jnp.abs(gq_bkt))) > 0
  np.testing.assert_allclose(
    np.asarray(gq_bkt), np.asarray(gq_base), atol=1e-12
  )


def test_outer_jit_with_bucket_seconds():
  """haspi_v2 and hasqi_v2 compose cleanly inside an outer jax.jit."""
  jitted_haspi = jax.jit(
    haspi.haspi_v2,
    static_argnames=(
      "reference_rate",
      "processed_rate",
      "f_lp",
      "itype",
      "bucket_seconds",
    ),
  )
  jitted_hasqi = jax.jit(
    hasqi.hasqi_v2,
    static_argnames=(
      "reference_rate",
      "processed_rate",
      "equalisation",
      "silence_threshold",
      "segment_covariance",
      "bucket_seconds",
    ),
  )
  case_a = goldens.case("speech_snr20")
  case_b = goldens.case("both_silent")
  ha, _ = jitted_haspi(
    jnp.asarray(case_a.reference),
    case_a.reference_sample_rate,
    jnp.asarray(case_a.processed),
    case_a.processed_sample_rate,
    jnp.asarray(case_a.levels),
    bucket_seconds=0.5,
  )
  hb, _ = jitted_haspi(
    jnp.asarray(case_b.reference),
    case_b.reference_sample_rate,
    jnp.asarray(case_b.processed),
    case_b.processed_sample_rate,
    jnp.asarray(case_b.levels),
    bucket_seconds=0.5,
  )
  qa, *_ = jitted_hasqi(
    jnp.asarray(case_a.reference),
    case_a.reference_sample_rate,
    jnp.asarray(case_a.processed),
    case_a.processed_sample_rate,
    jnp.asarray(case_a.levels),
    bucket_seconds=0.5,
  )
  qb, *_ = jitted_hasqi(
    jnp.asarray(case_b.reference),
    case_b.reference_sample_rate,
    jnp.asarray(case_b.processed),
    case_b.processed_sample_rate,
    jnp.asarray(case_b.levels),
    bucket_seconds=0.5,
  )
  assert float(ha) == pytest.approx(case_a.score("haspi"), rel=1e-7)
  assert float(hb) == 0.0
  assert float(qa) == pytest.approx(case_a.score("hasqi"), rel=1e-7)
  assert float(qb) == 0.0
  assert jitted_haspi._cache_size() == 1
  assert jitted_hasqi._cache_size() == 1


def test_bucketed_emits_no_float64_fft():
  """Bucketed lowering keeps every StableHLO FFT operand at <= float32."""
  ref, proc = _signal_pair(4000, 16000, seed=13)
  jax.clear_caches()
  with _captured_ir_dumps() as modules:
    jax.block_until_ready(
      haspi.haspi_v2(ref, 16000, proc, 16000, LEVELS, bucket_seconds=0.5)
    )
    jax.block_until_ready(
      hasqi.hasqi_v2(ref, 16000, proc, 16000, LEVELS, bucket_seconds=0.5)
    )

  assert modules
  widths = []
  for module in modules:
    for line in module.splitlines():
      match = _FFT_LINE.match(line)
      if match is not None:
        widths.extend(
          int(w) for w in _FLOAT_WIDTH.findall(match.group("types"))
        )
  assert widths
  assert max(widths) <= 32
