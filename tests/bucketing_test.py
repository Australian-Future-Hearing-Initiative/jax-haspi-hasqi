"""Length-bucketing tests for HASPI and HASQI."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi

_LEVELS = np.array([20.0, 20.0, 25.0, 35.0, 45.0, 50.0])


def _signal_pair(n, sample_rate, seed=0):
  rng = np.random.default_rng(seed)
  t = np.arange(n, dtype=np.float64) / sample_rate
  ref = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1100 * t)
  proc = np.roll(ref, min(12, max(n - 1, 0))) + 0.05 * rng.standard_normal(n)
  return ref, proc


def _cache_snapshot():
  return {
    "resample": ear_model._resample_24khz._cache_size(),
    "align": ear_model.input_align._cache_size(),
    "ear": ear_model._ear_model_aligned._cache_size(),
    "haspi": haspi._haspi_backend._cache_size(),
    "hasqi": hasqi._hasqi_backend._cache_size(),
  }


@pytest.mark.parametrize("sample_rate", (16000, 24000, 48000))
def test_compiles_once_per_bucket(sample_rate):
  if sample_rate == 16000:
    jax.clear_caches()
  step = sample_rate // 4
  # Three lengths in bucket 1 (including exact boundary), zero recompiles.
  for i, n in enumerate((int(0.55 * step), int(0.83 * step), step)):
    ref, proc = _signal_pair(n, sample_rate, seed=i)
    haspi.haspi_v2(
      ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
    )
    hasqi.hasqi_v2(
      ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
    )
    if i == 0:
      after_first = _cache_snapshot()
    else:
      assert _cache_snapshot() == after_first

  # Crossing into bucket 2 compiles one additional entry per stage.
  if sample_rate == 16000:
    ref, proc = _signal_pair(step + 1, sample_rate, seed=99)
    haspi.haspi_v2(
      ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
    )
    hasqi.hasqi_v2(
      ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
    )
    after_second = _cache_snapshot()
    assert after_second["resample"] == after_first["resample"] + 1
    assert after_second["align"] == after_first["align"] + 1
    assert after_second["ear"] > after_first["ear"]
    assert after_second["haspi"] == after_first["haspi"] + 1
    assert after_second["hasqi"] == after_first["hasqi"] + 1


@pytest.mark.parametrize("sample_rate", (16000, 24000, 48000))
def test_matches_unbucketed_across_rates(sample_rate):
  ref, proc = _signal_pair(int(0.43 * sample_rate), sample_rate, seed=3)
  h_base, corr_base = haspi.haspi_v2(
    ref, sample_rate, proc, sample_rate, _LEVELS
  )
  h_buck, corr_buck = haspi.haspi_v2(
    ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
  )
  assert float(h_buck) == pytest.approx(float(h_base), abs=1e-12)
  np.testing.assert_allclose(corr_buck, corr_base, atol=1e-12, rtol=1e-12)

  q_base = hasqi.hasqi_v2(ref, sample_rate, proc, sample_rate, _LEVELS)
  q_buck = hasqi.hasqi_v2(
    ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
  )
  for got, want in zip(q_buck, q_base, strict=True):
    np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-12)


def test_unequal_lengths_and_goldens_match_unbucketed():
  ref, _ = _signal_pair(int(0.65 * 16000), 16000, seed=4)
  _, proc = _signal_pair(int(0.2 * 16000), 16000, seed=5)
  h0, _ = haspi.haspi_v2(ref, 16000, proc, 16000, _LEVELS)
  h1, _ = haspi.haspi_v2(ref, 16000, proc, 16000, _LEVELS, bucket_seconds=0.25)
  q0, *_ = hasqi.hasqi_v2(ref, 16000, proc, 16000, _LEVELS)
  q1, *_ = hasqi.hasqi_v2(ref, 16000, proc, 16000, _LEVELS, bucket_seconds=0.25)
  assert float(h1) == pytest.approx(float(h0), abs=1e-12)
  assert float(q1) == pytest.approx(float(q0), abs=1e-12)

  for name in (
    "speech_snr5",
    "speech_snr20",
    "reference_silent",
    "processed_silent",
  ):
    c = goldens.case(name)
    args = (
      c.reference,
      c.reference_sample_rate,
      c.processed,
      c.processed_sample_rate,
      c.levels,
    )
    kw = {"level1": c.level1}
    assert float(
      haspi.haspi_v2(*args, **kw, bucket_seconds=1.0)[0]
    ) == pytest.approx(float(haspi.haspi_v2(*args, **kw)[0]), abs=1e-12)
    assert float(
      hasqi.hasqi_v2(*args, **kw, bucket_seconds=1.0)[0]
    ) == pytest.approx(float(hasqi.hasqi_v2(*args, **kw)[0]), abs=1e-12)

  # Better-ear wrapper forwards bucket_seconds cleanly.
  ref_s, proc_s = _signal_pair(4500, 24000, seed=6)
  b0 = haspi.haspi_v2_better_ear(
    ref_s, ref_s, proc_s, proc_s, 24000, _LEVELS, _LEVELS
  )
  b1 = haspi.haspi_v2_better_ear(
    ref_s, ref_s, proc_s, proc_s, 24000, _LEVELS, _LEVELS, bucket_seconds=0.25
  )
  assert float(b1) == pytest.approx(float(b0), abs=1e-12)


@pytest.mark.parametrize(
  "n,sample_rate", [(3, 24000), (250, 24000), (8, 48000), (8, 96000)]
)
def test_short_and_degenerate_clips(n, sample_rate):
  ref, proc = _signal_pair(n, sample_rate, seed=7)
  h0, _ = haspi.haspi_v2(ref, sample_rate, proc, sample_rate, _LEVELS)
  h1, _ = haspi.haspi_v2(
    ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
  )
  assert float(h1) == pytest.approx(float(h0), abs=1e-12)
  # Unbucketed HASQI requires at least one 192-sample half-window at 24 kHz.
  if (n * 24000) // sample_rate >= 192:
    q0, *_ = hasqi.hasqi_v2(ref, sample_rate, proc, sample_rate, _LEVELS)
    q1, *_ = hasqi.hasqi_v2(
      ref, sample_rate, proc, sample_rate, _LEVELS, bucket_seconds=0.25
    )
    assert float(q1) == pytest.approx(float(q0), abs=1e-12)


def test_empty_and_silent_clips():
  empty = np.zeros(0, dtype=np.float64)
  assert (
    float(
      haspi.haspi_v2(empty, 24000, empty, 24000, _LEVELS, bucket_seconds=0.25)[
        0
      ]
    )
    == 0.0
  )
  assert (
    float(
      hasqi.hasqi_v2(empty, 24000, empty, 24000, _LEVELS, bucket_seconds=0.25)[
        0
      ]
    )
    == 0.0
  )

  zeros = np.zeros(4321, dtype=np.float64)
  assert (
    float(
      haspi.haspi_v2(zeros, 16000, zeros, 16000, _LEVELS, bucket_seconds=0.25)[
        0
      ]
    )
    == 0.0
  )
  assert (
    float(
      hasqi.hasqi_v2(zeros, 16000, zeros, 16000, _LEVELS, bucket_seconds=0.25)[
        0
      ]
    )
    == 0.0
  )


def test_gradients_and_float32_match_unbucketed():
  ref, proc = _signal_pair(3000, 16000, seed=8)
  ref32, proc32 = ref.astype(np.float32), proc.astype(np.float32)
  h0, _ = haspi.haspi_v2(
    ref32, 16000, proc32, 16000, _LEVELS.astype(np.float32)
  )
  h1, _ = haspi.haspi_v2(
    ref32,
    16000,
    proc32,
    16000,
    _LEVELS.astype(np.float32),
    bucket_seconds=0.25,
  )
  assert h1.dtype == jnp.float32
  assert float(h1) == pytest.approx(float(h0), abs=1e-5)

  g0 = jax.grad(lambda p: haspi.haspi_v2(ref, 16000, p, 16000, _LEVELS)[0])(
    jnp.asarray(proc)
  )
  g1 = jax.grad(
    lambda p: haspi.haspi_v2(
      ref, 16000, p, 16000, _LEVELS, bucket_seconds=0.25
    )[0]
  )(jnp.asarray(proc))
  np.testing.assert_allclose(g1, g0, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
  "bad", (0, 0.0, -1.0, float("nan"), float("inf"), True, "1.0", 1e-9)
)
def test_invalid_bucket_seconds_raises(bad):
  ref, proc = _signal_pair(1000, 24000)
  with pytest.raises(ValueError, match="bucket_seconds"):
    haspi.haspi_v2(ref, 24000, proc, 24000, _LEVELS, bucket_seconds=bad)
