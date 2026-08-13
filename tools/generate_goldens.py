#!/usr/bin/env python3
"""Regenerate goldens/ by running pyclarity 0.9.0. NOT part of the package.

Requires pyclarity, which jax_haspi_hasqi does not depend on. Build a throwaway
environment:

    uv venv --python 3.12 --managed-python /tmp/clarity_ref
    uv pip install --python /tmp/clarity_ref/bin/python pyclarity==0.9.0
    /tmp/clarity_ref/bin/python tools/generate_goldens.py

Takes a few minutes. Rerun only when the pinned reference version changes.
"""

import argparse
import contextlib
import json
import pathlib
import sys

import numpy as np

# Fails loudly rather than being an optional import: this script is useless
# without the reference.
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import eb
from clarity.evaluator.haspi import haspi as haspi_ref
from clarity.evaluator.hasqi import hasqi as hasqi_ref
from clarity.utils.audiogram import Audiogram

PYCLARITY_VERSION = "0.9.0"
NOISE_SEED = 20260811
SIGNAL_SEED = 12345

AUDIOGRAM_FREQUENCIES = np.array([250, 500, 1000, 2000, 4000, 6000])

# Named audiograms at the 6 HASPI/HASQI frequencies. "clinical" is a typical
# mild-to-moderate sensorineural loss; note that an audiogram measured at
# 8000 Hz has to be resampled onto 6000 before it reaches these metrics.
AUDIOGRAMS = {
  "flat_zero": np.zeros(6),
  "mild": np.array([10.0, 15.0, 19.0, 25.0, 31.0, 35.0]),
  "clinical": np.array([21.0, 26.0, 30.0, 32.0, 33.0, 33.0]),
  "moderate": np.array([20.0, 20.0, 25.0, 35.0, 45.0, 55.0]),
  "severe": np.array([40.0, 45.0, 55.0, 65.0, 75.0, 80.0]),
  "sloping_steep": np.array([0.0, 5.0, 10.0, 40.0, 80.0, 90.0]),
}


def _tone(duration, sample_rate, freq=220.0, modulation=3.0):
  """Amplitude-modulated tone; a stand-in for voiced speech."""
  t = np.arange(int(duration * sample_rate)) / sample_rate
  carrier = np.sin(2 * np.pi * freq * t)
  return 0.1 * carrier * (1 + 0.5 * np.sin(2 * np.pi * modulation * t))


def _speechlike(duration, sample_rate, seed):
  """Broadband noise under a syllabic envelope, band-limited to 4 kHz."""
  rng = np.random.default_rng(seed)
  n = int(duration * sample_rate)
  t = np.arange(n) / sample_rate
  noise = rng.standard_normal(n)
  spectrum = np.fft.rfft(noise)
  spectrum[np.fft.rfftfreq(n, 1 / sample_rate) > 4000] = 0
  band = np.fft.irfft(spectrum, n)
  band /= np.max(np.abs(band))
  envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) ** 2
  return 0.1 * band * envelope


def _add_noise_at_snr(signal, snr_db, seed):
  rng = np.random.default_rng(seed)
  noise = rng.standard_normal(len(signal))
  scale = np.sqrt(np.mean(signal**2) / np.mean(noise**2)) * 10 ** (-snr_db / 20)
  return signal + scale * noise


def _delay(signal, samples):
  return np.concatenate((np.zeros(samples), signal[: len(signal) - samples]))


@contextlib.contextmanager
def _captured_noise(seed):
  """Seed the reference's RNG and record every draw it makes.

  The reference calls np.random.standard_normal unseeded in three places, so
  its output is not reproducible as shipped. Seeding makes it so; recording
  the draws lets the port be fed the identical noise.
  """
  draws = []
  original = np.random.standard_normal

  def spy(*args, **kwargs):
    value = original(*args, **kwargs)
    draws.append(np.asarray(value))
    return value

  np.random.seed(seed)
  np.random.standard_normal = spy
  try:
    yield draws
  finally:
    np.random.standard_normal = original


@contextlib.contextmanager
def _zero_noise():
  """Force every RNG draw to zero, making the reference deterministic."""
  original = np.random.standard_normal
  # np.zeros takes the same shape argument the reference passes.
  np.random.standard_normal = np.zeros

  try:
    yield
  finally:
    np.random.standard_normal = original


def _audiogram(levels):
  return Audiogram(levels=np.asarray(levels), frequencies=AUDIOGRAM_FREQUENCIES)


def _run_haspi(case):
  return haspi_ref.haspi_v2(
    case["reference"],
    case["reference_sample_rate"],
    case["processed"],
    case["processed_sample_rate"],
    _audiogram(case["levels"]),
    level1=case["level1"],
    itype=case["itype"],
  )


def _run_hasqi(case):
  return hasqi_ref.hasqi_v2(
    case["reference"],
    case["reference_sample_rate"],
    case["processed"],
    case["processed_sample_rate"],
    _audiogram(case["levels"]),
    equalisation=case["equalisation"],
    level1=case["level1"],
  )


def _fingerprint(array):
  """Enough of an array to confirm a regenerated copy is the same one."""
  flat = np.asarray(array, dtype=np.float64).ravel()
  return {
    "shape": list(np.shape(array)),
    "mean": repr(float(flat.mean())),
    "std": repr(float(flat.std())),
    "first": repr(float(flat[0])),
    "last": repr(float(flat[-1])),
  }


def _evaluate(runner, case, seed):
  """Run one metric twice: noise-free, then seeded with recorded draws."""
  result = {}
  try:
    with _zero_noise():
      result["noise_free"] = runner(case)
  # A raise is a valid golden, so every exception type is recorded.
  except Exception as error:
    result["noise_free"] = None
    result["raises"] = type(error).__name__
    result["message"] = str(error)
    return result

  with _captured_noise(seed) as draws:
    result["seeded"] = runner(case)
  result["noise_shapes"] = [list(d.shape) for d in draws]
  result["noise_fingerprints"] = [_fingerprint(d) for d in draws]
  return result


def _build_cases():
  """Every case the goldens cover. Signals are stored, not recomputed."""
  # 0.25 s is the shortest duration at which no stage degenerates (see the
  # duration_* cases), which keeps the checked-in goldens small.
  sr = 24000
  clean = _tone(0.25, sr)
  speech = _speechlike(0.25, sr, SIGNAL_SEED)

  cases = []

  def add(name, reference, processed, comment, **overrides):
    case = {
      "name": name,
      "comment": comment,
      "reference": np.asarray(reference, dtype=np.float64),
      "processed": np.asarray(processed, dtype=np.float64),
      "reference_sample_rate": sr,
      "processed_sample_rate": sr,
      "levels": AUDIOGRAMS["clinical"],
      "audiogram": "clinical",
      "level1": 65.0,
      "itype": 0,
      "equalisation": 1,
    }
    case.update(overrides)
    cases.append(case)

  # Ordinary pairs at a range of degradations.
  for snr in (30, 20, 10, 0):
    add(
      f"tone_snr{snr}",
      clean,
      _add_noise_at_snr(clean, snr, SIGNAL_SEED + snr),
      f"AM tone degraded to {snr} dB SNR",
    )
  add(
    "speech_snr20",
    speech,
    _add_noise_at_snr(speech, 20, SIGNAL_SEED),
    "Speech-like broadband signal at 20 dB SNR",
  )
  add(
    "speech_snr5",
    speech,
    _add_noise_at_snr(speech, 5, SIGNAL_SEED + 1),
    "Speech-like broadband signal at 5 dB SNR",
  )

  # Degenerate and near-degenerate pairs.
  add("identical", clean, clean, "Processed is the reference, bit for bit")
  add(
    "identical_no_eq",
    clean,
    clean,
    "Identical, itype=2 so no NAL-R is added to the reference: the true ceiling",
    itype=2,
    equalisation=2,
  )

  add(
    "near_identical",
    clean,
    clean + 1e-6 * np.random.default_rng(7).standard_normal(len(clean)),
    "Difference at the 1e-6 level; exercises the small-value guards",
  )
  add("scaled_half", clean, 0.5 * clean, "Pure level change, no distortion")
  add("processed_silent", clean, np.zeros_like(clean), "Processed is all zeros")
  add(
    "both_silent",
    np.zeros_like(clean),
    np.zeros_like(clean),
    "Both silent; the reference raises IndexError from input_align",
  )
  add(
    "reference_silent",
    np.zeros_like(clean),
    clean,
    "Reference silent; the reference raises IndexError from input_align",
  )
  add(
    "very_quiet",
    1e-6 * clean,
    1e-6 * clean,
    "Below the cepstral silence threshold; HASPI raises, HASQI returns 0",
  )
  add(
    "clipped",
    clean,
    np.clip(clean, -0.05, 0.05),
    "Hard clipping: distortion without added noise",
  )

  # Alignment: input_align and envelope_align must find these shifts.
  for shift_ms in (2, 10):
    samples = int(shift_ms * sr / 1000)
    add(
      f"delayed_{shift_ms}ms",
      clean,
      _delay(_add_noise_at_snr(clean, 20, SIGNAL_SEED), samples),
      f"Processed delayed by {shift_ms} ms",
    )

  # Audiograms, including flat zero (no loss, and NAL-R becomes a pure delay).
  for name, levels in AUDIOGRAMS.items():
    add(
      f"audiogram_{name}",
      clean,
      _add_noise_at_snr(clean, 15, SIGNAL_SEED + 2),
      f"Audiogram {name}",
      levels=levels,
      audiogram=name,
    )

  # itype / equalisation branches. itype selects the reference ear: 0 gives it
  # normal hearing, 1 adds NAL-R to it, 2 assumes NAL-R is already applied.
  for itype in (0, 1, 2):
    add(
      f"itype{itype}",
      clean,
      _add_noise_at_snr(clean, 15, SIGNAL_SEED + 3),
      f"ear_model itype={itype}",
      itype=itype,
      equalisation=max(itype, 1),
    )
  add(
    "itype1_flat_zero",
    clean,
    _add_noise_at_snr(clean, 15, SIGNAL_SEED + 4),
    "NAL-R path with no loss: the filter degenerates to a pure delay",
    itype=1,
    equalisation=1,
    levels=AUDIOGRAMS["flat_zero"],
    audiogram="flat_zero",
  )

  # Sample rates. 24 kHz is a no-op, below it upsamples, above it downsamples
  # through a Chebyshev anti-alias filter.
  for rate in (16000, 22050, 32000, 44100):
    signal = _tone(0.25, rate)
    add(
      f"rate_{rate}",
      signal,
      _add_noise_at_snr(signal, 20, SIGNAL_SEED + 5),
      f"Both signals at {rate} Hz",
      reference_sample_rate=rate,
      processed_sample_rate=rate,
    )
  add(
    "rate_mismatch_24k_16k",
    clean,
    _add_noise_at_snr(_tone(0.25, 16000), 20, SIGNAL_SEED + 6),
    "Reference 24 kHz, processed 16 kHz",
    processed_sample_rate=16000,
  )

  # Durations. Below about 40 ms the covariance segmentation degenerates.
  for duration in (0.05, 0.1, 0.25, 1.0):
    signal = _tone(duration, sr)
    add(
      f"duration_{duration}s",
      signal,
      _add_noise_at_snr(signal, 20, SIGNAL_SEED + 7),
      f"{duration} s of audio",
    )

  # Presentation level. 65 is the reference's own default; the higher values
  # cover callers that calibrate to a louder full-scale level.
  for level1 in (65.0, 100.0, 107.0):
    add(
      f"level1_{int(level1)}",
      clean,
      _add_noise_at_snr(clean, 20, SIGNAL_SEED + 8),
      f"level1={level1} dB SPL for RMS 1",
      level1=level1,
    )

  # Unequal lengths: the reference truncates to the shorter.
  add(
    "unequal_length",
    clean,
    _add_noise_at_snr(clean, 20, SIGNAL_SEED + 9)[: len(clean) // 2],
    "Processed is half as long; the reference truncates",
  )
  return cases


def _stage_goldens(arrays, manifest):
  """Per-stage values for a short signal, to localise a port mismatch."""
  sr = 24000
  signal = _tone(0.05, sr)
  degraded = _add_noise_at_snr(signal, 20, SIGNAL_SEED)

  arrays["stage__center_frequency_32"] = eb.center_frequency(32)
  arrays["stage__center_frequency_32_shift02"] = eb.center_frequency(32, 0.02)
  arrays["stage__middle_ear"] = eb.middle_ear(signal, sr)
  arrays["stage__reference_signal"] = signal
  arrays["stage__processed_signal"] = degraded

  stages = {}
  for name, levels in (
    ("flat_zero", AUDIOGRAMS["flat_zero"]),
    ("clinical", AUDIOGRAMS["clinical"]),
  ):
    for key, value in zip(
      ("attn_ohc", "bandwidth", "low_knee", "compression_ratio", "attn_ihc"),
      eb.loss_parameters(np.asarray(levels), eb.center_frequency(32)),
    ):
      arrays[f"stage__loss_parameters__{name}__{key}"] = value

    enhancer = NALR(140, 24000.0)
    fir, delay = enhancer.build(_audiogram(levels))
    arrays[f"stage__nalr_fir__{name}"] = fir
    arrays[f"stage__nalr_delay__{name}"] = delay

    for itype in (0, 1):
      with _zero_noise():
        outputs = eb.ear_model(
          signal, sr, degraded, sr, np.asarray(levels), itype, 65.0, shift=None
        )
      tag = f"{name}__itype{itype}"
      for key, value in zip(
        (
          "reference_db",
          "reference_bm",
          "processed_db",
          "processed_bm",
          "reference_sl",
          "processed_sl",
        ),
        outputs[:6],
      ):
        arrays[f"stage__ear_model__{tag}__{key}"] = np.asarray(value)
      stages[tag] = {
        "sample_rate": float(outputs[6]),
        "envelope_shape": list(np.shape(outputs[0])),
      }

  manifest["stages"] = {
    "signal": "0.05 s AM tone, 24 kHz, degraded copy at 20 dB SNR",
    "noise": "all RNG draws forced to zero",
    "ear_model": stages,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "jax_haspi_hasqi"
    / "_golden_data",
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)

  arrays = {}
  manifest = {
    "pyclarity_version": PYCLARITY_VERSION,
    "numpy_version": np.__version__,
    "python_version": sys.version.split()[0],
    "noise_seed": NOISE_SEED,
    "signal_seed": SIGNAL_SEED,
    "audiogram_frequencies": AUDIOGRAM_FREQUENCIES.tolist(),
    "audiograms": {k: v.tolist() for k, v in AUDIOGRAMS.items()},
    "notes": [
      "Values are pyclarity 0.9.0 output, recorded at full float64 precision.",
      "noise_free: every np.random.standard_normal draw replaced by zeros.",
      "seeded: np.random.seed(noise_seed) before the call; noise_shapes lists"
      " the draws in order, so a port can reproduce them with numpy's legacy"
      " RandomState and be fed the identical noise.",
      "raises: the reference raises for this input. That is the golden.",
      "HASPI is called with shift=None, preserving the documented v2 defect.",
    ],
    "cases": [],
  }

  for index, case in enumerate(_build_cases()):
    print(f"[{index:2d}] {case['name']}", flush=True)
    arrays[f"case{index:02d}__reference"] = case["reference"]
    arrays[f"case{index:02d}__processed"] = case["processed"]
    arrays[f"case{index:02d}__levels"] = np.asarray(case["levels"])

    entry = {
      "index": index,
      "name": case["name"],
      "comment": case["comment"],
      "reference_sample_rate": case["reference_sample_rate"],
      "processed_sample_rate": case["processed_sample_rate"],
      "reference_samples": int(len(case["reference"])),
      "processed_samples": int(len(case["processed"])),
      "audiogram": case["audiogram"],
      "level1": case["level1"],
      "itype": case["itype"],
      "equalisation": case["equalisation"],
    }

    haspi_result = _evaluate(_run_haspi, case, NOISE_SEED)
    entry["haspi"] = {}
    for variant in ("noise_free", "seeded"):
      value = haspi_result.get(variant)
      if value is None:
        continue
      score, raw = value
      entry["haspi"][variant] = repr(float(score))
      arrays[f"case{index:02d}__haspi__{variant}__score"] = np.float64(score)
      arrays[f"case{index:02d}__haspi__{variant}__raw"] = np.asarray(raw)
    for key in ("raises", "message", "noise_shapes", "noise_fingerprints"):
      if key in haspi_result:
        entry["haspi"][key] = haspi_result[key]

    hasqi_result = _evaluate(_run_hasqi, case, NOISE_SEED)
    entry["hasqi"] = {}
    for variant in ("noise_free", "seeded"):
      value = hasqi_result.get(variant)
      if value is None:
        continue
      combined, nonlinear, linear, raw = value
      entry["hasqi"][variant] = {
        "combined": repr(float(combined)),
        "nonlinear": repr(float(nonlinear)),
        "linear": repr(float(linear)),
      }
      prefix = f"case{index:02d}__hasqi__{variant}"
      arrays[f"{prefix}__combined"] = np.float64(combined)
      arrays[f"{prefix}__nonlinear"] = np.float64(nonlinear)
      arrays[f"{prefix}__linear"] = np.float64(linear)
      arrays[f"{prefix}__raw"] = np.asarray(raw, dtype=np.float64)
    for key in ("raises", "message", "noise_shapes", "noise_fingerprints"):
      if key in hasqi_result:
        entry["hasqi"][key] = hasqi_result[key]

    manifest["cases"].append(entry)

  print("stage goldens", flush=True)
  _stage_goldens(arrays, manifest)

  np.savez_compressed(args.output / "reference_values.npz", **arrays)
  (args.output / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
  )
  print(f"wrote {len(arrays)} arrays and {len(manifest['cases'])} cases")


if __name__ == "__main__":
  main()
