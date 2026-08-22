# jax-haspi-hasqi

HASPI v2 and HASQI v2 in JAX.

Two hearing-aid perceptual metrics, reimplemented on JAX so they run fast
without dragging in torch and the CUDA stack. The reference implementation
([pyclarity](https://github.com/claritychallenge/clarity)) costs 108 packages
and 5.1 GB for two scalar numbers; this costs 15 and 444 MB, and is 4.2× faster
once warm.

> [!IMPORTANT]
> **This is a faithful port, not an improved one.** Its contract is to
> reproduce `pyclarity==0.9.0` numerically. Where the reference behaves oddly,
> so does this. Do not "fix" things here — see *Preserved defects* below.

## Using it

```python
from jax_haspi_hasqi import haspi, hasqi

score, _ = haspi.haspi_v2(reference, 24000, processed, 24000, levels)
quality, *_ = hasqi.hasqi_v2(reference, 24000, processed, 24000, levels)
```

No JAX configuration required. float32 inputs are fine, and so is a float32
process.

`levels` is six audiometric thresholds in dB HL at 250, 500, 1000, 2000, 4000
and 6000 Hz. **These frequencies are the contract.** pyclarity's entry points
resample an `Audiogram` onto them; this package takes the six values directly
and does not, so a caller holding thresholds at other frequencies must
interpolate first — on a log-frequency axis, as
`clarity.utils.audiogram.Audiogram.resample` does. A 6 kHz-vs-8 kHz mismatch is
silent and can reach 10 dB in the top band.

> [!NOTE]
> **These metrics compute in float64 internally, whatever you have
> configured.** Both entry points enable x64 for the duration of the call and
> restore your setting afterwards, and both cast their input signals up, since
> enabling x64 does not promote an array that is already float32.
>
> This is not fussiness. The gammatone filter bank is a 4th-order IIR whose
> poles sit against the unit circle — 0.9913 at the 80 Hz channel — which needs
> about nine significant digits in the denominator to place the filter
> correctly, and float32 carries seven. Computed in float32 the filter bank
> output is wrong by a relative 5.6e+03, HASPI then trips its own silence gate
> and HASQI returns NaN. Deriving the coefficients in float64 does not rescue
> it either: a float32 recursion still drifts by 6e-03.
>
> Passing float32 signals costs about **4e-08** on the score, which is what
> float32 sample data is worth, and three orders below the reference's own
> 6e-03 run-to-run spread. The promotion does not disturb the jit cache.

### Performance

Both metrics are `jit`-compiled internally, so the first call for a given
signal shape pays compilation and later calls with that shape do not. Measured
on 12 mixed-length pairs, scoring both metrics, against pyclarity on the same
signals:

| | s/pair |
|---|---|
| first pass (compiling) | 4.2 |
| **steady state** | **0.39** |
| pyclarity | 1.66 |

**4.2× faster than the reference** once warm. The shapes that drive
recompilation — the aligned length, the HASQI segment count, the HASPI
cepstral-gate count — all depend on the *reference* signal, not the processed
one, so a fixed validation set compiles once and stays warm as the model under
test changes. Across epochs holding the references fixed, all three were
invariant.

Set `jax_compilation_cache_dir` to carry compilation across processes.

> [!NOTE]
> Not differentiable, and not `jit`-able from outside. `input_align` and the
> cepstral gate crop to data-dependent shapes in numpy, which a tracer cannot
> pass through; `jax.grad` and an enclosing `jax.jit` both raise
> `TracerArrayConversionError`. These are reporting metrics, not losses. The
> alignment stages that resist differentiation are exactly the ones a faithful
> port has to keep.

## What it is faithful to

`pyclarity==0.9.0`, specifically `clarity.evaluator.haspi`,
`clarity.evaluator.hasqi` and the `NALR` enhancer they share. HAAQI is out of
scope.

The contract is `src/jax_haspi_hasqi/_golden_data/`: recorded outputs of that
release across 40 cases, shipped inside the package so an installed wheel can
verify itself. There is no shared upstream test suite to inherit — pyclarity
has no JAX backend — so the goldens are the entire specification.

## Preserved defects

Deliberate. Each is something a reasonable person would otherwise correct.

| | Behaviour |
|---|---|
| `shift=None` | HASPI v2 documents a `0.02` basal shift of the basilar membrane that is never applied. Upstream [discussed it](https://github.com/claritychallenge/clarity/issues/105) and kept the bug, because every published HASPI number was computed with it. |
| Unseeded noise | The reference draws Gaussian noise from the global `np.random` in three places, so repeated calls disagree — HASPI by up to 3e-3. |
| Raises on silence | A silent reference has nothing above the threshold `input_align` prunes to, so the pair cannot be aligned and no score exists. Preserved: it still raises, and still on exactly the inputs upstream fails on. The one divergence in this table — upstream reaches this by indexing an empty array and raising `IndexError`, which reads as a bug in the port rather than an unscoreable clip, so this raises `ValueError` naming the cause. No score changes; the goldens still record upstream's `IndexError`. |
| `nbands` scaling | `spectrum_diff` multiplies by the channel count, and `ave_covary2` re-derives centre frequencies from it. Both assume the 32-channel bank. |
| Short-input scores | Below about 0.5 s of surviving audio HASPI inflates towards 1 and stops being usable: unrelated noise scores about `0.06` at 1 s but about `0.98` at 0.062 s, as medians over seeds at the default `level1`. The cause is the cross-covariance over too few envelope frames, not silence as such — silence only matters because `input_align` crops it and can leave a short record. pyclarity does the same, so this is preserved, not introduced. Guard the duration in the caller. |

## Noise

Because the reference is stochastic, the goldens record two variants per case:

- **`noise_free`** — every RNG draw replaced by zeros. Deterministic, and the
  intended default here.
- **`seeded`** — `np.random.seed(20260811)` before the call, with the shape and
  a fingerprint of each draw recorded in order, so the port can be handed the
  identical noise for an exact comparison.

For HASPI the noise-free value is interchangeable with a pyclarity run: it sits
inside the reference's own run-to-run spread for every golden case. **For HASQI
it is not.** Over 12 pyclarity runs per case it sits systematically *below* the
stochastic mean, by up to 1.4e-02 on a score of 0.25 — 5 % relative, and ten
times that case's peak-to-peak spread. Only HASQI consumes the basilar-membrane
motion the reference's threshold noise is injected into; HASPI sees only the
envelopes. The bias is not decorrelation but tile revival: where the processed
BM motion is exactly zero, `bm_covary`'s validity guard scores the tile 0 while
still counting it in the average, and any noise at all lifts it off zero.

| case | metric | noise-free | 12-run mean | peak-to-peak | bias |
|---|---|---|---|---|---|
| `speech_snr5` | HASQI | 0.2545012 | 0.2683184 | 1.3e-03 | **−1.4e-02** |
| `speech_snr20` | HASQI | 0.2558208 | 0.2675992 | 1.4e-03 | **−1.2e-02** |
| `speech_snr5` | HASPI | 0.9989436 | 0.9989102 | 2.2e-04 | +3.3e-05 |

So report a HASQI figure from this package as the deterministic noise-free
variant, not as a plain pyclarity HASQI v2 number. Use the `seeded` goldens if
you need exact parity.

## Regenerating the goldens

Rarely needed: only when the pinned reference version changes. Requires
pyclarity, which this package does not depend on.

```
uv venv --python 3.12 --managed-python /tmp/clarity_ref
uv pip install --python /tmp/clarity_ref/bin/python pyclarity==0.9.0
/tmp/clarity_ref/bin/python tools/generate_goldens.py
```

Writes `reference_values.npz` and `manifest.json` into
`src/jax_haspi_hasqi/_golden_data/`. Inputs are stored alongside the outputs,
so the port never regenerates a signal and never needs pyclarity.

`tools/` is not part of the package.

## Development

```
uv sync --all-groups
uv run pytest -q            # ~70 s, 195 tests
uv run ruff format .
uv run ruff check .
```

CI runs exactly those on 3.11 and 3.13, and separately builds a wheel,
installs it into a clean environment and scores a golden case from it. That
last job exists because a source checkout finds the goldens whether or not
they are packaged, so the suite passes either way; only an install catches a
wheel that ships without its own specification.

There is no committed lockfile. Dependencies resolve fresh on every run, so a
numpy or scipy release that moves the port off its goldens fails CI rather
than waiting to be discovered — which is the point of having goldens.

## Licence

MIT. This is a derivative work of pyclarity, also MIT — see [NOTICE](NOTICE)
for attribution and [licenses/pyclarity/](licenses/pyclarity/) for the
upstream licence. The algorithms are due to James M. Kates; citations are in
`NOTICE`.

`nalr.py` implements **NAL-R**, the linear prescription rule published in Byrne
& Dillon (1986), reimplemented from the MIT-licensed pyclarity source. It is
not NAL-NL1 or NAL-NL2, which are separate proprietary procedures licensed by
National Acoustic Laboratories and are not present in this repository.
