# Porting notes

Findings from reading `pyclarity==0.9.0` that the port has to act on. Kept
here rather than in comments, per the survey.

## Measured: the carrier does not drift enough to matter

`gammatone_bandwidth_demodulation` builds the demodulation carrier by iterated
2-D rotation rather than evaluating `cos(tpt·cf·n)`. Measured difference
between the two, in float64:

| | max abs difference |
|---|---|
| n=1200, cf=80 Hz | 8e-15 |
| n=1200, cf=8 kHz | 2.4e-13 |
| n=24000, cf=1 kHz | 1.2e-12 |
| n=96000, cf=1 kHz | 4.7e-12 |

Substituting one for the other changes the `ear_model` outputs by less than
the residual already present (2.2e-08 either way, identical to three digits).
**Resolved: use the direct evaluation.** It vectorises, and the rotation's
drift is four orders of magnitude below the error floor of the filter chain.

## Measured: the scan was never the bottleneck; the correlation was

The survey flagged ~256 sequential IIRs as the performance risk. Measured, it
is not: a `lax.scan` in direct form II transposed, `vmap`ed over 32 channels
of 24 000 samples, runs in **5 ms**. An associative-scan formulation agrees to
8e-16 and is no faster below ~24 000 samples. Not built.

The real cost was `envelope_align`'s cross-correlation. `jnp.convolve` lowers
to a general convolution that is quadratic when *both* operands are batched,
which is exactly how the filter bank calls it:

| | direct | via FFT |
|---|---|---|
| `vmap` correlate, 32 × 6000 | 9.34 s | 0.017 s |
| whole model, 32 channels | 19.1 s | 0.060 s |
| whole model, 4 channels | 0.07 s | 0.016 s |

The 4→32 channel scaling was ~16× per doubling, i.e. quadratic in the batch,
which is what gave it away. `filters.correlate_full` now multiplies spectra.
Accuracy is unchanged to three digits.

**Lesson worth keeping: profile the whole function, not its parts.** Every
component measured in milliseconds in isolation; the cost only appeared under
`vmap` with two batched operands.

## Achieved tolerance

Against the per-stage goldens, worst relative error over all six `ear_model`
outputs, both audiograms, both itypes: **2.2e-08**. Component stages are much
tighter -- centre frequencies and loss parameters are exact, the middle ear is
2e-14, NAL-R is 1e-16.

The residual enters at the gammatone recursion and is amplified by the
`20·log10` in `envelope_sl`: a 1.7e-08 relative difference in a linear
envelope becomes ~6e-07 absolute in dB, on values of order 40 dB. Channel 0
is worst because its filter is narrowest and its recursion the longest-lived.
This is scan-versus-`scipy.lfilter` accumulation order, not an algorithmic
difference: feeding both the same coefficients on the same input gives 5e-13,
and the error grows only through the cascade.

**float64 vs float32.** Using the factored 4th-order one-pole cascade
(`filters.cascaded_one_pole`) and closed-form group delay, `float32` runs
cleanly within ~6e-03 relative on HASPI and ~1.3e-02 relative on HASQI, while
`float64` (`jax_enable_x64`) achieves 2.8e-08 relative. `tests/conftest.py`
enables x64 so the golden suite checks exact reference parity, while
`tests/precision_test.py` tests both `float32` and `float64` execution.

## Differentiability and static-shape compilation

The envelope is `sqrt(re² + im²)`, and a silent input sample makes that
exactly zero, where `sqrt` has an infinite derivative. `_safe_sqrt` uses the
double-`where` trick, so the value is untouched and the derivative at zero is 0
instead. Verified against a finite difference to 1e-2 relative.

Both silence gates (`ear_model.input_align` and
`modulation.cepstral_correlation_coef`) run in static-shape JAX by shifting or
compacting the active samples (`valid_len` and `nkept`) to index `0` of the
fixed-length array and zero-padding the trailing tail, while masking and
normalizing downstream reductions by `valid_len` / `nkept`. Because all
intermediate filters are causal, the active prefix is mathematically identical
to cropping, without data-dependent tensor shapes or XLA recompilation.

## HASQI

Every back-end stage agrees with the reference to machine precision when fed
identical inputs: `env_smooth` 1.0e-15, `mel_cepstrum_correlation` exact,
`spectrum_diff` 2.7e-15, `bm_covary` 4.4e-16, `ave_covary2` 2.3e-16. So the
end-to-end error is inherited from `ear_model` and nothing else: worst
**2.8e-08** relative over the 38 scorable golden cases.

The two ceiling cases reproduce, which is the check that the port is faithful
rather than merely plausible:

| case | reference | port |
|---|---|---|
| `identical` (itype 0/1) | 0.778153809 | 0.778153811 |
| `identical_no_eq` (itype 2) | 0.924515205 | 0.924515206 |

Band-count scaling is linear -- 4 to 64 bands takes `bm_covary` from 3.9 ms to
22.5 ms -- so the quadratic-convolution trap that cost 19 s in `ear_model` is
not present here. `bm_covary` uses the same FFT `correlate_full`.

Note `mel_cepstrum_correlation`'s silence gate is mask-equivalent, as the
survey predicted: the selected columns are only summed over, never filtered
along time, so `jnp.where` reproduces the reference's column selection
exactly. `ave_covary2` is the same. This is why HASQI needed no
variable-length machinery while HASPI will.

## HASPI

Same discipline: every stage against the reference on identical inputs before
the end-to-end number. `env_filter` 7.4e-16, `cepstral_correlation_coef`
1.2e-15, `fir_modulation_filter` 3.9e-16, `modulation_cross_correlation`
1.1e-16, the neural-net ensemble **bit-exact**. End to end, worst **7.1e-09**
relative over the 37 scorable cases -- better than HASQI's 2.8e-08 because the
back end compresses rather than amplifies the `ear_model` residual.

All three silent-reference cases (`both_silent`, `reference_silent`,
`very_quiet`) return `0.0`.

### Static-shape cepstral silence gate

`cepstral_correlation_coef` keeps a data-dependent number of *time samples*
`nkept` and the modulation filterbank then convolves along that axis, so a
simple in-place zero mask is not equivalent -- dropping a sample changes the
sequence, not just a weighted sum. Instead, `cepstral_correlation_coef`
compacts the `nkept` active frames to `[:nkept]` via stable `jnp.argsort` and
zeros `[nkept:nsub]`. Because `fir_modulation_filter` performs a linear
convolution with symmetric FIR taps and slices `[half_len : half_len + nsub]`,
the first `nkept` output samples of convolving the zero-padded sequence are
identical to convolving the length-`nkept` sequence, and
`modulation_cross_correlation` masks `[nkept:nsub]` and normalizes by `nkept`.

### The dither-dominated cases

`processed_silent` and `audiogram_severe` land on the **noise-free** value,
0.003924696, matching to 0.0e+00. Both give exactly zero for all ten
modulation correlations, and 0.0039 is simply what the network outputs for a
zero feature vector -- it is the ensemble's floor, not a signal measurement.
The reference's 0.056 comes from correlating its own unseeded dither. Named in
`DITHER_DOMINATED`, asserted directly, not covered by a widened tolerance.

### Scaling

Checked the way the `jnp.convolve` trap was found. `modulation_cross_correlation`
is linear in bands: 0.1 ms at 4 bands to 1.7 ms at 64. No quadratic surprise --
it reduces over the sample axis rather than convolving two batched operands.

`fir_modulation_filter` looked alarming at a flat ~1.5 s regardless of size,
which is the signature of **tracing cost, not compute**: it loops over ten
modulation bands in Python. Under `jit` it is 6 ms warm at nsamp=400 and 18 ms
at 1600, with a 0.4 s one-off trace. Left as-is; the loop bound is a constant
ten and unrolling it in Python is what makes the band-dependent filter lengths
readable.

## Exact reformulations, verified

- `scipy.signal.group_delay((b, a), w=1)` equals `Σk·b/Σb − Σk·a/Σa`. Agreed to
  6e-10 on a representative filter.
- `bandwidth_adjust`'s three-way branch is exactly
  `bw_min + clip((db − 50)/50, 0, 1)·(bw_max − bw_min)`.
- NAL-R with a flat-zero audiogram degenerates to a pure delay. Asserted in
  `goldens_test.py`.

## Faithfulness traps

- `haspi_v2` calls `ear_model(shift=None)`. The `0.02` shift in the docstring
  is never applied. Keep it.
- `ear_model` rounds the sample rate to the nearest kHz before deciding whether
  to resample, so 22050 Hz is treated as 22 kHz.
- `input_align` trims to `|x| > 0.001·max(|x|)`, producing a **variable-length**
  output. Downstream shapes therefore depend on signal content.
- `group_delay_compensate` is called with `reference_bandwidth` for **both**
  signals, including the processed one. Looks like a bug; reproduce it.
- `convert_rms_to_sl` reassigns `threshold_high = 100` and `small = 1e-30`,
  shadowing its own parameters. The arguments are dead.
- `env_filter` transposes its input when `ncol > nrow`, so orientation is
  content-dependent for short signals.
- `melcor9` is dead code for both metrics — HASQI uses
  `mel_cepstrum_correlation`. Do not port it.

## Reference behaviour worth knowing

- Silent reference → `IndexError` from `input_align` indexing an empty
  threshold-crossing array. Recorded as a golden. The port rejects the same
  inputs with `ValueError`, so a caller can catch it as an unscoreable clip.
- Below the cepstral silence threshold, HASPI raises `ValueError` while HASQI
  returns 0.0. The two metrics genuinely differ here.
- Unseeded RNG draws: 4 per call, in a fixed order. HASPI's are
  `(32, n)`, `(32, n)`, `(nsub, 32)`, `(nsub, 32)`; HASQI's third and fourth are
  `(32, nseg)` and are multiplied by `add_noise=0.0`, so they are inert but
  still consume the stream.
