"""Parity with the seeded reference, using its recorded noise draws."""

import numpy as np
import pytest

from jax_haspi_hasqi import goldens
from jax_haspi_hasqi import haspi
from jax_haspi_hasqi import hasqi

TOLERANCE = 1e-8

HASPI_CASES = ("tone_snr20", "audiogram_clinical", "processed_silent")
HASQI_CASES = ("tone_snr20", "audiogram_clinical", "scaled_half")


def recorded_draws(case, metric):
  """Reproduce the draws the reference made, in the order it made them."""
  np.random.seed(goldens.manifest()["noise_seed"])
  shapes = getattr(case, metric)["noise_shapes"]
  return [np.random.standard_normal(tuple(shape)) for shape in shapes]


@pytest.mark.parametrize("name", HASPI_CASES)
def test_haspi_matches_the_seeded_reference(name):
  case = goldens.case(name)
  bm_reference, bm_processed, cepstral_reference, cepstral_processed = (
    recorded_draws(case, "haspi")
  )
  score, _ = haspi.haspi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    level1=case.level1,
    itype=case.itype,
    reference_noise=bm_reference,
    processed_noise=bm_processed,
    cepstral_noise=(cepstral_reference, cepstral_processed),
  )
  assert float(score) == pytest.approx(
    case.score("haspi", "seeded"), rel=TOLERANCE
  )


@pytest.mark.parametrize("name", HASQI_CASES)
def test_hasqi_matches_the_seeded_reference(name):
  """HASQI's third and fourth draws are multiplied by zero, so are unused."""
  case = goldens.case(name)
  bm_reference, bm_processed = recorded_draws(case, "hasqi")[:2]
  combined, _, _, _ = hasqi.hasqi_v2(
    case.reference,
    case.reference_sample_rate,
    case.processed,
    case.processed_sample_rate,
    case.levels,
    equalisation=case.equalisation,
    level1=case.level1,
    reference_noise=bm_reference,
    processed_noise=bm_processed,
  )
  assert float(combined) == pytest.approx(
    case.score("hasqi", "seeded"), rel=TOLERANCE
  )
