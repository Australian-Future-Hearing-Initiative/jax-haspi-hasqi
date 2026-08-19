"""The goldens must travel with the package, not beside it."""

from jax_haspi_hasqi import goldens


def test_goldens_live_inside_the_package():
  """Resolving them beside the source tree broke every non-editable install."""
  data = goldens._GOLDENS
  assert data.is_dir()
  assert (data / "reference_values.npz").is_file()
  assert (data / "manifest.json").is_file()


def test_the_loader_reads_from_that_directory():
  assert goldens._GOLDENS.name == "_golden_data"
  assert goldens._GOLDENS.parent.name == "jax_haspi_hasqi"
  assert len(goldens.cases()) > 0
