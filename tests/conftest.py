"""Test configuration: the port matches the reference only in double precision."""

import jax

jax.config.update("jax_enable_x64", True)
