import jax


def test_jax_config():
    assert jax.config.read("jax_enable_x64") == True, "JAX must be in 64-bit mode"
