import jax
import jax.numpy as jnp


def matrix_initializer(normalization):
    def initialize(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) / normalization

    return initialize


def decay_log_initializer(r_min, r_max):
    def initialize(key, shape, dtype=jnp.float32):
        values = jax.random.uniform(key, shape, dtype)
        radii_squared = values * (r_max**2 - r_min**2) + r_min**2
        return jnp.log(-0.5 * jnp.log(radii_squared))

    return initialize


def frequency_log_initializer(max_phase):
    def initialize(key, shape, dtype=jnp.float32):
        values = jax.random.uniform(key, shape, dtype)
        return jnp.log(max_phase * values)

    return initialize


def write_log_gain_initializer(key, decay_log, frequency_log):
    transition = jnp.exp(-jnp.exp(decay_log) + 1j * jnp.exp(frequency_log))
    return jnp.log(jnp.sqrt(1.0 - jnp.abs(transition) ** 2))


def content_initializer(input_size):
    return matrix_initializer(jnp.sqrt(2.0 * input_size))


def readout_initializer(modes):
    return matrix_initializer(jnp.sqrt(modes))
