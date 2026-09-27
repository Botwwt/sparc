import jax
import jax.numpy as jnp


MODAL_PARAMETER_NAMES = (
    "decay_log",
    "frequency_log",
    "write_log_gain",
    "write_phase_response",
    "write_retention_response",
)

CONTENT_PARAMETER_NAMES = ("content_real", "content_imag")
CONTROLLER_PARAMETER_NAMES = ("phase_controller", "retention_controller")
RECURRENT_PARAMETER_NAMES = frozenset(
    MODAL_PARAMETER_NAMES + CONTENT_PARAMETER_NAMES + CONTROLLER_PARAMETER_NAMES
)


def control_signals(parameters, inputs):
    ones = jnp.ones(inputs.shape[:-1] + (1,), inputs.dtype)
    augmented = jnp.concatenate((inputs, ones), axis=-1)
    phase = jnp.tanh(augmented @ parameters["phase_controller"])
    retention = jnp.tanh(augmented @ parameters["retention_controller"])
    return phase, retention, augmented


def recurrence_terms(parameters, inputs, content_activation="tanh"):
    phase, retention, augmented = control_signals(parameters, inputs)
    decay = jnp.exp(parameters["decay_log"])
    frequency = jnp.exp(parameters["frequency_log"])
    effective_decay = decay * jnp.exp(jnp.log(16.0) * retention[..., None])
    radius = jnp.exp(-effective_decay)
    phase_offset = (1.0 - radius) * (jnp.pi / 2.0) * phase[..., None]
    transition = jnp.exp(-effective_decay + 1j * (frequency + phase_offset))
    ratio = jnp.sqrt(
        -jnp.expm1(-2.0 * effective_decay) / -jnp.expm1(-2.0 * decay)
    )
    raw_real = inputs @ parameters["content_real"]
    raw_imag = inputs @ parameters["content_imag"]
    if content_activation == "tanh":
        content_real = jnp.tanh(raw_real)
        content_imag = jnp.tanh(raw_imag)
    elif content_activation == "linear":
        content_real = raw_real
        content_imag = raw_imag
    else:
        raise ValueError(content_activation)
    gate = 2.0 * jax.nn.sigmoid(
        phase[..., None] * parameters["write_phase_response"]
        + retention[..., None] * parameters["write_retention_response"]
    )
    gain = jnp.exp(parameters["write_log_gain"]) * ratio * gate
    write = gain * (content_real + 1j * content_imag)
    details = {
        "augmented": augmented,
        "gain": gain,
        "phase": phase,
        "raw_imag": raw_imag,
        "raw_real": raw_real,
        "retention": retention,
    }
    return transition, write, details


def recurrent_step(parameters, state, inputs, content_activation="tanh"):
    transition, write, _ = recurrence_terms(parameters, inputs, content_activation)
    return transition * state + write
