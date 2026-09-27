import jax.numpy as jnp
from flax import linen as nn

from .initialization import (
    content_initializer,
    decay_log_initializer,
    frequency_log_initializer,
    matrix_initializer,
    readout_initializer,
    write_log_gain_initializer,
)
from .recurrence import recurrence_terms
from .training.bptt import associative_states
from .training.rtrl import initialize_carry, stored_rtrl_step


def _parameters(module, input_size, modes, r_min, r_max, max_phase):
    frequency_log = module.param(
        "frequency_log", frequency_log_initializer(max_phase), (modes,)
    )
    decay_log = module.param(
        "decay_log", decay_log_initializer(r_min, r_max), (modes,)
    )
    content_real = module.param(
        "content_real", content_initializer(input_size), (input_size, modes)
    )
    content_imag = module.param(
        "content_imag", content_initializer(input_size), (input_size, modes)
    )
    phase_controller = module.param(
        "phase_controller", nn.initializers.zeros, (input_size + 1,)
    )
    retention_controller = module.param(
        "retention_controller", nn.initializers.zeros, (input_size + 1,)
    )
    write_log_gain = module.param(
        "write_log_gain",
        write_log_gain_initializer,
        decay_log,
        frequency_log,
    )
    write_phase_response = module.param(
        "write_phase_response", nn.initializers.zeros, (modes,)
    )
    write_retention_response = module.param(
        "write_retention_response", nn.initializers.zeros, (modes,)
    )
    return {
        "content_imag": content_imag,
        "content_real": content_real,
        "decay_log": decay_log,
        "frequency_log": frequency_log,
        "phase_controller": phase_controller,
        "retention_controller": retention_controller,
        "write_log_gain": write_log_gain,
        "write_phase_response": write_phase_response,
        "write_retention_response": write_retention_response,
    }


class SPARCRTRLCell(nn.Module):
    modes: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    content_activation: str = "tanh"
    output_activation: str = "relu"

    @nn.compact
    def __call__(self, carry, inputs, reset):
        parameters = _parameters(
            self, inputs.shape[-1], self.modes, self.r_min, self.r_max, self.max_phase
        )
        next_carry, state = stored_rtrl_step(
            parameters, carry, inputs, reset, self.content_activation
        )
        output = jnp.concatenate((state.real, state.imag), axis=-1)
        if self.output_activation == "relu":
            output = nn.relu(output)
        elif self.output_activation != "linear":
            raise ValueError(self.output_activation)
        return next_carry, output

    def initial_carry(self, batch_size, input_size, dtype=jnp.complex64):
        return initialize_carry(batch_size, input_size, self.modes, dtype)


class SPARCSequenceLayer(nn.Module):
    model_width: int
    modes: int
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = 6.28
    content_activation: str = "tanh"

    @nn.compact
    def __call__(self, inputs):
        if inputs.shape[-1] != self.model_width:
            raise ValueError((inputs.shape[-1], self.model_width))
        parameters = _parameters(
            self,
            self.model_width,
            self.modes,
            self.r_min,
            self.r_max,
            self.max_phase,
        )
        transition, write, _ = recurrence_terms(
            parameters, inputs, self.content_activation
        )
        states = associative_states(transition, write, axis=1)
        readout_real = self.param(
            "readout_real",
            readout_initializer(self.modes),
            (self.model_width, self.modes),
        )
        readout_imag = self.param(
            "readout_imag",
            readout_initializer(self.modes),
            (self.model_width, self.modes),
        )
        skip = self.param(
            "skip", matrix_initializer(1.0), (self.model_width,)
        )
        readout = readout_real + 1j * readout_imag
        return jnp.real(states @ readout.T) + skip * inputs
