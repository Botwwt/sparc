from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from ..recurrence import (
    CONTENT_PARAMETER_NAMES,
    CONTROLLER_PARAMETER_NAMES,
    MODAL_PARAMETER_NAMES,
    recurrence_terms,
    recurrent_step,
)


@dataclass(frozen=True)
class RTRLConfig:
    learning_rate: float
    gradient_clip: float | None = None
    adam_epsilon: float = 1.0e-5


def initialize_traces(batch_size, input_size, modes, dtype=jnp.complex64):
    traces = {
        name: jnp.zeros((batch_size, modes), dtype) for name in MODAL_PARAMETER_NAMES
    }
    traces.update(
        {
            name: jnp.zeros((batch_size, input_size, modes), dtype)
            for name in CONTENT_PARAMETER_NAMES
        }
    )
    traces.update(
        {
            name: jnp.zeros((batch_size, input_size + 1, modes), dtype)
            for name in CONTROLLER_PARAMETER_NAMES
        }
    )
    return traces


def initialize_carry(batch_size, input_size, modes, dtype=jnp.complex64):
    state = jnp.zeros((batch_size, modes), dtype)
    return state, initialize_traces(batch_size, input_size, modes, dtype)


def _reset_carry(carry, reset):
    state, traces = carry
    state = jnp.where(reset[:, None], 0, state)
    traces = jax.tree_util.tree_map(
        lambda value: jnp.where(
            reset.reshape((reset.shape[0],) + (1,) * (value.ndim - 1)), 0, value
        ),
        traces,
    )
    return state, traces


def _trace_step(parameters, carry, inputs, reset, content_activation):
    state, traces = _reset_carry(carry, reset)
    transition, _, details = recurrence_terms(parameters, inputs, content_activation)
    new_traces = {}
    for name in MODAL_PARAMETER_NAMES:
        _, direct = jax.jvp(
            lambda value: recurrent_step(
                {**parameters, name: value}, state, inputs, content_activation
            ),
            (parameters[name],),
            (jnp.ones_like(parameters[name]),),
        )
        new_traces[name] = transition * traces[name] + direct
    real_slope = (
        1.0 - jnp.tanh(details["raw_real"]) ** 2
        if content_activation == "tanh"
        else jnp.ones_like(details["raw_real"])
    )
    imag_slope = (
        1.0 - jnp.tanh(details["raw_imag"]) ** 2
        if content_activation == "tanh"
        else jnp.ones_like(details["raw_imag"])
    )
    real_direct = details["gain"] * real_slope
    imag_direct = 1j * details["gain"] * imag_slope
    new_traces["content_real"] = (
        transition[:, None, :] * traces["content_real"]
        + real_direct[:, None, :] * inputs[:, :, None]
    )
    new_traces["content_imag"] = (
        transition[:, None, :] * traces["content_imag"]
        + imag_direct[:, None, :] * inputs[:, :, None]
    )
    for name in CONTROLLER_PARAMETER_NAMES:
        direction = jnp.zeros_like(parameters[name]).at[-1].set(1.0)
        _, direct = jax.jvp(
            lambda value: recurrent_step(
                {**parameters, name: value}, state, inputs, content_activation
            ),
            (parameters[name],),
            (direction,),
        )
        new_traces[name] = (
            transition[:, None, :] * traces[name]
            + details["augmented"][:, :, None] * direct[:, None, :]
        )
    next_state = recurrent_step(parameters, state, inputs, content_activation)
    return (next_state, new_traces), next_state, (parameters, state, inputs, new_traces)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def stored_rtrl_step(parameters, carry, inputs, reset, content_activation):
    next_carry, state, _ = _trace_step(
        parameters, carry, inputs, reset, content_activation
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, next_carry), state


def _stored_rtrl_forward(parameters, carry, inputs, reset, content_activation):
    next_carry, state, residual = _trace_step(
        parameters, carry, inputs, reset, content_activation
    )
    output = jax.tree_util.tree_map(jax.lax.stop_gradient, next_carry), state
    return output, residual


def _stored_rtrl_backward(content_activation, residual, cotangent):
    parameters, state, inputs, traces = residual
    state_cotangent = cotangent[1]
    parameter_cotangents = {}
    for name, trace in traces.items():
        if trace.ndim == 2:
            value = jnp.real(state_cotangent * trace).sum(axis=0)
        else:
            value = jnp.real(state_cotangent[:, None, :] * trace).sum(axis=0)
            if name in CONTROLLER_PARAMETER_NAMES:
                value = value.sum(axis=-1)
        parameter_cotangents[name] = value
    _, pullback = jax.vjp(
        lambda value: recurrent_step(
            parameters, state, value, content_activation
        ),
        inputs,
    )
    input_cotangent = pullback(state_cotangent)[0]
    carry_cotangent = jax.tree_util.tree_map(jnp.zeros_like, (state, traces))
    return parameter_cotangents, carry_cotangent, input_cotangent, None


stored_rtrl_step.defvjp(
    _stored_rtrl_forward,
    _stored_rtrl_backward,
    symbolic_zeros=False,
)


def create_rtrl_state(model, parameters, config):
    optimizer = optax.adam(config.learning_rate, eps=config.adam_epsilon)
    if config.gradient_clip is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip), optimizer
        )
    return TrainState.create(apply_fn=model.apply, params=parameters, tx=optimizer)


def online_update(state, carry, inputs, reset, loss_function):
    def objective(parameters):
        next_carry, outputs = state.apply_fn(
            {"params": parameters}, carry, inputs, reset
        )
        return loss_function(outputs), next_carry

    (loss, next_carry), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(state.params)
    return state.apply_gradients(grads=gradients), next_carry, loss
