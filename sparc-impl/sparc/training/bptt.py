from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.training.train_state import TrainState

from ..recurrence import RECURRENT_PARAMETER_NAMES


@dataclass(frozen=True)
class BPTTConfig:
    learning_rate: float
    recurrent_learning_rate_scale: float = 0.25
    minimum_learning_rate: float = 1.0e-6
    warmup_steps: int = 0
    total_steps: int = 1
    weight_decay: float = 0.0
    gradient_clip: float | None = None


def affine_composition(left, right):
    left_transition, left_write = left
    right_transition, right_write = right
    return (
        right_transition * left_transition,
        right_transition * left_write + right_write,
    )


def associative_states(transitions, writes, axis=1):
    return jax.lax.associative_scan(
        affine_composition, (transitions, writes), axis=axis
    )[1]


def _schedule(config, scale):
    peak = config.learning_rate * scale
    floor = config.minimum_learning_rate * scale
    if config.warmup_steps == 0:
        return optax.cosine_decay_schedule(
            peak, max(config.total_steps, 1), alpha=floor / peak
        )
    warmup = optax.linear_schedule(floor, peak, config.warmup_steps)
    decay = optax.cosine_decay_schedule(
        peak,
        max(config.total_steps - config.warmup_steps, 1),
        alpha=floor / peak,
    )
    return optax.join_schedules([warmup, decay], [config.warmup_steps])


def parameter_labels(parameters):
    return traverse_util.path_aware_map(
        lambda path, value: (
            "recurrent" if path[-1] in RECURRENT_PARAMETER_NAMES else "regular"
        ),
        parameters,
    )


def create_bptt_state(model, parameters, config):
    transforms = {
        "recurrent": optax.adam(
            _schedule(config, config.recurrent_learning_rate_scale)
        ),
        "regular": optax.adamw(
            _schedule(config, 1.0), weight_decay=config.weight_decay
        ),
    }
    optimizer = optax.multi_transform(transforms, parameter_labels(parameters))
    if config.gradient_clip is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip), optimizer
        )
    return TrainState.create(apply_fn=model.apply, params=parameters, tx=optimizer)


def classification_update(state, inputs, labels, dropout_key=None):
    def objective(parameters):
        variables = {"params": parameters}
        randoms = None if dropout_key is None else {"dropout": dropout_key}
        log_probabilities = state.apply_fn(variables, inputs, rngs=randoms)
        loss = -jnp.take_along_axis(
            log_probabilities, labels[:, None], axis=-1
        ).mean()
        return loss, log_probabilities

    (loss, log_probabilities), gradients = jax.value_and_grad(
        objective, has_aux=True
    )(state.params)
    return state.apply_gradients(grads=gradients), loss, log_probabilities
