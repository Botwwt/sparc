from pathlib import Path
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparc import RTRLConfig, SPARCActorCritic, create_rtrl_state
from sparc.training.rtrl import online_update

def main():
    batch_size = 4
    observation_size = 12
    action_size = 5
    model = SPARCActorCritic(
        action_size=action_size,
        modes=64,
        encoder_width=64,
        policy_width=64,
        value_width=64,
        continuous_actions=False,
    )
    carry = model.initial_carry(batch_size)
    observations = jnp.zeros((batch_size, observation_size), jnp.float32)
    reset = jnp.zeros((batch_size,), jnp.bool_)
    variables = model.init(jax.random.PRNGKey(0), carry, observations, reset)
    state = create_rtrl_state(
        model,
        variables["params"],
        RTRLConfig(learning_rate=1.0e-4, gradient_clip=0.5),
    )
    actions = jnp.arange(batch_size) % action_size
    value_targets = jnp.linspace(-1.0, 1.0, batch_size)

    def loss_function(outputs):
        log_probabilities = jax.nn.log_softmax(outputs["policy"], axis=-1)
        policy_loss = -jnp.take_along_axis(
            log_probabilities, actions[:, None], axis=-1
        ).mean()
        value_loss = jnp.mean((outputs["value"] - value_targets) ** 2)
        return policy_loss + 0.5 * value_loss

    state, carry, loss = online_update(
        state, carry, observations, reset, loss_function
    )
    print({"loss": float(loss), "state_shape": carry[0].shape})


if __name__ == "__main__":
    main()
