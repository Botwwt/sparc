import jax.numpy as jnp
from flax import linen as nn
from ..layers import SPARCRTRLCell
from ..training.rtrl import initialize_carry


class SPARCActorCritic(nn.Module):
    action_size: int
    modes: int
    encoder_width: int
    policy_width: int
    value_width: int
    continuous_actions: bool
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    content_activation: str = "tanh"

    @nn.compact
    def __call__(self, carry, observations, reset):
        encoded = jnp.tanh(nn.Dense(self.encoder_width, name="encoder")(observations))
        carry, recurrent = SPARCRTRLCell(
            modes=self.modes,
            r_min=self.r_min,
            r_max=self.r_max,
            max_phase=self.max_phase,
            content_activation=self.content_activation,
            name="sparc",
        )(carry, encoded, reset)
        policy_hidden = jnp.tanh(
            nn.Dense(self.policy_width, name="policy_hidden")(recurrent)
        )
        value_hidden = jnp.tanh(
            nn.Dense(self.value_width, name="value_hidden")(recurrent)
        )
        policy = nn.Dense(self.action_size, name="policy_output")(policy_hidden)
        value = nn.Dense(1, name="value_output")(value_hidden)[..., 0]
        outputs = {"policy": policy, "value": value}
        if self.continuous_actions:
            outputs["log_standard_deviation"] = self.param(
                "log_standard_deviation",
                nn.initializers.zeros,
                (self.action_size,),
            )
        return carry, outputs

    def initial_carry(self, batch_size):
        return initialize_carry(batch_size, self.encoder_width, self.modes)
