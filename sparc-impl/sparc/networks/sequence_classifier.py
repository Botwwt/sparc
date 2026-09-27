from flax import linen as nn
import jax
import jax.numpy as jnp
from ..layers import SPARCSequenceLayer


class SPARCSequenceBlock(nn.Module):
    model_width: int
    modes: int
    dropout_rate: float
    training: bool
    r_min: float
    r_max: float
    max_phase: float
    content_activation: str

    @nn.compact
    def __call__(self, inputs):
        hidden = nn.LayerNorm(name="normalization")(inputs)
        hidden = SPARCSequenceLayer(
            model_width=self.model_width,
            modes=self.modes,
            r_min=self.r_min,
            r_max=self.r_max,
            max_phase=self.max_phase,
            content_activation=self.content_activation,
            name="sparc",
        )(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(
            self.dropout_rate, broadcast_dims=(1,), deterministic=not self.training
        )(hidden)
        first = nn.Dense(self.model_width, name="glu_value")(hidden)
        second = nn.Dense(self.model_width, name="glu_gate")(hidden)
        hidden = first * jax.nn.sigmoid(second)
        hidden = nn.Dropout(
            self.dropout_rate, broadcast_dims=(1,), deterministic=not self.training
        )(hidden)
        return inputs + hidden


class SPARCSequenceClassifier(nn.Module):
    class_count: int
    model_width: int
    modes: int
    layer_count: int
    dropout_rate: float
    training: bool
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = 6.28
    content_activation: str = "tanh"

    @nn.compact
    def __call__(self, inputs):
        hidden = nn.Dense(self.model_width, name="embedding")(inputs)
        for index in range(self.layer_count):
            hidden = SPARCSequenceBlock(
                model_width=self.model_width,
                modes=self.modes,
                dropout_rate=self.dropout_rate,
                training=self.training,
                r_min=self.r_min,
                r_max=self.r_max,
                max_phase=self.max_phase,
                content_activation=self.content_activation,
                name=f"layer_{index}",
            )(hidden)
        pooled = jnp.mean(hidden, axis=1)
        logits = nn.Dense(self.class_count, name="classifier")(pooled)
        return nn.log_softmax(logits, axis=-1)
