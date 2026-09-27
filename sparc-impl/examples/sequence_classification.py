from pathlib import Path
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparc import BPTTConfig, SPARCSequenceClassifier, create_bptt_state
from sparc.training.bptt import classification_update


def main():
    batch_size = 2
    sequence_length = 32
    class_count = 4
    model = SPARCSequenceClassifier(
        class_count=class_count,
        model_width=64,
        modes=64,
        layer_count=4,
        dropout_rate=0.1,
        training=True,
    )
    inputs = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, sequence_length, 1)
    )
    labels = jnp.arange(batch_size) % class_count
    variables = model.init(
        {"params": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        inputs,
    )
    config = BPTTConfig(
        learning_rate=1.95e-3,
        recurrent_learning_rate_scale=0.25,
        minimum_learning_rate=1.0e-6,
        warmup_steps=2,
        total_steps=10,
        weight_decay=0.05,
    )
    state = create_bptt_state(model, variables["params"], config)
    state, loss, outputs = classification_update(
        state, inputs, labels, jax.random.PRNGKey(3)
    )
    print({"loss": float(loss), "output_shape": outputs.shape})


if __name__ == "__main__":
    main()
