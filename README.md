# SPARC

Code for **Shared Phase and Retention Control for Efficient Adaptive Spectral Recurrence**.

## Implementations

- [JAX reference implementation](sparc-impl/README.md): Flax layers, actor-critic and sequence-classification networks, structured RTRL and BPTT training, and usage examples.
- [GPU-efficient implementation](sparc-kernal/README.md): a differentiable PyTorch module with Triton forward and backward operators for sequence training and single-step decoding.

See each implementation's README for dependencies, installation, and usage.
