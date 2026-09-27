# SPARC JAX Reference Implementation

This package provides a compact, modular JAX implementation of the SPARC networks used in the paper. It includes reusable Flax layers, the actor-critic and sequence-classification networks, and both supported training paths.

## Components

- `SPARCRTRLCell` provides recurrent single-step execution with structured RTRL traces.
- `SPARCSequenceLayer` evaluates full sequences with an associative scan.
- `SPARCActorCritic` provides policy and value outputs for recurrent control.
- `SPARCSequenceClassifier` provides the four-layer residual classification network.
- `create_rtrl_state` and `create_bptt_state` construct the corresponding optimizer states.

## Directory Structure

```text
sparc-impl/
├── README.md
├── requirement.txt
│
├── sparc/
│   ├── __init__.py
│   ├── recurrence.py                # Shared controls, spectral transition, write path, and recurrent update
│   ├── initialization.py            # Spectrum, projection, write, and readout initializers
│   ├── layers.py                    # Reusable online and full-sequence Flax layers
│   │
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── actor_critic.py          # Recurrent actor-critic network
│   │   └── sequence_classifier.py   # Four-layer residual sequence classifier
│   │
│   └── training/
│       ├── __init__.py
│       ├── rtrl.py                  # Structured RTRL traces, resets, gradients, and online updates
│       └── bptt.py                  # Associative scan, optimizer groups, schedules, and BPTT updates
│
└── examples/
    ├── online_control_rtrl.py        # Small actor-critic update using structured RTRL
    └── sequence_classification.py    # Small sequence-classification update using BPTT
```

## Installation

Install the dependencies:

```bash
python -m pip install -r requirement.txt
```

## Run the Examples

Run both commands from the `sparc-impl` directory:

```bash
python examples/online_control_rtrl.py
python examples/sequence_classification.py
```

Each command initializes its network, executes one training update, and prints the loss together with the principal output shape.

## Package Entry Points

The top-level `sparc` package exports the two layers, the two complete networks, both configuration classes, and both training-state constructors. Lower-level recurrence and training functions remain available from their respective modules for integrations that need direct control over state handling or optimization.

```python
from sparc import (
    BPTTConfig,
    RTRLConfig,
    SPARCActorCritic,
    SPARCRTRLCell,
    SPARCSequenceClassifier,
    SPARCSequenceLayer,
    create_bptt_state,
    create_rtrl_state,
)
```
