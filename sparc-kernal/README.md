# SPARC GPU-Efficient Implementation

This directory provides the PyTorch differentiable SPARC module and its Triton forward and backward operators for training and single-step decoding.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with CUDA support
- PyTorch 2.4.0
- Triton 3.0.0

## Directory Structure

```text
sparc-kernal/
├── README.md
├── requirement.txt
├── common.py
├── sparc.py
├── forward/
│   ├── affine.py
│   ├── control.py
│   ├── decode.py
│   └── scan.py
└── backward/
    ├── affine.py
    ├── control.py
    └── scan.py
```

## Usage

Install the dependencies:

```bash
python -m pip install -r requirement.txt
```

### Sequence training

```python
import torch

from sparc import SPARCMixer

x = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
module = SPARCMixer(128).cuda().train()

output, _ = module(x)
loss = output.float().square().mean()
loss.backward()
```

The sequence input and output shape is `[batch, sequence, width]`. `width` must be even and corresponds to twice the number of complex modes. The first and second halves represent the real and imaginary coordinates. With `return_cache=True`, the returned cache is a pair of FP32 tensors, each with shape `[batch, width / 2]`.

The default `auto` scan backend selects an implementation from the input shape and dtype. It can also be selected explicitly with `set_scan_backend("auto")`.

### Single-step decoding

The cache returned by a sequence call can be passed directly to `step`:

```python
module.eval()

with torch.no_grad():
    prompt = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    _, state = module(prompt, return_cache=True)

    module.prepare_inference()
    next_input = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    next_output, state = module.step(next_input, state, segment_pos=None)
```

For decoding, `next_input` and `next_output` have shape `[batch, width]`. The recurrent state remains a pair of FP32 tensors with shape `[batch, width / 2]`.
