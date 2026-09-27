
from __future__ import annotations

import math

import torch
from torch import nn

from common import (
    SparcParameters,
    _affine_scan_real2,
    make_sparc_parameters,
    sparc_sequential_from_shared,
    sparc_chunked_compressed,
    sparc_serial,
    sparc_tree_materialized,
)
from backward.scan import (
    complex_scan,
    complex_scan_with_last,
    real_chunk_scan,
    real_scan,
    real_scan_with_last,
    sparc_chunk_scan,
    sparc_scan,
    sparc_scan_precomputed_decay,
    sparc_scan_with_last,
    sparc_tiled_serial_scan,
    sparc_training_scan_dispatch,
)
from forward.affine import sparc_affine_tile_scan
from forward.control import (
    formal_sparc_controller_recompute,
    formal_sparc_coordinates,
    formal_sparc_coordinates_forward_cache,
    reference_controller_projection_recompute,
    reference_forward_triton_backward,
    shared_controller_projection,
)
from forward.scan import (
    pack_sparc_parameters,
    sparc_packed_reference,
    sparc_triton_auto,
    sparc_triton_chunked,
    sparc_triton_decode,
    sparc_triton_decode_split,
    sparc_triton_serial,
)
from forward.decode import (
    prepare_decode,
    run_decode_step,
    sparc_decode,
    sparc_decode_blocked,
)
from forward.scan import run_sequence


class SPARCMixer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        if width % 2:
            raise ValueError("SPARC rnn_width must be even")
        self.width = width
        self.modes = width // 2
        radius_squared = torch.rand(self.modes).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        nu = -0.5 * torch.log(radius_squared)
        theta = (2.0 * math.pi * torch.rand(self.modes)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        self.nu_log = nn.Parameter(torch.log(nu))
        self.theta_log = nn.Parameter(torch.log(theta))
        self.phase_direction = nn.Parameter(torch.zeros(width + 1))
        self.radial_direction = nn.Parameter(torch.zeros(width + 1))
        self.register_buffer("phase_amplitude", torch.zeros(()))
        self.register_buffer("radial_amplitude", torch.zeros(()))
        initial_gain = torch.log(torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8)
        self.write_log_gain = nn.Parameter(initial_gain)
        self.energy_response = nn.Parameter(torch.full((self.modes,), 0.5493061443340548))
        self.write_phase_response = nn.Parameter(torch.zeros(self.modes))
        self.write_radial_response = nn.Parameter(torch.zeros(self.modes))
        self._inference_fused = False
        self._decode_backend = "auto"
        self._inference_pack = None
        self._tiled_training_block_size = 128
        self._controller_projection_dtype = "fp32_fused_coords"
        self.scan_backend = "auto"

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        return run_sequence(self, x, return_cache)

    def set_scan_backend(self, backend: str) -> None:
        self.scan_backend = backend

    def set_tiled_training_block_size(self, block_size: int) -> None:
        if block_size not in (32, 64, 128, 256):
            raise ValueError(block_size)
        self._tiled_training_block_size = block_size

    def set_controller_projection_dtype(self, dtype: str) -> None:
        if dtype not in (
            "fp32", "bf16", "triton_fp32",
            "fp32_rounded", "triton_fp32_rounded", "fp32_recompute",
            "fp32_triton_backward", "fp32_cache_save",
            "fp32_cache_recompute", "fp32_fused_coords",
        ):
            raise ValueError(dtype)
        self._controller_projection_dtype = dtype

    def prepare_inference(self) -> None:
        prepare_decode(self)

    def set_decode_backend(self, backend: str) -> None:
        if backend not in (
            "auto", "fused", "packed",
            "blocked16", "blocked32", "blocked64",
        ):
            raise ValueError(backend)
        self._decode_backend = backend

    def step(self, x: torch.Tensor, state, segment_pos):
        return run_decode_step(self, x, state, segment_pos)


affine_scan_real2 = _affine_scan_real2

__all__ = [name for name in globals() if not name.startswith("_")]
