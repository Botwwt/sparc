from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _formal_coordinate_forward(
    projected, phase_amplitude, radial_amplitude,
    eta, delta, phase_coordinate, radial_raw,
    rows: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    phase_projection = tl.load(
        projected + offset * 2, mask=mask, other=0.0
    ).to(tl.float32)
    radial_projection = tl.load(
        projected + offset * 2 + 1, mask=mask, other=0.0
    ).to(tl.float32)
    phase = libdevice.tanh(phase_projection)
    radial = libdevice.tanh(radial_projection)
    radial_coordinate = radial / (1.0 + radial * radial)
    phase_strength = libdevice.tanh(
        tl.load(phase_amplitude).to(tl.float32)
    )
    radial_strength = libdevice.tanh(
        tl.load(radial_amplitude).to(tl.float32)
    )
    tl.store(phase_coordinate + offset, phase, mask=mask)
    tl.store(radial_raw + offset, radial, mask=mask)
    tl.store(delta + offset, SCALE * phase_strength * phase, mask=mask)
    tl.store(eta + offset, SCALE * radial_strength * radial_coordinate, mask=mask)

def formal_sparc_coordinates(projected, phase_amplitude, radial_amplitude, modes):
    from backward.control import _FormalSPARCCoordinates
    return _FormalSPARCCoordinates.apply(
        projected, phase_amplitude, radial_amplitude, int(modes)
    )

def formal_sparc_coordinates_forward_cache(
    projected, phase_amplitude, radial_amplitude, modes,
):
    projected = projected.contiguous()
    output_shape = projected.shape[:-1]
    eta = torch.empty(output_shape, device=projected.device, dtype=torch.float32)
    delta = torch.empty_like(eta)
    phase_coordinate = torch.empty_like(eta)
    radial_raw = torch.empty_like(eta)
    rows = eta.numel()
    block = 256
    _formal_coordinate_forward[(triton.cdiv(rows, block),)](
        projected, phase_amplitude, radial_amplitude,
        eta, delta, phase_coordinate, radial_raw,
        rows=rows, SCALE=1.0 / math.sqrt(int(modes)),
        BLOCK=block, num_warps=4,
    )
    return eta, delta, phase_coordinate, radial_raw

def formal_sparc_controller_recompute(
    x, phase_direction, radial_direction,
    phase_amplitude, radial_amplitude, modes,
):
    from backward.control import _FormalSPARCControllerRecompute
    return _FormalSPARCControllerRecompute.apply(
        x, phase_direction, radial_direction,
        phase_amplitude, radial_amplitude, int(modes),
    )


@triton.jit
def _controller_partial(x, weight, partial, rows: tl.constexpr,
                        width: tl.constexpr, tiles: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    lane = tile * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    value = tl.load(x + row * width + lane, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight + lane, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + width + lane, mask=mask, other=0.0).to(tl.float32)
    base = (row * tiles + tile) * 2
    tl.store(partial + base, tl.sum(value * w0, axis=0))
    tl.store(partial + base + 1, tl.sum(value * w1, axis=0))


@triton.jit
def _controller_reduce(partial, bias, output,
                       rows: tl.constexpr, tiles: tl.constexpr,
                       REDUCE: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, REDUCE)
    mask = tile < tiles
    base = row * tiles * 2
    value0 = tl.load(partial + base + tile * 2, mask=mask, other=0.0)
    value1 = tl.load(partial + base + tile * 2 + 1, mask=mask, other=0.0)
    tl.store(output + row * 2, tl.sum(value0, axis=0) + tl.load(bias))
    tl.store(output + row * 2 + 1,
             tl.sum(value1, axis=0) + tl.load(bias + 1))


def shared_controller_projection(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    from backward.control import _SharedControllerProjection
    return _SharedControllerProjection.apply(x, weight, bias)




def reference_controller_projection_recompute(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    from backward.control import _ReferenceControllerRecompute
    return _ReferenceControllerRecompute.apply(x, weight, bias)


def reference_forward_triton_backward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    from backward.control import _ReferenceForwardTritonBackward
    return _ReferenceForwardTritonBackward.apply(x, weight, bias)


def make_controls(module, x: torch.Tensor):
    directions = torch.stack((module.phase_direction, module.radial_direction))
    with torch.autocast(device_type=x.device.type, enabled=False):
        projected = F.linear(
            x.float(), directions[:, :-1].float(), directions[:, -1].float()
        )
        inactive_energy_slot = 0.0 * module.energy_response.sum()
        return (
            math.log(16.0) * torch.tanh(projected[..., 1]) + inactive_energy_slot,
            (math.pi / 2.0) * torch.tanh(projected[..., 0]),
        )

def make_scan_inputs(module, x: torch.Tensor):
    eta, delta = make_controls(module, x)
    if x.is_cuda:
        from backward.control import latest_sparc_write
        write_r, write_i = latest_sparc_write(
            x, eta, delta, module.nu_log, module.write_log_gain,
            module.write_phase_response, module.write_radial_response,
        )
        return eta, delta, write_r, write_i
    nu = torch.exp(module.nu_log.float())
    effective_nu = nu * torch.exp(eta[..., None])
    ratio = torch.sqrt(
        (-torch.expm1(-2.0 * effective_nu)) / (-torch.expm1(-2.0 * nu))
    )
    phase, radial = delta / (math.pi / 2.0), eta / math.log(16.0)
    gate = 2.0 * torch.sigmoid(
        phase[..., None] * module.write_phase_response
        + radial[..., None] * module.write_radial_response
    )
    gain = torch.exp(module.write_log_gain) * ratio * gate
    write_r = torch.tanh(x[..., :module.modes].float()) * gain
    write_i = torch.tanh(x[..., module.modes:].float()) * gain
    return eta, delta, write_r.to(x.dtype), write_i.to(x.dtype)


@triton.jit
def _latest_write_forward(
    raw_x, eta, delta, nu_log, write_log_gain, phase_response, radial_response,
    write_r, write_i, rows: tl.constexpr, modes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, mode_block = tl.program_id(0), tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    eta_value = tl.load(eta + row).to(tl.float32)
    delta_value = tl.load(delta + row).to(tl.float32)
    effective_nu = nu * tl.exp(eta_value)
    numerator = -libdevice.expm1(-2.0 * effective_nu)
    denominator = -libdevice.expm1(-2.0 * nu)
    ratio = tl.sqrt(numerator / denominator)
    phase = delta_value * 0.6366197723675814
    radial = eta_value * 0.36067376022224085
    q = (
        phase * tl.load(phase_response + lane, mask=mask, other=0.0)
        + radial * tl.load(radial_response + lane, mask=mask, other=0.0)
    )
    gate = 2.0 * tl.sigmoid(q)
    gain = tl.exp(tl.load(write_log_gain + lane, mask=mask, other=0.0)) * ratio * gate
    raw_offset = row * (2 * modes) + lane
    out_offset = row * modes + lane
    content_r = libdevice.tanh(tl.load(raw_x + raw_offset, mask=mask, other=0.0).to(tl.float32))
    content_i = libdevice.tanh(tl.load(raw_x + raw_offset + modes, mask=mask, other=0.0).to(tl.float32))
    tl.store(write_r + out_offset, gain * content_r, mask=mask)
    tl.store(write_i + out_offset, gain * content_i, mask=mask)
