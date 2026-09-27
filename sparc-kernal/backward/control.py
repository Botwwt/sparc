from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from forward.control import _latest_write_forward

@triton.jit
def _latest_write_backward(
    raw_x, eta, delta, nu_log, write_log_gain, phase_response, radial_response,
    grad_write_r, grad_write_i, grad_x, grad_eta, grad_delta,
    grad_nu_log, grad_write_log_gain, grad_phase_response, grad_radial_response,
    rows: tl.constexpr, modes: tl.constexpr, ROW_CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mode_block, row_chunk = tl.program_id(0), tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    phase_weight = tl.load(phase_response + lane, mask=mask, other=0.0).to(tl.float32)
    radial_weight = tl.load(radial_response + lane, mask=mask, other=0.0).to(tl.float32)
    log_gain = tl.load(write_log_gain + lane, mask=mask, other=0.0).to(tl.float32)
    exp_minus_two_nu = tl.exp(-2.0 * nu)
    denominator = -libdevice.expm1(-2.0 * nu)
    static_energy_derivative = nu * exp_minus_two_nu / denominator
    total_nu = tl.zeros((BLOCK,), tl.float32)
    total_gain = tl.zeros((BLOCK,), tl.float32)
    total_phase_response = tl.zeros((BLOCK,), tl.float32)
    total_radial_response = tl.zeros((BLOCK,), tl.float32)
    for local_row in tl.static_range(0, ROW_CHUNK):
        row = row_chunk * ROW_CHUNK + local_row
        valid = row < rows
        eta_value = tl.load(eta + row, mask=valid, other=0.0).to(tl.float32)
        delta_value = tl.load(delta + row, mask=valid, other=0.0).to(tl.float32)
        effective_nu = nu * tl.exp(eta_value)
        exp_minus_two_effective = tl.exp(-2.0 * effective_nu)
        numerator = -libdevice.expm1(-2.0 * effective_nu)
        ratio = tl.sqrt(numerator / denominator)
        phase = delta_value * 0.6366197723675814
        radial = eta_value * 0.36067376022224085
        gate = 2.0 * tl.sigmoid(phase * phase_weight + radial * radial_weight)
        gain = tl.exp(log_gain) * ratio * gate
        raw_offset = row * (2 * modes) + lane
        out_offset = row * modes + lane
        raw_r = tl.load(raw_x + raw_offset, mask=valid & mask, other=0.0).to(tl.float32)
        raw_i = tl.load(raw_x + raw_offset + modes, mask=valid & mask, other=0.0).to(tl.float32)
        content_r, content_i = libdevice.tanh(raw_r), libdevice.tanh(raw_i)
        grad_r = tl.load(grad_write_r + out_offset, mask=valid & mask, other=0.0).to(tl.float32)
        grad_i = tl.load(grad_write_i + out_offset, mask=valid & mask, other=0.0).to(tl.float32)
        gain_gradient = gain * (grad_r * content_r + grad_i * content_i)
        gate_gradient = gain_gradient * (1.0 - 0.5 * gate)
        dynamic_energy_derivative = (
            effective_nu * exp_minus_two_effective / numerator
        )
        eta_gradient = (
            gain_gradient * dynamic_energy_derivative
            + gate_gradient * radial_weight * 0.36067376022224085
        )
        delta_gradient = gate_gradient * phase_weight * 0.6366197723675814
        tl.atomic_add(
            grad_eta + row,
            tl.sum(tl.where(valid & mask, eta_gradient, 0.0), axis=0),
            mask=valid,
        )
        tl.atomic_add(
            grad_delta + row,
            tl.sum(tl.where(valid & mask, delta_gradient, 0.0), axis=0),
            mask=valid,
        )
        tl.store(
            grad_x + raw_offset, grad_r * gain * (1.0 - content_r * content_r),
            mask=valid & mask,
        )
        tl.store(
            grad_x + raw_offset + modes, grad_i * gain * (1.0 - content_i * content_i),
            mask=valid & mask,
        )
        total_nu += tl.where(
            valid, gain_gradient * (dynamic_energy_derivative - static_energy_derivative), 0.0
        )
        total_gain += tl.where(valid, gain_gradient, 0.0)
        total_phase_response += tl.where(valid, gate_gradient * phase, 0.0)
        total_radial_response += tl.where(valid, gate_gradient * radial, 0.0)
    tl.atomic_add(grad_nu_log + lane, total_nu, mask=mask)
    tl.atomic_add(grad_write_log_gain + lane, total_gain, mask=mask)
    tl.atomic_add(grad_phase_response + lane, total_phase_response, mask=mask)
    tl.atomic_add(grad_radial_response + lane, total_radial_response, mask=mask)


class _LatestSPARCWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_x, eta, delta, nu_log, write_log_gain,
                phase_response, radial_response):
        raw_x, eta, delta = raw_x.contiguous(), eta.contiguous(), delta.contiguous()
        batch, length, width = raw_x.shape
        modes, rows = width // 2, batch * length
        write_r = torch.empty((batch, length, modes), device=raw_x.device, dtype=raw_x.dtype)
        write_i = torch.empty_like(write_r)
        block = min(128, triton.next_power_of_2(modes))
        _latest_write_forward[(rows, triton.cdiv(modes, block))](
            raw_x, eta, delta, nu_log, write_log_gain, phase_response, radial_response,
            write_r, write_i, rows=rows, modes=modes, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        ctx.save_for_backward(
            raw_x, eta, delta, nu_log, write_log_gain,
            phase_response, radial_response,
        )
        return write_r, write_i

    @staticmethod
    def backward(ctx, grad_write_r, grad_write_i):
        (raw_x, eta, delta, nu_log, write_log_gain,
         phase_response, radial_response) = ctx.saved_tensors
        rows, modes = eta.numel(), nu_log.numel()
        grad_x = torch.empty_like(raw_x)
        grad_eta, grad_delta = torch.zeros_like(eta), torch.zeros_like(delta)
        parameter_grads = [torch.zeros_like(nu_log) for _ in range(4)]
        block, row_chunk = min(128, triton.next_power_of_2(modes)), 16
        _latest_write_backward[(triton.cdiv(modes, block), triton.cdiv(rows, row_chunk))](
            raw_x, eta, delta, nu_log, write_log_gain, phase_response, radial_response,
            grad_write_r.contiguous(), grad_write_i.contiguous(), grad_x,
            grad_eta, grad_delta, *parameter_grads,
            rows=rows, modes=modes, ROW_CHUNK=row_chunk, BLOCK=block,
            num_warps=4 if block >= 64 else 2,
        )
        return grad_x, grad_eta, grad_delta, *parameter_grads


def latest_sparc_write(raw_x, eta, delta, nu_log, write_log_gain,
                      phase_response, radial_response):
    return _LatestSPARCWrite.apply(
        raw_x, eta, delta, nu_log, write_log_gain,
        phase_response, radial_response,
    )

@triton.jit
def _reduce_shared_control_gradients(partial_eta, partial_delta,
                                     grad_eta, grad_delta,
                                     rows: tl.constexpr,
                                     mode_blocks: tl.constexpr,
                                     REDUCE_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    mask = offsets < mode_blocks
    base = row * mode_blocks + offsets
    eta_value = tl.sum(tl.load(partial_eta + base, mask=mask, other=0.0), axis=0)
    delta_value = tl.sum(tl.load(partial_delta + base, mask=mask, other=0.0), axis=0)
    tl.store(grad_eta + row, eta_value)
    tl.store(grad_delta + row, delta_value)


@triton.jit
def _sparc_fused_controller_backward(
    raw_x, grad_write_x, grad_eta, grad_delta,
    phase_coordinate, radial_raw,
    normalized_phase, normalized_radial,
    phase_amplitude, radial_amplitude,
    grad_x, direction_partial, amplitude_partial,
    rows: tl.constexpr, width: tl.constexpr,
    row_chunks: tl.constexpr, ROW_CHUNK: tl.constexpr,
    SCALE: tl.constexpr, ROUND_INPUT_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    width_block = tl.program_id(0)
    row_chunk = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    augmented_width = width + 1
    lane_mask = lane < augmented_width
    input_lane_mask = lane < width
    norm_phase = tl.load(
        normalized_phase + lane, mask=lane_mask, other=0.0
    ).to(tl.float32)
    norm_radial = tl.load(
        normalized_radial + lane, mask=lane_mask, other=0.0
    ).to(tl.float32)
    phase_strength = libdevice.tanh(
        tl.load(phase_amplitude).to(tl.float32)
    )
    radial_strength = libdevice.tanh(
        tl.load(radial_amplitude).to(tl.float32)
    )
    partial_phase = tl.zeros((BLOCK,), tl.float32)
    partial_radial = tl.zeros((BLOCK,), tl.float32)
    partial_phase_amplitude = 0.0
    partial_radial_amplitude = 0.0
    for local_row in tl.range(0, ROW_CHUNK, 1, num_stages=1):
        row = row_chunk * ROW_CHUNK + local_row
        valid_row = row < rows
        phase = tl.load(
            phase_coordinate + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        radial = tl.load(
            radial_raw + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        g_eta = tl.load(
            grad_eta + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        g_delta = tl.load(
            grad_delta + row, mask=valid_row, other=0.0
        ).to(tl.float32)
        radial_square = radial * radial
        denominator = 1.0 + radial_square
        radial_coordinate = radial / denominator
        radial_q_derivative = (
            (1.0 - radial_square) / (denominator * denominator)
        )
        grad_phase_projection = (
            g_delta * SCALE * phase_strength * (1.0 - phase * phase)
        )
        grad_radial_projection = (
            g_eta * SCALE * radial_strength
            * radial_q_derivative * (1.0 - radial_square)
        )
        input_value = tl.load(
            raw_x + row * width + lane,
            mask=valid_row & input_lane_mask, other=0.0,
        ).to(tl.float32)
        augmented_value = tl.where(lane < width, input_value, 1.0)
        partial_phase += tl.where(
            valid_row & lane_mask,
            grad_phase_projection * augmented_value, 0.0,
        )
        partial_radial += tl.where(
            valid_row & lane_mask,
            grad_radial_projection * augmented_value, 0.0,
        )
        controller_dx = (
            grad_phase_projection * norm_phase
            + grad_radial_projection * norm_radial
        )
        if ROUND_INPUT_BF16:
            controller_dx = controller_dx.to(tl.bfloat16).to(tl.float32)
        write_dx = tl.load(
            grad_write_x + row * width + lane,
            mask=valid_row & input_lane_mask, other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_x + row * width + lane, write_dx + controller_dx,
            mask=valid_row & input_lane_mask,
        )
        partial_phase_amplitude += tl.where(
            valid_row,
            g_delta * SCALE * phase *
            (1.0 - phase_strength * phase_strength), 0.0,
        )
        partial_radial_amplitude += tl.where(
            valid_row,
            g_eta * SCALE * radial_coordinate *
            (1.0 - radial_strength * radial_strength), 0.0,
        )
    partial_offset = row_chunk * 2 * augmented_width + lane
    tl.store(direction_partial + partial_offset, partial_phase, mask=lane_mask)
    tl.store(
        direction_partial + partial_offset + augmented_width,
        partial_radial, mask=lane_mask,
    )
    first_width_block = width_block == 0
    tl.store(
        amplitude_partial + row_chunk * 2, partial_phase_amplitude,
        mask=first_width_block,
    )
    tl.store(
        amplitude_partial + row_chunk * 2 + 1,
        partial_radial_amplitude, mask=first_width_block,
    )


from forward.control import (
    _controller_partial,
    _controller_reduce,
    _formal_coordinate_forward,
)

@triton.jit
def _formal_coordinate_backward(
    phase_coordinate, radial_raw, grad_eta, grad_delta,
    phase_amplitude, radial_amplitude, grad_projected, partial_amplitude,
    rows: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):


    program = tl.program_id(0)
    offset = program * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    phase = tl.load(phase_coordinate + offset, mask=mask, other=0.0).to(tl.float32)
    radial = tl.load(radial_raw + offset, mask=mask, other=0.0).to(tl.float32)
    g_eta = tl.load(grad_eta + offset, mask=mask, other=0.0).to(tl.float32)
    g_delta = tl.load(grad_delta + offset, mask=mask, other=0.0).to(tl.float32)
    phase_s = libdevice.tanh(tl.load(phase_amplitude).to(tl.float32))
    radial_s = libdevice.tanh(tl.load(radial_amplitude).to(tl.float32))
    radial_square = radial * radial
    denominator = 1.0 + radial_square
    radial_coordinate = radial / denominator
    radial_derivative = (1.0 - radial_square) / (denominator * denominator)
    grad_phase = g_delta * SCALE * phase_s * (1.0 - phase * phase)
    grad_radial = (
        g_eta * SCALE * radial_s * radial_derivative * (1.0 - radial_square)
    )
    tl.store(grad_projected + offset * 2, grad_phase, mask=mask)
    tl.store(grad_projected + offset * 2 + 1, grad_radial, mask=mask)
    phase_amplitude = g_delta * SCALE * phase * (1.0 - phase_s * phase_s)
    radial_amplitude = (
        g_eta * SCALE * radial_coordinate * (1.0 - radial_s * radial_s)
    )
    tl.store(partial_amplitude + program * 2,
             tl.sum(tl.where(mask, phase_amplitude, 0.0), axis=0))
    tl.store(partial_amplitude + program * 2 + 1,
             tl.sum(tl.where(mask, radial_amplitude, 0.0), axis=0))

@triton.jit
def _formal_coordinate_reduce(
    partial_amplitude, grad_phase_amplitude, grad_radial_amplitude,
    blocks: tl.constexpr, REDUCE: tl.constexpr,
):
    offset = tl.arange(0, REDUCE)
    mask = offset < blocks
    phase = tl.load(partial_amplitude + offset * 2, mask=mask, other=0.0)
    radial = tl.load(partial_amplitude + offset * 2 + 1, mask=mask, other=0.0)
    tl.store(grad_phase_amplitude, tl.sum(phase, axis=0))
    tl.store(grad_radial_amplitude, tl.sum(radial, axis=0))

class _FormalSPARCCoordinates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, phase_amplitude, radial_amplitude, modes: int):
        projected = projected.contiguous()
        output_shape = projected.shape[:-1]
        eta = torch.empty(output_shape, device=projected.device, dtype=torch.float32)
        delta = torch.empty_like(eta)
        phase_coordinate = torch.empty_like(eta)
        radial_raw = torch.empty_like(eta)
        scale = 1.0 / math.sqrt(modes)
        rows = eta.numel()
        block = 256
        _formal_coordinate_forward[(triton.cdiv(rows, block),)](
            projected, phase_amplitude, radial_amplitude,
            eta, delta, phase_coordinate, radial_raw,
            rows=rows, SCALE=scale, BLOCK=block, num_warps=4,
        )
        ctx.save_for_backward(
            phase_coordinate, radial_raw, phase_amplitude, radial_amplitude
        )
        ctx.scale = scale
        return eta, delta

    @staticmethod
    def backward(ctx, grad_eta, grad_delta):
        phase_coordinate, radial_raw, phase_amplitude, radial_amplitude = ctx.saved_tensors
        grad_eta, grad_delta = grad_eta.contiguous(), grad_delta.contiguous()
        rows = grad_eta.numel()
        block = 256
        blocks = triton.cdiv(rows, block)
        grad_projected = torch.empty(
            (*grad_eta.shape, 2), device=grad_eta.device, dtype=torch.float32
        )
        partial_amplitude = torch.empty(
            (blocks, 2), device=grad_eta.device, dtype=torch.float32
        )
        _formal_coordinate_backward[(blocks,)](
            phase_coordinate, radial_raw, grad_eta, grad_delta,
            phase_amplitude, radial_amplitude, grad_projected, partial_amplitude,
            rows=rows, SCALE=ctx.scale, BLOCK=block, num_warps=4,
        )
        grad_phase_amplitude = torch.empty_like(phase_amplitude)
        grad_radial_amplitude = torch.empty_like(radial_amplitude)
        reduce = triton.next_power_of_2(blocks)
        _formal_coordinate_reduce[(1,)](
            partial_amplitude, grad_phase_amplitude, grad_radial_amplitude,
            blocks=blocks, REDUCE=reduce,
            num_warps=1 if reduce <= 32 else 4,
        )
        return grad_projected, grad_phase_amplitude, grad_radial_amplitude, None

class _FormalSPARCControllerRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, phase_direction, radial_direction,
                phase_amplitude, radial_amplitude, modes: int):
        directions = F.normalize(
            torch.stack((phase_direction, radial_direction)).float(), dim=1
        )
        projected = F.linear(
            x.float(), directions[:, :-1], directions[:, -1]
        )
        phase_coordinate = torch.tanh(projected[..., 0])
        radial_raw = torch.tanh(projected[..., 1])
        radial_coordinate = radial_raw / (1.0 + radial_raw.square())
        scale = 1.0 / math.sqrt(modes)
        delta = scale * torch.tanh(phase_amplitude) * phase_coordinate
        eta = scale * torch.tanh(radial_amplitude) * radial_coordinate
        ctx.save_for_backward(
            x, phase_direction, radial_direction,
            phase_amplitude, radial_amplitude,
        )
        ctx.modes = int(modes)
        return eta, delta

    @staticmethod
    def backward(ctx, grad_eta, grad_delta):
        (x, phase_direction, radial_direction,
         phase_amplitude, radial_amplitude) = ctx.saved_tensors
        with torch.enable_grad():
            x_replay = x.detach().float().requires_grad_(True)
            phase_replay = phase_direction.detach().requires_grad_(True)
            radial_replay = radial_direction.detach().requires_grad_(True)
            phase_amplitude_replay = phase_amplitude.detach().requires_grad_(True)
            radial_amplitude_replay = radial_amplitude.detach().requires_grad_(True)
            directions = F.normalize(
                torch.stack((phase_replay, radial_replay)).float(), dim=1
            )
            projected = F.linear(
                x_replay, directions[:, :-1], directions[:, -1]
            )
            phase_coordinate = torch.tanh(projected[..., 0])
            radial_raw = torch.tanh(projected[..., 1])
            radial_coordinate = radial_raw / (1.0 + radial_raw.square())
            scale = 1.0 / math.sqrt(ctx.modes)
            delta = scale * torch.tanh(phase_amplitude_replay) * phase_coordinate
            eta = scale * torch.tanh(radial_amplitude_replay) * radial_coordinate
            gradients = torch.autograd.grad(
                (eta, delta),
                (x_replay, phase_replay, radial_replay,
                 phase_amplitude_replay, radial_amplitude_replay),
                (grad_eta.contiguous(), grad_delta.contiguous()),
            )
        return gradients[0].to(x.dtype), *gradients[1:], None

@triton.jit
def _controller_grad_x(grad_output, weight, grad_x,
                       rows: tl.constexpr, width: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    width_block = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    grad0 = tl.load(grad_output + row * 2).to(tl.float32)
    grad1 = tl.load(grad_output + row * 2 + 1).to(tl.float32)
    w0 = tl.load(weight + lane, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + width + lane, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_x + row * width + lane, grad0 * w0 + grad1 * w1,
             mask=mask)


@triton.jit
def _controller_grad_weight_partial(x, grad_output, partial,
                                    rows: tl.constexpr,
                                    width: tl.constexpr,
                                    row_chunks: tl.constexpr,
                                    ROW_CHUNK: tl.constexpr,
                                    BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    row_chunk = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    lane_mask = lane < width
    accumulator0 = tl.zeros((BLOCK,), tl.float32)
    accumulator1 = tl.zeros((BLOCK,), tl.float32)
    for local_row in tl.range(0, ROW_CHUNK, 1, num_stages=1):
        row = row_chunk * ROW_CHUNK + local_row
        valid = row < rows
        value = tl.load(
            x + row * width + lane,
            mask=lane_mask & valid,
            other=0.0,
        ).to(tl.float32)
        grad0 = tl.load(
            grad_output + row * 2, mask=valid, other=0.0
        ).to(tl.float32)
        grad1 = tl.load(
            grad_output + row * 2 + 1, mask=valid, other=0.0
        ).to(tl.float32)
        accumulator0 += value * grad0
        accumulator1 += value * grad1
    base = row_chunk * 2 * width + lane
    tl.store(partial + base, accumulator0, mask=lane_mask)
    tl.store(partial + base + width, accumulator1, mask=lane_mask)


class _SharedControllerProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if x.ndim != 3 or weight.shape != (2, x.shape[-1]) or bias.shape != (2,):
            raise ValueError("controller expects x[B,L,D], weight[2,D], bias[2]")
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        rows, width = x.numel() // x.shape[-1], x.shape[-1]
        block = 256
        tiles = triton.cdiv(width, block)
        partial = torch.empty((rows, tiles, 2), device=x.device, dtype=torch.float32)
        output = torch.empty((rows, 2), device=x.device, dtype=torch.float32)
        _controller_partial[(rows, tiles)](
            x, weight, partial, rows=rows, width=width, tiles=tiles,
            BLOCK=block, num_warps=4,
        )
        reduce = triton.next_power_of_2(tiles)
        _controller_reduce[(rows,)](
            partial, bias, output, rows=rows, tiles=tiles, REDUCE=reduce,
            num_warps=1,
        )
        ctx.save_for_backward(x, weight)
        return output.view(*x.shape[:-1], 2)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, 2)
        rows, width = grad_output.shape[0], x.shape[-1]
        grad_x = torch.empty_like(x)
        block = 128
        width_blocks = triton.cdiv(width, block)
        _controller_grad_x[(rows, width_blocks)](
            grad_output, weight, grad_x,
            rows=rows, width=width, BLOCK=block, num_warps=4,
        )
        row_chunk = 128
        row_chunks = triton.cdiv(rows, row_chunk)
        partial = torch.empty(
            (row_chunks, 2, width), device=x.device, dtype=torch.float32
        )
        _controller_grad_weight_partial[(width_blocks, row_chunks)](
            x, grad_output, partial,
            rows=rows, width=width, row_chunks=row_chunks,
            ROW_CHUNK=row_chunk, BLOCK=block, num_warps=4,
        )
        grad_weight = partial.sum(dim=0)
        grad_bias = grad_output.sum(dim=0)
        return grad_x, grad_weight, grad_bias

class _ReferenceControllerRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.nn.functional.linear(x.float(), weight, bias)
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        with torch.enable_grad():
            x_fp32 = x.detach().float().requires_grad_(True)
            weight_replay = weight.detach().requires_grad_(True)
            replay = torch.nn.functional.linear(x_fp32, weight_replay, None)
            grad_x_fp32, grad_weight = torch.autograd.grad(
                replay, (x_fp32, weight_replay), grad_output.contiguous()
            )
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        return grad_x_fp32.to(x.dtype), grad_weight, grad_bias


def reference_controller_projection_recompute(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _ReferenceControllerRecompute.apply(x, weight, bias)


class _ReferenceForwardTritonBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        output = torch.nn.functional.linear(x.float(), weight, bias)
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, 2)
        rows, width = grad_output.shape[0], x.shape[-1]
        grad_x = torch.empty_like(x)
        block = 128
        width_blocks = triton.cdiv(width, block)
        _controller_grad_x[(rows, width_blocks)](
            grad_output, weight, grad_x,
            rows=rows, width=width, BLOCK=block, num_warps=4,
        )
        row_chunk = 128
        row_chunks = triton.cdiv(rows, row_chunk)
        partial = torch.empty(
            (row_chunks, 2, width), device=x.device, dtype=torch.float32
        )
        _controller_grad_weight_partial[(width_blocks, row_chunks)](
            x, grad_output, partial,
            rows=rows, width=width, row_chunks=row_chunks,
            ROW_CHUNK=row_chunk, BLOCK=block, num_warps=4,
        )
        return grad_x, partial.sum(dim=0), grad_output.sum(dim=0)
