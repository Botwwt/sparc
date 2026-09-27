from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .control import make_scan_inputs

@triton.jit
def _latest_sparc_decode(
    x, phase_control, radial_control, nu_log, theta_log, write_log_gain,
    phase_response, radial_response, state_r, state_i, out, last_r, last_i,
    width: tl.constexpr, modes: tl.constexpr, BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    width_mask, mode_mask = lane < width, lane < modes
    values = tl.load(x + batch * width + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_vector = tl.load(phase_control + lane, mask=width_mask, other=0.0).to(tl.float32)
    radial_vector = tl.load(radial_control + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_raw = tl.sum(values * phase_vector, axis=0) + tl.load(phase_control + width)
    radial_raw = tl.sum(values * radial_vector, axis=0) + tl.load(radial_control + width)
    phase_signal = 2.0 * tl.sigmoid(2.0 * phase_raw) - 1.0
    radial_signal = 2.0 * tl.sigmoid(2.0 * radial_raw) - 1.0
    eta = 2.772588722239781 * radial_signal
    delta = 1.5707963267948966 * phase_signal

    nu = tl.exp(tl.load(nu_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    effective_nu = nu * tl.exp(eta)
    rho = tl.exp(-effective_nu)
    actual_phase = theta + (1.0 - rho) * delta
    ar, ai = rho * tl.cos(actual_phase), rho * tl.sin(actual_phase)

    ratio = tl.sqrt(
        -libdevice.expm1(-2.0 * effective_nu) / -libdevice.expm1(-2.0 * nu)
    )
    gate = 2.0 * tl.sigmoid(
        phase_signal * tl.load(phase_response + lane, mask=mode_mask, other=0.0)
        + radial_signal * tl.load(radial_response + lane, mask=mode_mask, other=0.0)
    )
    gain = tl.exp(tl.load(write_log_gain + lane, mask=mode_mask, other=0.0)) * ratio * gate
    content_r = libdevice.tanh(values)
    content_i = libdevice.tanh(tl.load(
        x + batch * width + modes + lane, mask=mode_mask, other=0.0
    ).to(tl.float32))
    previous_r = tl.load(state_r + batch * modes + lane, mask=mode_mask, other=0.0)
    previous_i = tl.load(state_i + batch * modes + lane, mask=mode_mask, other=0.0)
    new_r = ar * previous_r - ai * previous_i + gain * content_r
    new_i = ai * previous_r + ar * previous_i + gain * content_i
    tl.store(last_r + batch * modes + lane, new_r, mask=mode_mask)
    tl.store(last_i + batch * modes + lane, new_i, mask=mode_mask)
    tl.store(out + batch * width + lane, tl.maximum(new_r, 0.0), mask=mode_mask)
    tl.store(out + batch * width + modes + lane, tl.maximum(new_i, 0.0), mask=mode_mask)


@torch.no_grad()
def latest_sparc_decode(module, x: torch.Tensor, state):
    x = x.contiguous()
    batch, width = x.shape
    modes = width // 2
    out = torch.empty_like(x)
    last_r = torch.empty_like(state[0], dtype=torch.float32)
    last_i = torch.empty_like(state[1], dtype=torch.float32)
    block = triton.next_power_of_2(width)
    _latest_sparc_decode[(batch,)](
        x, module.phase_direction, module.radial_direction,
        module.nu_log, module.theta_log, module.write_log_gain,
        module.write_phase_response, module.write_radial_response,
        state[0], state[1], out, last_r, last_i,
        width=width, modes=modes, BLOCK=block,
        num_warps=8 if block >= 256 else 4,
    )
    return out, (last_r, last_i)

@triton.jit
def _sparc_decode(x, phase_direction, radial_direction,
                 phase_amplitude, radial_amplitude,
                 nu_log, theta_log, state_r, state_i,
                 out, last_r, last_i,
                 width: tl.constexpr, modes: tl.constexpr,
                 SCALE: tl.constexpr,
                 BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    width_mask = lane < width
    values = tl.load(x + batch * width + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_vector = tl.load(phase_direction + lane, mask=width_mask, other=0.0).to(tl.float32)
    radial_vector = tl.load(radial_direction + lane, mask=width_mask, other=0.0).to(tl.float32)
    phase_bias = tl.load(phase_direction + width).to(tl.float32)
    radial_bias = tl.load(radial_direction + width).to(tl.float32)
    phase_norm = tl.sqrt(tl.sum(phase_vector * phase_vector, axis=0) + phase_bias * phase_bias)
    radial_norm = tl.sqrt(tl.sum(radial_vector * radial_vector, axis=0) + radial_bias * radial_bias)
    phase_raw = (tl.sum(values * phase_vector, axis=0) + phase_bias) / tl.maximum(phase_norm, 1e-12)
    radial_raw = (tl.sum(values * radial_vector, axis=0) + radial_bias) / tl.maximum(radial_norm, 1e-12)
    phase_selector = 2.0 * tl.sigmoid(2.0 * phase_raw) - 1.0
    radial_selector = 2.0 * tl.sigmoid(2.0 * radial_raw) - 1.0
    physical_radial = radial_selector / (1.0 + radial_selector * radial_selector)
    raw_phase_amplitude = tl.load(phase_amplitude).to(tl.float32)
    raw_radial_amplitude = tl.load(radial_amplitude).to(tl.float32)
    bounded_phase = 2.0 * tl.sigmoid(2.0 * raw_phase_amplitude) - 1.0
    bounded_radial = 2.0 * tl.sigmoid(2.0 * raw_radial_amplitude) - 1.0
    phase_delta = SCALE * bounded_phase * phase_selector
    radial_delta = SCALE * bounded_radial * physical_radial
    mode_mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mode_mask, other=0.0).to(tl.float32))
    rho = tl.exp(-nu * tl.exp(radial_delta))
    phase = theta + phase_delta
    ar, ai = rho * tl.cos(phase), rho * tl.sin(phase)
    gamma = tl.sqrt(tl.maximum(1.0 - tl.exp(-2.0 * nu), 0.0)) + 1.0e-8
    write_r = values * gamma
    write_i_values = tl.load(
        x + batch * width + modes + lane, mask=mode_mask, other=0.0
    ).to(tl.float32)
    write_i = write_i_values * gamma

    previous_r = tl.load(state_r + batch * modes + lane, mask=mode_mask, other=0.0)
    previous_i = tl.load(state_i + batch * modes + lane, mask=mode_mask, other=0.0)
    new_r = ar * previous_r - ai * previous_i + write_r
    new_i = ai * previous_r + ar * previous_i + write_i
    tl.store(last_r + batch * modes + lane, new_r, mask=mode_mask)
    tl.store(last_i + batch * modes + lane, new_i, mask=mode_mask)
    tl.store(out + batch * width + lane, tl.maximum(new_r, 0.0), mask=mode_mask)
    tl.store(out + batch * width + modes + lane, tl.maximum(new_i, 0.0), mask=mode_mask)


@triton.jit
def _sparc_decode_blocked(x, normalized_phase_direction,
                         normalized_radial_direction,
                         phase_scale, radial_scale,
                         nu, cos_theta, sin_theta, gamma,
                         state_r, state_i, out, last_r, last_i,
                         width: tl.constexpr, modes: tl.constexpr,
                         BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < width
    values = tl.load(x + batch * width + d, mask=dmask, other=0.0).to(tl.float32)
    phase_vector = tl.load(
        normalized_phase_direction + d, mask=dmask, other=0.0
    ).to(tl.float32)
    radial_vector = tl.load(
        normalized_radial_direction + d, mask=dmask, other=0.0
    ).to(tl.float32)
    phase_bias = tl.load(normalized_phase_direction + width).to(tl.float32)
    radial_bias = tl.load(normalized_radial_direction + width).to(tl.float32)
    phase_raw = tl.sum(values * phase_vector, axis=0) + phase_bias
    radial_raw = tl.sum(values * radial_vector, axis=0) + radial_bias
    phase_selector = 2.0 * tl.sigmoid(2.0 * phase_raw) - 1.0
    radial_selector = 2.0 * tl.sigmoid(2.0 * radial_raw) - 1.0
    physical_radial = radial_selector / (1.0 + radial_selector * radial_selector)
    phase_delta = tl.load(phase_scale).to(tl.float32) * phase_selector
    radial_delta = tl.load(radial_scale).to(tl.float32) * physical_radial
    local = tl.arange(0, BLOCK_M)
    mode = mode_block * BLOCK_M + local
    mask = mode < modes
    mode_nu = tl.load(nu + mode, mask=mask, other=0.0)
    base_cos = tl.load(cos_theta + mode, mask=mask, other=1.0)
    base_sin = tl.load(sin_theta + mode, mask=mask, other=0.0)
    radius = tl.exp(-mode_nu * tl.exp(radial_delta))
    cos_delta, sin_delta = tl.cos(phase_delta), tl.sin(phase_delta)
    cosine = base_cos * cos_delta - base_sin * sin_delta
    sine = base_sin * cos_delta + base_cos * sin_delta
    ar, ai = radius * cosine, radius * sine
    mode_gamma = tl.load(gamma + mode, mask=mask, other=0.0)
    write_r = tl.load(
        x + batch * width + mode, mask=mask, other=0.0
    ).to(tl.float32) * mode_gamma
    write_i = tl.load(
        x + batch * width + modes + mode, mask=mask, other=0.0
    ).to(tl.float32) * mode_gamma
    previous_r = tl.load(
        state_r + batch * modes + mode, mask=mask, other=0.0
    )
    previous_i = tl.load(
        state_i + batch * modes + mode, mask=mask, other=0.0
    )
    new_r = ar * previous_r - ai * previous_i + write_r
    new_i = ai * previous_r + ar * previous_i + write_i
    tl.store(last_r + batch * modes + mode, new_r, mask=mask)
    tl.store(last_i + batch * modes + mode, new_i, mask=mask)
    tl.store(out + batch * width + mode, tl.maximum(new_r, 0.0), mask=mask)
    tl.store(
        out + batch * width + modes + mode,
        tl.maximum(new_i, 0.0), mask=mask,
    )

@torch.no_grad()
def sparc_decode(x: torch.Tensor, phase_direction: torch.Tensor,
                radial_direction: torch.Tensor, phase_amplitude: torch.Tensor,
                radial_amplitude: torch.Tensor, nu_log: torch.Tensor,
                theta_log: torch.Tensor, state_r: torch.Tensor,
                state_i: torch.Tensor):
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("SPARC decode expects [B, 2M]")
    x = x.contiguous()
    batch, width = x.shape
    modes = width // 2
    if state_r.shape != (batch, modes) or state_i.shape != state_r.shape:
        raise ValueError("SPARC decode state shape mismatch")
    out = torch.empty_like(x)
    last_r = torch.empty_like(state_r, dtype=torch.float32)
    last_i = torch.empty_like(last_r)
    block = triton.next_power_of_2(width)
    _sparc_decode[(batch,)](
        x, phase_direction, radial_direction,
        phase_amplitude, radial_amplitude, nu_log, theta_log,
        state_r, state_i, out, last_r, last_i,
        width=width, modes=modes, SCALE=1.0 / (modes ** 0.5), BLOCK=block,
        num_warps=8 if block >= 256 else 4,
    )
    return out, (last_r, last_i)


@torch.no_grad()
def sparc_decode_blocked(
    x: torch.Tensor,
    normalized_phase_direction: torch.Tensor,
    normalized_radial_direction: torch.Tensor,
    phase_scale: torch.Tensor,
    radial_scale: torch.Tensor,
    nu: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    gamma: torch.Tensor,
    state_r: torch.Tensor,
    state_i: torch.Tensor,
    block_m: int = 32,
):
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("blocked SPARC decode expects [B, 2M]")
    x = x.contiguous()
    batch, width = x.shape
    modes = width // 2
    out = torch.empty_like(x)
    last_r = torch.empty_like(state_r, dtype=torch.float32)
    last_i = torch.empty_like(state_i, dtype=torch.float32)
    block_d = triton.next_power_of_2(width)
    _sparc_decode_blocked[(triton.cdiv(modes, block_m), batch)](
        x, normalized_phase_direction, normalized_radial_direction,
        phase_scale, radial_scale, nu, cos_theta, sin_theta, gamma,
        state_r, state_i, out, last_r, last_i,
        width=width, modes=modes, BLOCK_D=block_d, BLOCK_M=block_m,
        num_warps=8 if block_d >= 256 else 4,
    )
    return out, (last_r, last_i)


def prepare_decode(module) -> None:
    module._inference_fused = True
    module._inference_pack = None
    return

    with torch.no_grad():
        nu = torch.exp(module.nu_log.float()).contiguous()
        theta = torch.exp(module.theta_log.float())
        scale = 1.0 / math.sqrt(module.modes)
        module._inference_pack = (
            F.normalize(module.phase_direction.float(), dim=0).contiguous(),
            F.normalize(module.radial_direction.float(), dim=0).contiguous(),
            (scale * torch.tanh(module.phase_amplitude.float())).reshape(1),
            (scale * torch.tanh(module.radial_amplitude.float())).reshape(1),
            nu,
            torch.cos(theta).contiguous(),
            torch.sin(theta).contiguous(),
            (torch.sqrt(1.0 - torch.exp(-2.0 * nu)) + 1.0e-8).contiguous(),
        )


def run_decode_step(module, x: torch.Tensor, state, segment_pos):
    del segment_pos
    if module._inference_fused:
        return latest_sparc_decode(module, x, state)
    if module._inference_fused:
        backend = module._decode_backend
        if backend == "auto":
            backend = "packed" if x.shape[0] == 16 else "fused"
        if backend == "packed":
            return sparc_decode_blocked(
                x, *module._inference_pack, state[0], state[1],
                block_m=1 << (module.modes - 1).bit_length(),
            )
        if backend.startswith("blocked"):
            return sparc_decode_blocked(
                x, *module._inference_pack, state[0], state[1],
                block_m=int(backend.removeprefix("blocked")),
            )
        return sparc_decode(
            x, module.phase_direction, module.radial_direction,
            module.phase_amplitude, module.radial_amplitude,
            module.nu_log, module.theta_log, state[0], state[1],
        )
    eta, delta, write_r, write_i = make_scan_inputs(module, x[:, None])
    nu, theta = torch.exp(module.nu_log), torch.exp(module.theta_log)
    radius = torch.exp(-nu * torch.exp(eta[:, 0]).unsqueeze(-1))
    phase = theta + (1.0 - radius) * delta[:, 0].unsqueeze(-1)
    ar, ai = radius * torch.cos(phase), radius * torch.sin(phase)
    write_r, write_i = write_r[:, 0].float(), write_i[:, 0].float()
    old_r, old_i = state
    new_r = ar * old_r - ai * old_i + write_r
    new_i = ai * old_r + ar * old_i + write_i
    output = torch.cat((new_r, new_i), -1).relu().to(x.dtype)
    return output, (new_r, new_i)

__all__ = ["prepare_decode", "run_decode_step"]
