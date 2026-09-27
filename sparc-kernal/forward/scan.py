from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from common import SparcParameters
from .affine import sparc_affine_tile_scan
from .control import (
    formal_sparc_coordinates_forward_cache,
    make_controls,
    make_scan_inputs,
)


@dataclass
class PackedSparc:
    weight: torch.Tensor
    phase_bias: float
    radial_bias: float
    phase_scale: float
    radial_scale: float
    nu: torch.Tensor
    cos_theta: torch.Tensor
    sin_theta: torch.Tensor
    chunk_cos_theta: dict[int, torch.Tensor]
    chunk_sin_theta: dict[int, torch.Tensor]
    gamma: torch.Tensor
    padded_width: int
    bounded_poly_safe: bool
    use_bounded_poly: bool
    max_abs_c: float
    max_decay_exponent: float
    rho_taylor_error_bound: float
    rho_taylor_safe: bool
    use_rho_taylor: bool


def pack_sparc_parameters(p: SparcParameters, dtype: torch.dtype = torch.bfloat16, *,
                         enable_bounded_poly: bool = False,
                         enable_rho_taylor: bool = False) -> PackedSparc:
    modes = p.nu.numel()
    interleaved = torch.stack((p.wr, p.wi), dim=-1).reshape(p.wr.shape[0], 2 * modes)
    selectors = torch.stack((p.phase_direction[:-1], p.radial_direction[:-1]), dim=-1)
    useful = torch.cat((interleaved, selectors.to(interleaved.dtype)), dim=-1)
    padded = math.ceil(useful.shape[-1] / 16) * 16
    weight = torch.zeros(useful.shape[0], padded, device=useful.device, dtype=dtype)
    weight[:, : useful.shape[-1]] = useful.to(dtype)
    phase_scale = float(torch.tanh(p.phase_amplitude).item() / math.sqrt(modes))
    radial_scale = float(torch.tanh(p.radial_amplitude).item() / math.sqrt(modes))

    max_abs_c = abs(radial_scale) * 0.5
    min_nu = float(p.nu.min().item())
    max_decay_exponent = float(p.nu.max().item()) * math.exp(max_abs_c)
    bounded_poly_safe = min_nu >= 0.0 and max_abs_c <= 0.125 and max_decay_exponent <= 0.375
    use_bounded_poly = enable_bounded_poly and bounded_poly_safe
    if enable_bounded_poly and not bounded_poly_safe:
        raise ValueError(
            "bounded polynomial exp requested outside its certified interval: "
            f"max_abs_c={max_abs_c:.6g}, max_decay_exponent={max_decay_exponent:.6g}"
        )

    qmax = max_decay_exponent
    fifth_derivative_bound = (
        qmax + 15.0 * qmax**2 + 25.0 * qmax**3
        + 10.0 * qmax**4 + qmax**5
    )
    rho_taylor_error_bound = fifth_derivative_bound * max_abs_c**5 / math.factorial(5)
    rho_taylor_safe = min_nu >= 0.0 and rho_taylor_error_bound <= 1e-6
    use_rho_taylor = enable_rho_taylor and rho_taylor_safe
    if enable_rho_taylor and not rho_taylor_safe:
        raise ValueError(
            "rho Taylor path requested outside its certified error budget: "
            f"bound={rho_taylor_error_bound:.6g}"
        )
    return PackedSparc(
        weight=weight.contiguous(),
        phase_bias=float(p.phase_direction[-1].item()),
        radial_bias=float(p.radial_direction[-1].item()),
        phase_scale=phase_scale,
        radial_scale=radial_scale,
        nu=p.nu.float().contiguous(),
        cos_theta=torch.cos(p.theta).float().contiguous(),
        sin_theta=torch.sin(p.theta).float().contiguous(),
        chunk_cos_theta={size: torch.cos(size * p.theta).float().contiguous() for size in (8, 16, 32)},
        chunk_sin_theta={size: torch.sin(size * p.theta).float().contiguous() for size in (8, 16, 32)},
        gamma=(torch.sqrt(1.0 - torch.exp(-2.0 * p.nu)) + 1.0e-8).float().contiguous(),
        padded_width=padded,
        bounded_poly_safe=bounded_poly_safe,
        use_bounded_poly=use_bounded_poly,
        max_abs_c=max_abs_c,
        max_decay_exponent=max_decay_exponent,
        rho_taylor_error_bound=rho_taylor_error_bound,
        rho_taylor_safe=rho_taylor_safe,
        use_rho_taylor=use_rho_taylor,
    )


@triton.jit
def _control(raw_phase, raw_radial, phase_bias: tl.constexpr, radial_bias: tl.constexpr,
             phase_scale: tl.constexpr, radial_scale: tl.constexpr):
    sp = 2.0 * tl.sigmoid(2.0 * (raw_phase.to(tl.float32) + phase_bias)) - 1.0
    sr = 2.0 * tl.sigmoid(2.0 * (raw_radial.to(tl.float32) + radial_bias)) - 1.0
    d = phase_scale * sp
    c = radial_scale * sr / (1.0 + sr * sr)
    return c, d


@triton.jit
def _exp_small_symmetric(x):
    return 1.0 + x * (1.0 + x * (0.5 + x * (
        0.16666666666666666 + x * (0.041666666666666664 + x * (
            0.008333333333333333 + x * 0.001388888888888889)))))


@triton.jit
def _exp_small_negative(x):
    return 1.0 + x * (1.0 + x * (0.5 + x * (
        0.16666666666666666 + x * (0.041666666666666664 + x * (
            0.008333333333333333 + x * (0.001388888888888889 + x * (
                0.0001984126984126984 + x * 0.0000248015873015873)))))))


@triton.jit
def _rho_taylor_coefficients(nu, gamma):
    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu2 * nu2
    rho0 = tl.exp(-nu)
    return (
        rho0,
        -nu * rho0,
        (nu2 - nu) * rho0 * 0.5,
        (-nu + 3.0 * nu2 - nu3) * rho0 * (1.0 / 6.0),
        (-nu + 7.0 * nu2 - 6.0 * nu3 + nu4) * rho0 * (1.0 / 24.0),
    )


@triton.jit
def _transition(nu, cos_theta, sin_theta, c, d,
                rho0, rho1, rho2, rho3, rho4,
                USE_BOUNDED_POLY: tl.constexpr,
                USE_RHO_TAYLOR: tl.constexpr):
    if USE_RHO_TAYLOR:
        rho = rho0 + c * (rho1 + c * (rho2 + c * (rho3 + c * rho4)))
        g = 0.0
    elif USE_BOUNDED_POLY:
        g = _exp_small_symmetric(c)
        rho = _exp_small_negative(-nu * g)
    else:
        g = tl.exp(c)
        rho = tl.exp(-nu * g)
    d2 = d * d
    d4 = d2 * d2
    d6 = d4 * d2
    cd = 1.0 - 0.5 * d2 + d4 * (1.0 / 24.0) - d6 * (1.0 / 720.0)
    sd = d * (1.0 - d2 * (1.0 / 6.0) + d4 * (1.0 / 120.0) - d6 * (1.0 / 5040.0))
    cp = cos_theta * cd - sin_theta * sd
    sp = sin_theta * cd + cos_theta * sd
    return rho * cp, rho * sp, g


@triton.jit
def _serial_prefill_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, out,
                           last_r, last_i, length, modes: tl.constexpr,
                           packed_width: tl.constexpr, RETURN_CACHE: tl.constexpr,
                           phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                           phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                           USE_BOUNDED_POLY: tl.constexpr,
                           USE_RHO_TAYLOR: tl.constexpr,
                           BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    x = tl.zeros((BLOCK_M,), tl.float32)
    y = tl.zeros((BLOCK_M,), tl.float32)

    for t in tl.range(0, length, 1, num_stages=1):
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, _ = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx = ar * x - ai * y + wr
        ny = ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask)
        tl.store(out + o + 1, y, mask=mask)
    if RETURN_CACHE:
        tl.store(last_r + batch * modes + m, x, mask=mask)
        tl.store(last_i + batch * modes + m, y, mask=mask)


@triton.jit
def _chunk_summary_kernel(packed, nu_ptr, ct_ptr, st_ptr, chunk_ct_ptr,
                          chunk_st_ptr, gamma_ptr,
                          pr, pi, qr, qi, length, modes: tl.constexpr,
                          chunks, packed_width: tl.constexpr,
                          phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                          phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                          USE_BOUNDED_POLY: tl.constexpr,
                          USE_RHO_TAYLOR: tl.constexpr,
                          USE_COMPRESSED_P: tl.constexpr,
                          CHUNK: tl.constexpr, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    if USE_COMPRESSED_P:
        total_g = 0.0
        total_d = 0.0
    else:
        px, py = tl.full((BLOCK_M,), 1.0, tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    x, y = tl.zeros((BLOCK_M,), tl.float32), tl.zeros((BLOCK_M,), tl.float32)

    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, g = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        if USE_COMPRESSED_P:
            total_g += g
            total_d += d
        else:
            npx, npy = ar * px - ai * py, ai * px + ar * py
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        if not USE_COMPRESSED_P:
            px, py = npx, npy
        x, y = nx, ny
    if USE_COMPRESSED_P:

        base_c = tl.load(chunk_ct_ptr + m, mask=mask, other=1.0)
        base_s = tl.load(chunk_st_ptr + m, mask=mask, other=0.0)
        delta_c, delta_s = tl.cos(total_d), tl.sin(total_d)
        magnitude = tl.exp(-nu * total_g)
        px = magnitude * (base_c * delta_c - base_s * delta_s)
        py = magnitude * (base_s * delta_c + base_c * delta_s)
    o = (chunk_program * modes + m)
    tl.store(pr + o, px, mask=mask); tl.store(pi + o, py, mask=mask)
    tl.store(qr + o, x, mask=mask); tl.store(qi + o, y, mask=mask)


@triton.jit
def _chunk_prefix_kernel(pr, pi, qr, qi, in_r, in_i, modes: tl.constexpr,
                         chunks, last_r, last_i, RETURN_CACHE: tl.constexpr,
                         BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    x, y = tl.zeros((BLOCK_M,), tl.float32), tl.zeros((BLOCK_M,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        o = (batch * chunks + chunk) * modes + m
        tl.store(in_r + o, x, mask=mask); tl.store(in_i + o, y, mask=mask)
        ar = tl.load(pr + o, mask=mask, other=1.0); ai = tl.load(pi + o, mask=mask, other=0.0)
        br = tl.load(qr + o, mask=mask, other=0.0); bi = tl.load(qi + o, mask=mask, other=0.0)
        nx, ny = ar * x - ai * y + br, ai * x + ar * y + bi
        x, y = nx, ny
    if RETURN_CACHE:
        tl.store(last_r + batch * modes + m, x, mask=mask)
        tl.store(last_i + batch * modes + m, y, mask=mask)


@triton.jit
def _chunk_replay_kernel(packed, nu_ptr, ct_ptr, st_ptr, gamma_ptr, in_r, in_i, out,
                         length, modes: tl.constexpr, chunks,
                         packed_width: tl.constexpr, phase_bias: tl.constexpr,
                         radial_bias: tl.constexpr, phase_scale: tl.constexpr,
                         radial_scale: tl.constexpr,
                         USE_BOUNDED_POLY: tl.constexpr, CHUNK: tl.constexpr,
                         USE_RHO_TAYLOR: tl.constexpr,
                         BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    state_o = chunk_program * modes + m
    x = tl.load(in_r + state_o, mask=mask, other=0.0)
    y = tl.load(in_i + state_o, mask=mask, other=0.0)
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mask, other=1.0), tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        row = (batch * length + t) * packed_width
        raw_phase = tl.load(packed + row + 2 * modes)
        raw_radial = tl.load(packed + row + 2 * modes + 1)
        c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias, phase_scale, radial_scale)
        ar, ai, _ = _transition(
            nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
            USE_BOUNDED_POLY, USE_RHO_TAYLOR,
        )
        wr = tl.load(packed + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
        wi = tl.load(packed + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
        nx, ny = ar * x - ai * y + wr, ai * x + ar * y + wi
        x, y = nx, ny
        o = ((batch * length + t) * modes + m) * 2
        tl.store(out + o, x, mask=mask); tl.store(out + o + 1, y, mask=mask)


@triton.jit
def _decode_kernel(u, wr_ptr, wi_ptr, phase_dir, radial_dir, state_r, state_i,
                   nu_ptr, ct_ptr, st_ptr, gamma_ptr,
                   out_r, out_i,
                   d_model: tl.constexpr, modes: tl.constexpr,
                   phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                   phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                   USE_BOUNDED_POLY: tl.constexpr,
                   USE_RHO_TAYLOR: tl.constexpr,
                   BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    mmask, dmask = m < modes, d < d_model
    uv = tl.load(u + batch * d_model + d, mask=dmask, other=0.0).to(tl.float32)
    phase = tl.sum(uv * tl.load(phase_dir + d, mask=dmask, other=0.0), axis=0)
    radial = tl.sum(uv * tl.load(radial_dir + d, mask=dmask, other=0.0), axis=0)
    c, delta = _control(phase, radial, phase_bias, radial_bias, phase_scale, radial_scale)
    weights_o = d[:, None] * modes + m[None, :]
    wr = tl.sum(uv[:, None] * tl.load(wr_ptr + weights_o, mask=dmask[:, None] & mmask[None, :], other=0.0).to(tl.float32), axis=0)
    wi = tl.sum(uv[:, None] * tl.load(wi_ptr + weights_o, mask=dmask[:, None] & mmask[None, :], other=0.0).to(tl.float32), axis=0)
    nu = tl.load(nu_ptr + m, mask=mmask, other=0.0)
    ct, st = tl.load(ct_ptr + m, mask=mmask, other=1.0), tl.load(st_ptr + m, mask=mmask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mmask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    ar, ai, _ = _transition(
        nu, ct, st, c, delta, rho0, rho1, rho2, rho3, rho4,
        USE_BOUNDED_POLY, USE_RHO_TAYLOR,
    )
    x = tl.load(state_r + batch * modes + m, mask=mmask, other=0.0)
    y = tl.load(state_i + batch * modes + m, mask=mmask, other=0.0)
    nx, ny = ar * x - ai * y + wr * gamma, ai * x + ar * y + wi * gamma
    tl.store(out_r + batch * modes + m, nx, mask=mmask)
    tl.store(out_i + batch * modes + m, ny, mask=mmask)


@triton.jit
def _decode_projected_kernel(projected, nu_ptr, ct_ptr, st_ptr, gamma_ptr,
                             state_r, state_i, out, last_r, last_i,
                             modes: tl.constexpr, packed_width: tl.constexpr,
                             phase_bias: tl.constexpr, radial_bias: tl.constexpr,
                             phase_scale: tl.constexpr, radial_scale: tl.constexpr,
                             USE_BOUNDED_POLY: tl.constexpr,
                             USE_RHO_TAYLOR: tl.constexpr,
                             BLOCK_M: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m < modes
    row = batch * packed_width
    raw_phase = tl.load(projected + row + 2 * modes)
    raw_radial = tl.load(projected + row + 2 * modes + 1)
    c, d = _control(raw_phase, raw_radial, phase_bias, radial_bias,
                    phase_scale, radial_scale)
    nu = tl.load(nu_ptr + m, mask=mask, other=0.0)
    ct = tl.load(ct_ptr + m, mask=mask, other=1.0)
    st = tl.load(st_ptr + m, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + m, mask=mask, other=0.0)
    if USE_RHO_TAYLOR:
        rho0, rho1, rho2, rho3, rho4 = _rho_taylor_coefficients(nu, gamma)
    else:
        rho0 = 0.0
        rho1 = 0.0
        rho2 = 0.0
        rho3 = 0.0
        rho4 = 0.0
    ar, ai, _ = _transition(
        nu, ct, st, c, d, rho0, rho1, rho2, rho3, rho4,
        USE_BOUNDED_POLY, USE_RHO_TAYLOR,
    )
    x = tl.load(state_r + batch * modes + m, mask=mask, other=0.0)
    y = tl.load(state_i + batch * modes + m, mask=mask, other=0.0)
    wr = tl.load(projected + row + 2 * m, mask=mask, other=0.0).to(tl.float32) * gamma
    wi = tl.load(projected + row + 2 * m + 1, mask=mask, other=0.0).to(tl.float32) * gamma
    nx = ar * x - ai * y + wr
    ny = ai * x + ar * y + wi
    output = (batch * modes + m) * 2
    tl.store(out + output, nx, mask=mask)
    tl.store(out + output + 1, ny, mask=mask)
    tl.store(last_r + batch * modes + m, nx, mask=mask)
    tl.store(last_i + batch * modes + m, ny, mask=mask)


def _launch_meta(packed: PackedSparc, bounded_poly: bool | None = None,
                 rho_taylor: bool | None = None):
    use_bounded_poly = packed.use_bounded_poly if bounded_poly is None else bounded_poly
    use_rho_taylor = packed.use_rho_taylor if rho_taylor is None else rho_taylor
    if use_bounded_poly and not packed.bounded_poly_safe:
        raise ValueError("bounded polynomial exp requested outside its certified interval")
    if use_rho_taylor and not packed.rho_taylor_safe:
        raise ValueError("rho Taylor path requested outside its certified error budget")
    if use_bounded_poly and use_rho_taylor:
        raise ValueError("select either the generic bounded-exp path or rho Taylor, not both")
    return dict(
        phase_bias=packed.phase_bias, radial_bias=packed.radial_bias,
        phase_scale=packed.phase_scale, radial_scale=packed.radial_scale,
        USE_BOUNDED_POLY=use_bounded_poly,
        USE_RHO_TAYLOR=use_rho_taylor,
    )


def sparc_triton_serial(u: torch.Tensor, p: SparcParameters,
                       packed: PackedSparc | None = None, *,
                       num_warps: int | None = None,
                       return_cache: bool = False):
    packed = packed or pack_sparc_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    last_i = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    block = min(128, triton.next_power_of_2(modes))
    launch_warps = num_warps or (4 if block >= 64 else 2)
    _serial_prefill_kernel[(triton.cdiv(modes, block), batch)](
        projected, packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        out, last_r, last_i, length, modes=modes,
        packed_width=packed.padded_width, RETURN_CACHE=return_cache, BLOCK_M=block,
        num_warps=launch_warps, **_launch_meta(packed),
    )
    return (out, (last_r, last_i)) if return_cache else out


def sparc_triton_chunked(u: torch.Tensor, p: SparcParameters, chunk_size: int,
                        packed: PackedSparc | None = None, *,
                        num_warps: int | None = None,
                        compressed_p: bool = False,
                        bounded_poly: bool | None = None,
                        rho_taylor: bool | None = None,
                        return_cache: bool = False):
    packed = packed or pack_sparc_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    if length % chunk_size:
        raise ValueError("length must be divisible by chunk_size")
    chunks = length // chunk_size
    projected = u @ packed.weight
    shape = (batch, chunks, modes)
    pr, pi, qr, qi = [torch.empty(shape, device=u.device, dtype=torch.float32) for _ in range(4)]
    in_r, in_i = [torch.empty(shape, device=u.device, dtype=torch.float32) for _ in range(2)]
    out = torch.empty(batch, length, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    last_i = torch.empty(batch, modes, device=u.device, dtype=torch.float32) if return_cache else packed.nu
    block = min(128, triton.next_power_of_2(modes)); grid = (triton.cdiv(modes, block), batch * chunks)
    launch_warps = num_warps or (2 if block <= 64 else 4)
    effective_bounded_poly = packed.use_bounded_poly if bounded_poly is None else bounded_poly
    effective_rho_taylor = packed.use_rho_taylor if rho_taylor is None else rho_taylor
    use_compressed_p = (
        compressed_p and chunk_size in packed.chunk_cos_theta
        and not effective_bounded_poly and not effective_rho_taylor
    )
    chunk_ct = packed.chunk_cos_theta[chunk_size] if use_compressed_p else packed.cos_theta
    chunk_st = packed.chunk_sin_theta[chunk_size] if use_compressed_p else packed.sin_theta
    meta = dict(length=length, modes=modes, chunks=chunks, packed_width=packed.padded_width,
                CHUNK=chunk_size, BLOCK_M=block, USE_COMPRESSED_P=use_compressed_p,
                num_warps=launch_warps,
                **_launch_meta(packed, effective_bounded_poly, effective_rho_taylor))

    _chunk_summary_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                                chunk_ct, chunk_st, packed.gamma,
                                pr, pi, qr, qi, **meta)
    _chunk_prefix_kernel[(triton.cdiv(modes, block), batch)](
        pr, pi, qr, qi, in_r, in_i, modes=modes, chunks=chunks,
        last_r=last_r, last_i=last_i, RETURN_CACHE=return_cache,
        BLOCK_M=block, num_warps=launch_warps,
    )
    _chunk_replay_kernel[grid](projected, packed.nu, packed.cos_theta, packed.sin_theta,
                               packed.gamma, in_r, in_i, out,
                               **{key: value for key, value in meta.items() if key != "USE_COMPRESSED_P"})
    return (out, (last_r, last_i)) if return_cache else out


def sparc_triton_auto(u: torch.Tensor, p: SparcParameters,
                     packed: PackedSparc | None = None, *,
                     compressed_p: bool = False,
                     return_cache: bool = False):
    length = u.shape[1]
    if length <= 256:
        return sparc_triton_chunked(
            u, p, 8, packed, num_warps=4, compressed_p=compressed_p,
            bounded_poly=packed.bounded_poly_safe if length <= 128 else False,
            return_cache=return_cache,
        )
    return sparc_triton_chunked(
        u, p, 16 if length <= 512 else 32, packed,
        num_warps=2,
        compressed_p=compressed_p,
        return_cache=return_cache,
    )


def sparc_triton_decode(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                       p: SparcParameters, packed: PackedSparc | None = None, *,
                       block_m: int | None = None, num_warps: int | None = None):
    packed = packed or pack_sparc_parameters(p, u.dtype)
    batch, d_model = u.shape; modes = p.nu.numel()
    out_r, out_i = torch.empty_like(state[0]), torch.empty_like(state[1])
    block_d = triton.next_power_of_2(d_model)
    if block_m is None:
        block_m = 64 if batch > 16 else min(32, triton.next_power_of_2(modes))
    if num_warps is None:
        num_warps = 8
    _decode_kernel[(triton.cdiv(modes, block_m), batch)](
        u, p.wr, p.wi, p.phase_direction, p.radial_direction, state[0], state[1],
        packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        out_r, out_i,
        d_model=d_model, modes=modes, BLOCK_D=block_d, BLOCK_M=block_m,
        num_warps=num_warps, **_launch_meta(packed),
    )
    return out_r, out_i


def sparc_triton_decode_split(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor],
                             packed: PackedSparc, *, block_m: int | None = None,
                             num_warps: int | None = None):
    if u.ndim != 2:
        raise ValueError("split decode expects [B, D]")
    batch, _ = u.shape
    modes = packed.nu.numel()
    if state[0].shape != (batch, modes) or state[1].shape != (batch, modes):
        raise ValueError("split decode state shape mismatch")
    projected = u @ packed.weight
    out = torch.empty(batch, modes, 2, device=u.device, dtype=u.dtype)
    last_r = torch.empty_like(state[0], dtype=torch.float32)
    last_i = torch.empty_like(state[1], dtype=torch.float32)
    block_m = block_m or (64 if batch <= 16 else min(128, triton.next_power_of_2(modes)))
    num_warps = num_warps or 4
    _decode_projected_kernel[(triton.cdiv(modes, block_m), batch)](
        projected, packed.nu, packed.cos_theta, packed.sin_theta, packed.gamma,
        state[0], state[1], out, last_r, last_i,
        modes=modes, packed_width=packed.padded_width, BLOCK_M=block_m,
        num_warps=num_warps, **_launch_meta(packed),
    )
    return out.reshape(batch, 2 * modes), (last_r, last_i)


def sparc_packed_reference(u: torch.Tensor, p: SparcParameters,
                          packed: PackedSparc | None = None, *, initial=None,
                          return_cache: bool = False):
    packed = packed or pack_sparc_parameters(p, u.dtype)
    batch, length, _ = u.shape; modes = p.nu.numel()
    projected = u @ packed.weight
    raw_phase, raw_radial = projected[..., 2*modes].float(), projected[..., 2*modes+1].float()
    sp = torch.tanh(raw_phase + packed.phase_bias)
    sr = torch.tanh(raw_radial + packed.radial_bias)
    d = packed.phase_scale * sp; c = packed.radial_scale * sr / (1 + sr.square())
    if initial is None:
        x = torch.zeros(batch, modes, device=u.device, dtype=torch.float32)
        y = torch.zeros_like(x)
    else:
        x, y = initial[0].float(), initial[1].float()
    outputs=[]
    for t in range(length):
        rho = torch.exp(-packed.nu * torch.exp(c[:, t]).unsqueeze(-1))
        cd, sd = torch.cos(d[:, t]).unsqueeze(-1), torch.sin(d[:, t]).unsqueeze(-1)
        cp = packed.cos_theta * cd - packed.sin_theta * sd; si = packed.sin_theta * cd + packed.cos_theta * sd
        wr = projected[:, t, 0:2*modes:2].float() * packed.gamma
        wi = projected[:, t, 1:2*modes:2].float() * packed.gamma
        nx = rho*cp*x - rho*si*y + wr; ny = rho*si*x + rho*cp*y + wi
        x,y=nx,ny; outputs.append(torch.stack((x,y),-1).to(u.dtype))
    output = torch.stack(outputs, 1)
    return (output, (x, y)) if return_cache else output


@triton.jit
def _real_forward(a, b, out, last, length: tl.constexpr, width: tl.constexpr,
                  BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    state = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        offset = (batch * length + t) * width + lane
        at = tl.load(a + offset, mask=mask, other=0.0).to(tl.float32)
        bt = tl.load(b + offset, mask=mask, other=0.0).to(tl.float32)
        state = at * state + bt
        tl.store(out + offset, state, mask=mask)
    tl.store(last + batch * width + lane, state, mask=mask)

@triton.jit
def _real_chunk_summary(a, b, p, q,
                        length: tl.constexpr, width: tl.constexpr,
                        chunks: tl.constexpr, CHUNK: tl.constexpr,
                        BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    product = tl.full((BLOCK,), 1.0, tl.float32)
    state = tl.zeros((BLOCK,), tl.float32)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        position = (batch * length + t) * width + lane
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        write = tl.load(b + position, mask=mask, other=0.0).to(tl.float32)
        product = transition * product
        state = transition * state + write
    summary = chunk_program * width + lane
    tl.store(p + summary, product, mask=mask)
    tl.store(q + summary, state, mask=mask)


@triton.jit
def _real_chunk_prefix(p, q, chunk_in,
                       width: tl.constexpr, chunks: tl.constexpr,
                       BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    state = tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        position = (batch * chunks + chunk) * width + lane
        tl.store(chunk_in + position, state, mask=mask)
        product = tl.load(p + position, mask=mask, other=1.0)
        write = tl.load(q + position, mask=mask, other=0.0)
        state = product * state + write


@triton.jit
def _real_chunk_replay(a, b, chunk_in, out,
                       length: tl.constexpr, width: tl.constexpr,
                       chunks: tl.constexpr, CHUNK: tl.constexpr,
                       BLOCK: tl.constexpr):
    width_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = width_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < width
    summary = chunk_program * width + lane
    state = tl.load(chunk_in + summary, mask=mask, other=0.0)
    for local_t in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + local_t
        position = (batch * length + t) * width + lane
        transition = tl.load(a + position, mask=mask, other=1.0).to(tl.float32)
        write = tl.load(b + position, mask=mask, other=0.0).to(tl.float32)
        state = transition * state + write
        tl.store(out + position, state, mask=mask)

@triton.jit
def _complex_forward(ar, ai, br, bi, out_r, out_i, last_r, last_i,
                     length: tl.constexpr, modes: tl.constexpr,
                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        offset = (batch * length + t) * modes + lane
        atr = tl.load(ar + offset, mask=mask, other=0.0).to(tl.float32)
        ati = tl.load(ai + offset, mask=mask, other=0.0).to(tl.float32)
        btr = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        bti = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = atr * xr - ati * xi + btr
        ni = ati * xr + atr * xi + bti
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)

@triton.jit
def _sparc_forward(eta, delta, nu_log, theta_log, br, bi,
                  out_r, out_i, last_r, last_i,
                  length: tl.constexpr, modes: tl.constexpr,
                  BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))

    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        rho = tl.exp(-nu * tl.exp(radial))
        modal_delta = (1.0 - rho) * phase_delta
        cos_delta, sin_delta = tl.cos(modal_delta), tl.sin(modal_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = rho * cosine, rho * sine
        offset = token_offset * modes + lane
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = ar * xr - ai * xi + write_r
        ni = ai * xr + ar * xi + write_i
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)


@triton.jit
def _sparc_precompute_decay(eta, nu_log, rho,
                           rows: tl.constexpr, modes: tl.constexpr,
                           BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    row = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (lane < modes)
    radial = tl.load(eta + row, mask=row < rows, other=0.0).to(tl.float32)
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    value = tl.exp(-nu * tl.exp(radial))
    tl.store(rho + row * modes + lane, value, mask=mask)


@triton.jit
def _sparc_forward_precomputed(rho, delta, theta_log, br, bi,
                              out_r, out_i, last_r, last_i,
                              length: tl.constexpr, modes: tl.constexpr,
                              BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < modes
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    xr = tl.zeros((BLOCK,), tl.float32)
    xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        offset = token_offset * modes + lane
        radius = tl.load(rho + offset, mask=mask, other=0.0)
        modal_delta = (1.0 - radius) * phase_delta
        cos_delta, sin_delta = tl.cos(modal_delta), tl.sin(modal_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        nr = ar * xr - ai * xi + write_r
        ni = ai * xr + ar * xi + write_i
        xr, xi = nr, ni
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)

@triton.jit
def _sparc_tiled_serial_forward(
    eta, delta, nu_log, theta_log, br, bi, segment_pos, h0_r, h0_i,
    out_r, out_i, last_r, last_i,
    length: tl.constexpr, modes: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    HAS_H0: tl.constexpr, BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if HAS_H0:
        xr = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
        xi = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        xr = tl.zeros((BLOCK,), tl.float32)
        xi = tl.zeros((BLOCK,), tl.float32)
    for t in tl.range(0, length, 1, num_stages=1):
        token_offset = batch * length + t
        radial = tl.load(eta + token_offset).to(tl.float32)
        phase_delta = tl.load(delta + token_offset).to(tl.float32)
        radius = tl.exp(-nu * tl.exp(radial))
        modal_delta = (1.0 - radius) * phase_delta
        cos_delta, sin_delta = tl.cos(modal_delta), tl.sin(modal_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        if HAS_SEGMENTS:
            reset = tl.load(segment_pos + token_offset) == 0
        elif RESET_FIRST:
            reset = t == 0
        else:
            reset = False
        ar = tl.where(reset, 0.0, ar)
        ai = tl.where(reset, 0.0, ai)
        offset = token_offset * modes + lane
        write_r = tl.load(br + offset, mask=mask, other=0.0).to(tl.float32)
        write_i = tl.load(bi + offset, mask=mask, other=0.0).to(tl.float32)
        next_r = ar * xr - ai * xi + write_r
        next_i = ai * xr + ar * xi + write_i
        xr, xi = next_r, next_i
        tl.store(out_r + offset, xr, mask=mask)
        tl.store(out_i + offset, xi, mask=mask)
    tl.store(last_r + batch * modes + lane, xr, mask=mask)
    tl.store(last_i + batch * modes + lane, xi, mask=mask)

@triton.jit
def _sparc_precompute_shared_controls(eta, delta, exp_eta, cos_delta,
                                     sin_delta, rows: tl.constexpr,
                                     BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offset = block * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows
    radial = tl.load(eta + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(exp_eta + offset, tl.exp(radial), mask=mask)


@triton.jit
def _sparc_precompute_static_spectrum(
    nu_log, theta_log, nu, theta, cos_theta, sin_theta,
    modes: tl.constexpr, BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    lane = program * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    nu_value = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
    theta_value = tl.exp(
        tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32)
    )
    tl.store(nu + lane, nu_value, mask=mask)
    tl.store(theta + lane, theta_value, mask=mask)
    tl.store(cos_theta + lane, tl.cos(theta_value), mask=mask)
    tl.store(sin_theta + lane, tl.sin(theta_value), mask=mask)


@triton.jit
def _sparc_chunk_summary(eta, delta, nu_log, theta_log, br, bi, raw_x,
                        write_gamma,
                        exp_eta_shared, cos_delta_shared, sin_delta_shared,
                        spectrum_nu, spectrum_theta,
                        spectrum_cos, spectrum_sin,
                        compressed_g, compressed_d,
                        pr, pi, qr, qi,
                        length: tl.constexpr, modes: tl.constexpr,
                        chunks: tl.constexpr, CHUNK: tl.constexpr,
                        PRECOMPUTED_SHARED: tl.constexpr,
                        PRECOMPUTED_SPECTRAL: tl.constexpr,
                        COMPRESSED_TRANSITION: tl.constexpr,
                        FUSED_WRITE: tl.constexpr,
                        ROUND_WRITE_BF16: tl.constexpr,
                        BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:

        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
    px = tl.full((BLOCK,), 1.0, tl.float32)
    py = tl.zeros((BLOCK,), tl.float32)
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    g_sum = 0.0
    d_sum = 0.0
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
        if COMPRESSED_TRANSITION:

            if PRECOMPUTED_SHARED:
                phase_delta = tl.load(delta + token_offset).to(tl.float32)
            g_sum += exp_radial
            d_sum += phase_delta
        radius = tl.exp(-nu * exp_radial)
        modal_delta = (1.0 - radius) * phase_delta
        cos_delta, sin_delta = tl.cos(modal_delta), tl.sin(modal_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        if FUSED_WRITE:
            raw_position = token_offset * (2 * modes) + lane
            raw_r = tl.load(raw_x + raw_position, mask=mask, other=0.0).to(tl.float32)
            raw_i = tl.load(
                raw_x + raw_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            write_r = raw_r * gamma
            write_i = raw_i * gamma

            if ROUND_WRITE_BF16:
                write_r = write_r.to(tl.bfloat16).to(tl.float32)
                write_i = write_i.to(tl.bfloat16).to(tl.float32)
        else:
            write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
            write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        if not COMPRESSED_TRANSITION:
            next_px = ar * px - ai * py
            next_py = ai * px + ar * py
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        if not COMPRESSED_TRANSITION:
            px, py = next_px, next_py
        x, y = next_x, next_y
    summary_offset = chunk_program * modes + lane
    if COMPRESSED_TRANSITION:
        first_mode_block = mode_block == 0
        tl.store(compressed_g + chunk_program, g_sum, mask=first_mode_block)
        tl.store(compressed_d + chunk_program, d_sum, mask=first_mode_block)
    else:
        tl.store(pr + summary_offset, px, mask=mask)
        tl.store(pi + summary_offset, py, mask=mask)
    tl.store(qr + summary_offset, x, mask=mask)
    tl.store(qi + summary_offset, y, mask=mask)


@triton.jit
def _sparc_chunk_prefix(pr, pi, qr, qi, chunk_in_r, chunk_in_i,
                       modes: tl.constexpr, chunks: tl.constexpr,
                       BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(chunk_in_r + offset, x, mask=mask)
        tl.store(chunk_in_i + offset, y, mask=mask)
        ar = tl.load(pr + offset, mask=mask, other=1.0)
        ai = tl.load(pi + offset, mask=mask, other=0.0)
        write_r = tl.load(qr + offset, mask=mask, other=0.0)
        write_i = tl.load(qi + offset, mask=mask, other=0.0)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y


@triton.jit
def _sparc_compressed_chunk_prefix(
    compressed_g, compressed_d, nu_log, theta_log, qr, qi,
    spectrum_nu, spectrum_theta,
    boundary_r, boundary_i,
    modes: tl.constexpr, chunks: tl.constexpr, CHUNK: tl.constexpr,
    REVERSE: tl.constexpr, PRECOMPUTED_SPECTRAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    for logical_chunk in tl.range(0, chunks, 1, num_stages=1):
        if REVERSE:
            chunk = chunks - 1 - logical_chunk
        else:
            chunk = logical_chunk
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(boundary_r + offset, x, mask=mask)
        tl.store(boundary_i + offset, y, mask=mask)
        scalar_offset = batch * chunks + chunk
        g_value = tl.load(compressed_g + scalar_offset).to(tl.float32)
        d_value = tl.load(compressed_d + scalar_offset).to(tl.float32)
        radius = tl.exp(-nu * g_value)
        phase = theta * CHUNK + d_value
        ar = radius * tl.cos(phase)
        ai = radius * tl.sin(phase)
        if REVERSE:
            ai = -ai
        write_r = tl.load(qr + offset, mask=mask, other=0.0)
        write_i = tl.load(qi + offset, mask=mask, other=0.0)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y


@triton.jit
def _sparc_reconstruct_chunk_transition(
    compressed_g, compressed_d, nu_log, theta_log, pr, pi,
    spectrum_nu, spectrum_theta,
    modes: tl.constexpr, chunks: tl.constexpr, CHUNK: tl.constexpr,
    REVERSE: tl.constexpr, PRECOMPUTED_SPECTRAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
    g_value = tl.load(compressed_g + chunk_program).to(tl.float32)
    d_value = tl.load(compressed_d + chunk_program).to(tl.float32)
    radius = tl.exp(-nu * g_value)
    phase = theta * CHUNK + d_value
    real = radius * tl.cos(phase)
    imaginary = radius * tl.sin(phase)
    if REVERSE:
        imaginary = -imaginary
    offset = chunk_program * modes + lane
    tl.store(pr + offset, real, mask=mask)
    tl.store(pi + offset, imaginary, mask=mask)


@triton.jit
def _complex_affine_prefix_stage(
    input_pr, input_pi, input_qr, input_qi,
    output_pr, output_pi, output_qr, output_qi,
    modes: tl.constexpr, chunks: tl.constexpr,
    STRIDE: tl.constexpr, REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    offset = (batch * chunks + chunk) * modes + lane
    cpr = tl.load(input_pr + offset, mask=mask, other=1.0)
    cpi = tl.load(input_pi + offset, mask=mask, other=0.0)
    cqr = tl.load(input_qr + offset, mask=mask, other=0.0)
    cqi = tl.load(input_qi + offset, mask=mask, other=0.0)
    if REVERSE:
        has_previous = chunk + STRIDE < chunks
        previous_chunk = chunk + STRIDE
    else:
        has_previous = chunk >= STRIDE
        previous_chunk = chunk - STRIDE
    previous_offset = (batch * chunks + previous_chunk) * modes + lane
    ppr = tl.load(
        input_pr + previous_offset, mask=mask & has_previous, other=1.0
    )
    ppi = tl.load(
        input_pi + previous_offset, mask=mask & has_previous, other=0.0
    )
    pqr = tl.load(
        input_qr + previous_offset, mask=mask & has_previous, other=0.0
    )
    pqi = tl.load(
        input_qi + previous_offset, mask=mask & has_previous, other=0.0
    )
    result_pr = cpr * ppr - cpi * ppi
    result_pi = cpi * ppr + cpr * ppi
    result_qr = cpr * pqr - cpi * pqi + cqr
    result_qi = cpi * pqr + cpr * pqi + cqi
    tl.store(output_pr + offset, result_pr, mask=mask)
    tl.store(output_pi + offset, result_pi, mask=mask)
    tl.store(output_qr + offset, result_qr, mask=mask)
    tl.store(output_qi + offset, result_qi, mask=mask)


@triton.jit
def _complex_affine_prefix_extract(prefix_qr, prefix_qi,
                                   boundary_r, boundary_i,
                                   modes: tl.constexpr,
                                   chunks: tl.constexpr,
                                   REVERSE: tl.constexpr,
                                   BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    if REVERSE:
        first = chunk == chunks - 1
        previous_chunk = chunk + 1
    else:
        first = chunk == 0
        previous_chunk = chunk - 1
    previous_offset = (batch * chunks + previous_chunk) * modes + lane
    state_r = tl.load(
        prefix_qr + previous_offset, mask=mask & ~first, other=0.0
    )
    state_i = tl.load(
        prefix_qi + previous_offset, mask=mask & ~first, other=0.0
    )
    offset = (batch * chunks + chunk) * modes + lane
    tl.store(boundary_r + offset, state_r, mask=mask)
    tl.store(boundary_i + offset, state_i, mask=mask)


@triton.jit
def _complex_affine_group_local(
    chunk_pr, chunk_pi, chunk_qr, chunk_qi,
    group_pr, group_pi, group_qr, group_qi,
    modes: tl.constexpr, chunks: tl.constexpr, groups: tl.constexpr,
    GROUP: tl.constexpr, REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    group_program = tl.program_id(1)
    batch = group_program // groups
    group = group_program - batch * groups
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    accumulated_pr = tl.full((BLOCK,), 1.0, tl.float32)
    accumulated_pi = tl.zeros((BLOCK,), tl.float32)
    accumulated_qr = tl.zeros((BLOCK,), tl.float32)
    accumulated_qi = tl.zeros((BLOCK,), tl.float32)

    for logical_chunk in tl.range(0, GROUP, 1, num_stages=1):
        if REVERSE:
            local_chunk = GROUP - 1 - logical_chunk
        else:
            local_chunk = logical_chunk
        chunk = group * GROUP + local_chunk
        valid = chunk < chunks
        offset = (batch * chunks + chunk) * modes + lane
        current_pr = tl.load(
            chunk_pr + offset, mask=mask & valid, other=1.0
        )
        current_pi = tl.load(
            chunk_pi + offset, mask=mask & valid, other=0.0
        )
        current_qr = tl.load(
            chunk_qr + offset, mask=mask & valid, other=0.0
        )
        current_qi = tl.load(
            chunk_qi + offset, mask=mask & valid, other=0.0
        )
        tl.store(chunk_pr + offset, accumulated_pr, mask=mask & valid)
        tl.store(chunk_pi + offset, accumulated_pi, mask=mask & valid)
        tl.store(chunk_qr + offset, accumulated_qr, mask=mask & valid)
        tl.store(chunk_qi + offset, accumulated_qi, mask=mask & valid)
        next_pr = current_pr * accumulated_pr - current_pi * accumulated_pi
        next_pi = current_pi * accumulated_pr + current_pr * accumulated_pi
        next_qr = current_pr * accumulated_qr - current_pi * accumulated_qi + current_qr
        next_qi = current_pi * accumulated_qr + current_pr * accumulated_qi + current_qi
        accumulated_pr = tl.where(valid, next_pr, accumulated_pr)
        accumulated_pi = tl.where(valid, next_pi, accumulated_pi)
        accumulated_qr = tl.where(valid, next_qr, accumulated_qr)
        accumulated_qi = tl.where(valid, next_qi, accumulated_qi)
    group_offset = group_program * modes + lane
    tl.store(group_pr + group_offset, accumulated_pr, mask=mask)
    tl.store(group_pi + group_offset, accumulated_pi, mask=mask)
    tl.store(group_qr + group_offset, accumulated_qr, mask=mask)
    tl.store(group_qi + group_offset, accumulated_qi, mask=mask)


@triton.jit
def _complex_affine_group_outer(
    group_pr, group_pi, group_qr, group_qi,
    group_boundary_r, group_boundary_i,
    modes: tl.constexpr, groups: tl.constexpr,
    REVERSE: tl.constexpr, BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    state_r = tl.zeros((BLOCK,), tl.float32)
    state_i = tl.zeros((BLOCK,), tl.float32)
    for logical_group in tl.range(0, groups, 1, num_stages=1):
        if REVERSE:
            group = groups - 1 - logical_group
        else:
            group = logical_group
        offset = (batch * groups + group) * modes + lane
        tl.store(group_boundary_r + offset, state_r, mask=mask)
        tl.store(group_boundary_i + offset, state_i, mask=mask)
        transform_r = tl.load(group_pr + offset, mask=mask, other=1.0)
        transform_i = tl.load(group_pi + offset, mask=mask, other=0.0)
        write_r = tl.load(group_qr + offset, mask=mask, other=0.0)
        write_i = tl.load(group_qi + offset, mask=mask, other=0.0)
        next_r = transform_r * state_r - transform_i * state_i + write_r
        next_i = transform_i * state_r + transform_r * state_i + write_i
        state_r, state_i = next_r, next_i


@triton.jit
def _complex_affine_group_correct(
    local_pr, local_pi, local_qr, local_qi,
    group_boundary_r, group_boundary_i, boundary_r, boundary_i,
    modes: tl.constexpr, chunks: tl.constexpr,
    groups: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    group = chunk // GROUP
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    local_offset = chunk_program * modes + lane
    group_offset = (batch * groups + group) * modes + lane
    transform_r = tl.load(local_pr + local_offset, mask=mask, other=1.0)
    transform_i = tl.load(local_pi + local_offset, mask=mask, other=0.0)
    write_r = tl.load(local_qr + local_offset, mask=mask, other=0.0)
    write_i = tl.load(local_qi + local_offset, mask=mask, other=0.0)
    outer_r = tl.load(group_boundary_r + group_offset, mask=mask, other=0.0)
    outer_i = tl.load(group_boundary_i + group_offset, mask=mask, other=0.0)
    result_r = transform_r * outer_r - transform_i * outer_i + write_r
    result_i = transform_i * outer_r + transform_r * outer_i + write_i
    tl.store(boundary_r + local_offset, result_r, mask=mask)
    tl.store(boundary_i + local_offset, result_i, mask=mask)


@triton.jit
def _sparc_chunk_replay(eta, delta, nu_log, theta_log, br, bi, raw_x,
                       write_gamma,
                       exp_eta_shared, cos_delta_shared, sin_delta_shared,
                       spectrum_nu, spectrum_theta,
                       spectrum_cos, spectrum_sin,
                       chunk_in_r, chunk_in_i, out_r, out_i, visible_out,
                       length: tl.constexpr, modes: tl.constexpr,
                       chunks: tl.constexpr, CHUNK: tl.constexpr,
                       PRECOMPUTED_SHARED: tl.constexpr,
                       PRECOMPUTED_SPECTRAL: tl.constexpr,
                       COMPRESSED_TRANSITION: tl.constexpr,
                       FUSED_WRITE: tl.constexpr,
                       ROUND_WRITE_BF16: tl.constexpr,
                       FUSED_OUTPUT_RELU: tl.constexpr,
                       BLOCK: tl.constexpr):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    lane = mode_block * BLOCK + tl.arange(0, BLOCK)
    mask = lane < modes
    summary_offset = chunk_program * modes + lane
    x = tl.load(chunk_in_r + summary_offset, mask=mask, other=0.0)
    y = tl.load(chunk_in_i + summary_offset, mask=mask, other=0.0)
    if PRECOMPUTED_SPECTRAL:
        nu = tl.load(spectrum_nu + lane, mask=mask, other=0.0)
        theta = tl.load(spectrum_theta + lane, mask=mask, other=0.0)
        cos_theta = tl.load(spectrum_cos + lane, mask=mask, other=1.0)
        sin_theta = tl.load(spectrum_sin + lane, mask=mask, other=0.0)
    else:
        nu = tl.exp(tl.load(nu_log + lane, mask=mask, other=0.0).to(tl.float32))
        theta = tl.exp(tl.load(theta_log + lane, mask=mask, other=0.0).to(tl.float32))
        cos_theta, sin_theta = tl.cos(theta), tl.sin(theta)
    if FUSED_WRITE:
        gamma = tl.load(write_gamma + lane, mask=mask, other=0.0).to(tl.float32)
    for offset in tl.static_range(0, CHUNK):
        t = chunk * CHUNK + offset
        token_offset = batch * length + t
        if PRECOMPUTED_SHARED:
            exp_radial = tl.load(exp_eta_shared + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
        else:
            radial = tl.load(eta + token_offset).to(tl.float32)
            phase_delta = tl.load(delta + token_offset).to(tl.float32)
            exp_radial = tl.exp(radial)
        radius = tl.exp(-nu * exp_radial)
        modal_delta = (1.0 - radius) * phase_delta
        cos_delta, sin_delta = tl.cos(modal_delta), tl.sin(modal_delta)
        cosine = cos_theta * cos_delta - sin_theta * sin_delta
        sine = sin_theta * cos_delta + cos_theta * sin_delta
        ar, ai = radius * cosine, radius * sine
        position = token_offset * modes + lane
        if FUSED_WRITE:
            raw_position = token_offset * (2 * modes) + lane
            raw_r = tl.load(raw_x + raw_position, mask=mask, other=0.0).to(tl.float32)
            raw_i = tl.load(
                raw_x + raw_position + modes, mask=mask, other=0.0
            ).to(tl.float32)
            write_r = raw_r * gamma
            write_i = raw_i * gamma
            if ROUND_WRITE_BF16:
                write_r = write_r.to(tl.bfloat16).to(tl.float32)
                write_i = write_i.to(tl.bfloat16).to(tl.float32)
        else:
            write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
            write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
        next_x = ar * x - ai * y + write_r
        next_y = ai * x + ar * y + write_i
        x, y = next_x, next_y
        tl.store(out_r + position, x, mask=mask)
        tl.store(out_i + position, y, mask=mask)
        if FUSED_OUTPUT_RELU:
            visible_position = token_offset * (2 * modes) + lane
            tl.store(visible_out + visible_position, tl.maximum(x, 0.0), mask=mask)
            tl.store(
                visible_out + visible_position + modes,
                tl.maximum(y, 0.0), mask=mask,
            )
def _eager_complex_scan(ar, ai, br, bi):
    xr = torch.zeros_like(br[:, 0], dtype=torch.float32)
    xi = torch.zeros_like(bi[:, 0], dtype=torch.float32)
    real, imag = [], []


    for time in range(ar.shape[1]):
        next_r = (
            ar[:, time].float() * xr
            - ai[:, time].float() * xi
            + br[:, time].float()
        )
        next_i = (
            ai[:, time].float() * xr
            + ar[:, time].float() * xi
            + bi[:, time].float()
        )
        xr, xi = next_r, next_i
        real.append(xr.to(br.dtype))
        imag.append(xi.to(bi.dtype))
    return torch.stack(real, 1), torch.stack(imag, 1)


def _associative_complex_scan(ar, ai, br, bi):
    par, pai, pbr, pbi = ar, ai, br, bi
    offset = 1
    while offset < ar.shape[1]:
        right_ar, right_ai = par[:, offset:], pai[:, offset:]
        left_ar, left_ai = par[:, :-offset], pai[:, :-offset]
        left_br, left_bi = pbr[:, :-offset], pbi[:, :-offset]
        right_br, right_bi = pbr[:, offset:], pbi[:, offset:]
        combined_ar = right_ar * left_ar - right_ai * left_ai
        combined_ai = right_ar * left_ai + right_ai * left_ar
        combined_br = right_ar * left_br - right_ai * left_bi + right_br
        combined_bi = right_ar * left_bi + right_ai * left_br + right_bi
        par = torch.cat((par[:, :offset], combined_ar), dim=1)
        pai = torch.cat((pai[:, :offset], combined_ai), dim=1)
        pbr = torch.cat((pbr[:, :offset], combined_br), dim=1)
        pbi = torch.cat((pbi[:, :offset], combined_bi), dim=1)
        offset *= 2
    return pbr, pbi


def run_sequence(module, x: torch.Tensor, return_cache: bool = False):
    from backward.scan import (
        complex_scan,
        sparc_scan,
        sparc_scan_precomputed_decay,
        sparc_scan_with_last,
        sparc_tiled_serial_scan,
        sparc_training_scan_dispatch,
    )
    backend = module.scan_backend
    if backend == "auto":
        batch, length, width = x.shape
        if x.dtype == torch.bfloat16 and not return_cache:
            if batch >= 4 and length <= 2048:
                backend = "fused_output_shared_sfu_chunk32"
            elif batch == 1 and length >= 65536:
                backend = (
                    "fused_output_shared_sfu_"
                    "serial_forward_prefix_grouped_prefix64_"
                    "hierarchical_chunk32"
                )
            elif batch == 1 and (
                length >= 16384 or (length >= 8192 and width >= 2048)
            ):
                backend = (
                    "fused_output_shared_sfu_"
                    "grouped_prefix64_hierarchical_chunk32"
                )
            elif batch == 1 and length >= 4096:
                backend = "fused_output_shared_sfu_chunk32"
            else:
                backend = "chunk16" if length >= 1024 else "triton"
        else:
            backend = "chunk16" if length >= 1024 else "triton"
    backend = backend.replace("_fused_write", "").replace("compressed_", "")
    if backend.startswith("affine_tile_s"):
        backend = "chunk16"
    fused_write = False
    fused_controller_backward = (
        "fused_controller_bwd" in backend and fused_write
    )
    controller_backward_cache = None
    packed_output = None
    if fused_write:
        if fused_controller_backward:

            with torch.no_grad():
                directions = F.normalize(
                    torch.stack((
                        module.phase_direction,
                        module.radial_direction,
                    )).float(),
                    dim=1,
                )
                projected = F.linear(
                    x.float(), directions[:, :-1], directions[:, -1]
                )
                eta, delta, phase_coordinate, radial_raw = (
                    formal_sparc_coordinates_forward_cache(
                        projected,
                        module.phase_amplitude,
                        module.radial_amplitude,
                        module.modes,
                    )
                )
            controller_backward_cache = (
                module.phase_direction, module.radial_direction,
                module.phase_amplitude, module.radial_amplitude,
                phase_coordinate, radial_raw,
            )
        else:
            eta, delta = make_controls(module, x)
        write_r = write_i = x.new_empty((0,))
    else:
        eta, delta, write_r, write_i = make_scan_inputs(module, x)
    if return_cache:
        if module.modes <= 512:
            out_r, out_i, last_r, last_i = sparc_scan_with_last(
                eta, delta, module.nu_log, module.theta_log, write_r, write_i
            )
        else:
            out_r, out_i, last_r, last_i = sparc_tiled_serial_scan(
                eta, delta, module.nu_log, module.theta_log, write_r, write_i,
                block_size=module._tiled_training_block_size,
            )
        cache = (last_r, last_i)
    elif backend.startswith("affine_tile_s"):
        configuration = backend.removeprefix("affine_tile_s")
        steps_text, mode_and_warps = configuration.split("_m", 1)
        mode_text, warps_text = mode_and_warps.split("_w", 1)
        out_r, out_i, _, _ = sparc_affine_tile_scan(
            eta, delta, module.nu_log, module.theta_log, write_r, write_i,
            steps=int(steps_text), mode_block=int(mode_text),
            num_warps=int(warps_text),
        )
        cache = None

    elif backend == "tiled_serial":
        out_r, out_i, _, _ = sparc_tiled_serial_scan(
            eta, delta, module.nu_log, module.theta_log, write_r, write_i,
            block_size=module._tiled_training_block_size,
        )
        cache = None
    elif backend == "triton":
        if module.modes <= 512:
            out_r, out_i = sparc_scan(
                eta, delta, module.nu_log, module.theta_log, write_r, write_i
            )
            cache = None
        else:
            nu = torch.exp(module.nu_log)
            theta = torch.exp(module.theta_log)
            radius = torch.exp(-nu * torch.exp(eta).unsqueeze(-1))
            phase = theta + (1.0 - radius) * delta.unsqueeze(-1)
            out_r, out_i = complex_scan(
                radius * torch.cos(phase),
                radius * torch.sin(phase),
                write_r,
                write_i,
            )
            cache = None

    elif backend == "precomputed_decay":
        out_r, out_i = sparc_scan_precomputed_decay(
            eta, delta, module.nu_log, module.theta_log, write_r, write_i
        )
        cache = None
    elif "chunk" in backend and backend.rsplit("chunk", 1)[1].isdigit():
        hierarchical_prefix = "hierarchical" in backend
        precompute_shared = "shared_sfu" in backend
        compressed_transition = "compressed" in backend
        atomic_shared = "atomic" in backend
        two_stage_spectral = "spectral2" in backend
        precompute_spectral = "spectral_cache" in backend
        backward_chunk_group = 2 if "replayg2" in backend else 1
        fused_output_relu = "fused_output" in backend
        prefix_group_size = next(
            (
                size for size in (32, 64, 128)
                if f"grouped_prefix{size}" in backend
            ),
            0,
        )
        if "serial_forward_prefix" in backend and prefix_group_size:
            prefix_group_size = -prefix_group_size
        compact_control_cache = (
            "cache" in module._controller_projection_dtype
        )
        chunk_size = int(backend.rsplit("chunk", 1)[1])
        scan_output = sparc_training_scan_dispatch(
            eta, delta, module.nu_log, module.theta_log,
            write_r, write_i, chunk_size=chunk_size,
            hierarchical_prefix=hierarchical_prefix,
            precompute_shared=precompute_shared,
            compressed_transition=compressed_transition,
            atomic_shared=atomic_shared,
            two_stage_spectral=two_stage_spectral,
            compact_control_cache=compact_control_cache,
            precompute_spectral=precompute_spectral,
            raw_x=x if fused_write else None,
            fused_write=fused_write,
            backward_chunk_group=backward_chunk_group,
            fused_output_relu=fused_output_relu,
            prefix_group_size=prefix_group_size,
            controller_backward_cache=controller_backward_cache,
            reset_first=True,
            block_size=module._tiled_training_block_size,
        )
        if fused_output_relu:
            packed_output = scan_output
        else:
            out_r, out_i = scan_output
        cache = None
    else:
        nu = torch.exp(module.nu_log)
        theta = torch.exp(module.theta_log)
        radius = torch.exp(-nu * torch.exp(eta).unsqueeze(-1))
        phase = theta + (1.0 - radius) * delta.unsqueeze(-1)
        ar, ai = radius * torch.cos(phase), radius * torch.sin(phase)
        if backend == "framework_eager":
            out_r, out_i = _eager_complex_scan(
                ar, ai, write_r, write_i
            )
        elif backend in ("associative_bf16", "associative_fp32"):
            scan_dtype = (
                torch.bfloat16
                if backend == "associative_bf16"
                else torch.float32
            )
            out_r, out_i = _associative_complex_scan(
                ar.to(scan_dtype), ai.to(scan_dtype),
                write_r.to(scan_dtype), write_i.to(scan_dtype),
            )
            out_r, out_i = out_r.to(x.dtype), out_i.to(x.dtype)
        else:
            raise ValueError(f"unknown scan backend: {backend}")
        cache = (
            (out_r[:, -1].float(), out_i[:, -1].float())
            if return_cache else None
        )
    output = (
        packed_output
        if packed_output is not None
        else torch.cat((out_r, out_i), dim=-1).relu()
    )
    return output, cache
