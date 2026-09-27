from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class SparcParameters:
    wr: torch.Tensor
    wi: torch.Tensor
    nu: torch.Tensor
    theta: torch.Tensor
    phase_direction: torch.Tensor
    radial_direction: torch.Tensor
    phase_amplitude: torch.Tensor
    radial_amplitude: torch.Tensor


def make_sparc_parameters(d_model: int, modes: int, device: str, dtype: torch.dtype, seed: int = 1729) -> SparcParameters:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + d_model * 1009 + modes * 9176)
    def normal(*shape, scale=1.0, out_dtype=dtype):
        return (torch.randn(*shape, generator=generator) * scale).to(device=device, dtype=out_dtype)
    nu = torch.exp(torch.linspace(math.log(0.003), math.log(0.3), modes)).to(device=device, dtype=torch.float32)
    theta = torch.linspace(0.02, math.pi * 0.92, modes).to(device=device, dtype=torch.float32)
    phase_direction = normal(d_model + 1, scale=1.0, out_dtype=torch.float32)
    radial_direction = normal(d_model + 1, scale=1.0, out_dtype=torch.float32)
    phase_direction = phase_direction / phase_direction.norm().clamp_min(1e-12)
    radial_direction = radial_direction / radial_direction.norm().clamp_min(1e-12)
    return SparcParameters(
        wr=normal(d_model, modes, scale=1 / math.sqrt(2 * d_model)),
        wi=normal(d_model, modes, scale=1 / math.sqrt(2 * d_model)),
        nu=nu,
        theta=theta,
        phase_direction=phase_direction,
        radial_direction=radial_direction,
        phase_amplitude=torch.tensor(0.72, device=device, dtype=torch.float32),
        radial_amplitude=torch.tensor(0.58, device=device, dtype=torch.float32),
    )


def sparc_controller(u: torch.Tensor, p: SparcParameters) -> tuple[torch.Tensor, torch.Tensor]:
    uf = u.float()
    augmented = torch.cat((uf, torch.ones_like(uf[..., :1])), dim=-1)
    s_phase = torch.tanh(augmented @ p.phase_direction)
    s_radial = torch.tanh(augmented @ p.radial_direction)
    radial_coordinate = s_radial / (1.0 + s_radial.square())
    scale = 1.0 / math.sqrt(p.nu.numel())
    d = scale * p.phase_amplitude.tanh() * s_phase
    c = scale * p.radial_amplitude.tanh() * radial_coordinate
    return c, d


def sparc_write(u: torch.Tensor, p: SparcParameters) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = (torch.sqrt(1.0 - torch.exp(-2.0 * p.nu)) + 1.0e-8).to(u.dtype)
    wr = (u @ p.wr) * gamma
    wi = (u @ p.wi) * gamma
    return wr, wi


def _transition_direct(c: torch.Tensor, d: torch.Tensor, p: SparcParameters, dtype: torch.dtype):


    rho = torch.exp(-p.nu * torch.exp(c.float()).unsqueeze(-1))
    phase = p.theta + d.float().unsqueeze(-1)
    return rho.to(dtype), torch.cos(phase).to(dtype), torch.sin(phase).to(dtype)


def _transition_factorized(c: torch.Tensor, d: torch.Tensor, p: SparcParameters, dtype: torch.dtype):
    rho = torch.exp(-p.nu * torch.exp(c.float()).unsqueeze(-1)).to(dtype)
    cos_theta, sin_theta = torch.cos(p.theta), torch.sin(p.theta)
    cos_d, sin_d = torch.cos(d.float()).unsqueeze(-1), torch.sin(d.float()).unsqueeze(-1)
    cos_phase = (cos_theta * cos_d - sin_theta * sin_d).to(dtype)
    sin_phase = (sin_theta * cos_d + cos_theta * sin_d).to(dtype)
    return rho, cos_phase, sin_phase


def _apply(ar, ai, x, y, wr, wi):
    return ar * x - ai * y + wr, ai * x + ar * y + wi


def sparc_serial(u: torch.Tensor, p: SparcParameters, *, factorized: bool, initial=None) -> torch.Tensor:
    wr, wi = sparc_write(u, p)
    c, d = sparc_controller(u, p)
    transition = _transition_factorized if factorized else _transition_direct
    batch, length, modes = wr.shape
    if initial is None:
        x = torch.zeros(batch, modes, device=u.device, dtype=u.dtype)
        y = torch.zeros_like(x)
    else:
        x, y = initial

    output = []
    for index in range(length):
        rho, cp, sp = transition(c[:, index], d[:, index], p, u.dtype)
        ar, ai = rho * cp, rho * sp
        x, y = _apply(ar, ai, x, y, wr[:, index], wi[:, index])
        output.append(torch.stack((x, y), dim=-1))
    return torch.stack(output, dim=1)


def _affine_scan_real2(ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor, bi: torch.Tensor):
    length = ar.shape[-2]
    if length == 1:
        return ar, ai, br, bi

    lar, rar = ar[..., 0::2, :], ar[..., 1::2, :]
    lai, rai = ai[..., 0::2, :], ai[..., 1::2, :]
    lbr, rbr = br[..., 0::2, :], br[..., 1::2, :]
    lbi, rbi = bi[..., 0::2, :], bi[..., 1::2, :]
    pairs = rar.shape[-2]
    lar, lai, lbr, lbi = lar[..., :pairs, :], lai[..., :pairs, :], lbr[..., :pairs, :], lbi[..., :pairs, :]
    par = rar * lar - rai * lai
    pai = rar * lai + rai * lar
    pbr = rar * lbr - rai * lbi + rbr
    pbi = rar * lbi + rai * lbr + rbi
    oar, oai, obr, obi = _affine_scan_real2(par, pai, pbr, pbi)

    if ar[..., 0::2, :].shape[-2] > 1:
        prior_ar, prior_ai = oar[..., :-1, :], oai[..., :-1, :]
        prior_br, prior_bi = obr[..., :-1, :], obi[..., :-1, :]
        ear0, eai0 = ar[..., :1, :], ai[..., :1, :]
        ebr0, ebi0 = br[..., :1, :], bi[..., :1, :]
        next_ar, next_ai = ar[..., 2::2, :], ai[..., 2::2, :]
        next_br, next_bi = br[..., 2::2, :], bi[..., 2::2, :]
        tail_ar = next_ar * prior_ar - next_ai * prior_ai
        tail_ai = next_ar * prior_ai + next_ai * prior_ar
        tail_br = next_ar * prior_br - next_ai * prior_bi + next_br
        tail_bi = next_ar * prior_bi + next_ai * prior_br + next_bi
        ear, eai = torch.cat((ear0, tail_ar), dim=-2), torch.cat((eai0, tail_ai), dim=-2)
        ebr, ebi = torch.cat((ebr0, tail_br), dim=-2), torch.cat((ebi0, tail_bi), dim=-2)
    else:
        ear, eai, ebr, ebi = ar[..., 0::2, :], ai[..., 0::2, :], br[..., 0::2, :], bi[..., 0::2, :]

    inter_ar = torch.stack((ear[..., :pairs, :], oar), dim=-2).flatten(-3, -2)
    inter_ai = torch.stack((eai[..., :pairs, :], oai), dim=-2).flatten(-3, -2)
    inter_br = torch.stack((ebr[..., :pairs, :], obr), dim=-2).flatten(-3, -2)
    inter_bi = torch.stack((ebi[..., :pairs, :], obi), dim=-2).flatten(-3, -2)
    if length % 2:
        inter_ar, inter_ai = torch.cat((inter_ar, ear[..., -1:, :]), -2), torch.cat((inter_ai, eai[..., -1:, :]), -2)
        inter_br, inter_bi = torch.cat((inter_br, ebr[..., -1:, :]), -2), torch.cat((inter_bi, ebi[..., -1:, :]), -2)
    return inter_ar, inter_ai, inter_br, inter_bi


def sparc_tree_materialized(u: torch.Tensor, p: SparcParameters, *, factorized: bool = True) -> torch.Tensor:
    wr, wi = sparc_write(u, p)
    c, d = sparc_controller(u, p)
    transition = _transition_factorized if factorized else _transition_direct
    rho, cp, sp = transition(c, d, p, u.dtype)
    _, _, xr, xi = _affine_scan_real2(rho * cp, rho * sp, wr, wi)
    return torch.stack((xr, xi), dim=-1)


def sparc_chunked_compressed(u: torch.Tensor, p: SparcParameters, chunk_size: int) -> torch.Tensor:
    batch, length, _ = u.shape
    if length % chunk_size:
        raise ValueError("length must be divisible by chunk_size")
    wr, wi = sparc_write(u, p)
    c, d = sparc_controller(u, p)
    chunks = length // chunk_size
    wr = wr.view(batch, chunks, chunk_size, -1)
    wi = wi.view(batch, chunks, chunk_size, -1)
    c = c.view(batch, chunks, chunk_size)
    d = d.view(batch, chunks, chunk_size)

    x = torch.zeros(batch, chunks, p.nu.numel(), device=u.device, dtype=u.dtype)
    y = torch.zeros_like(x)
    local_x, local_y = [], []
    for offset in range(chunk_size):
        rho, cp, sp = _transition_factorized(c[:, :, offset], d[:, :, offset], p, u.dtype)
        x, y = _apply(rho * cp, rho * sp, x, y, wr[:, :, offset], wi[:, :, offset])
        local_x.append(x)
        local_y.append(y)
    local_x, local_y = torch.stack(local_x, 2), torch.stack(local_y, 2)

    G = torch.exp(c.float()).sum(dim=2)
    D = d.float().sum(dim=2)
    phase = chunk_size * p.theta + D.unsqueeze(-1)
    radius = torch.exp(-p.nu * G.unsqueeze(-1))
    car, cai = (radius * torch.cos(phase)).to(u.dtype), (radius * torch.sin(phase)).to(u.dtype)
    _, _, inclusive_qr, inclusive_qi = _affine_scan_real2(car, cai, local_x[:, :, -1], local_y[:, :, -1])
    zero = torch.zeros_like(inclusive_qr[:, :1])
    chunk_in_x = torch.cat((zero, inclusive_qr[:, :-1]), dim=1)
    chunk_in_y = torch.cat((zero, inclusive_qi[:, :-1]), dim=1)


    Gp = torch.exp(c.float()).cumsum(dim=2)
    Dp = d.float().cumsum(dim=2)
    steps = torch.arange(1, chunk_size + 1, device=u.device, dtype=torch.float32)
    phase_p = steps.view(1, 1, -1, 1) * p.theta + Dp.unsqueeze(-1)
    radius_p = torch.exp(-p.nu * Gp.unsqueeze(-1))
    par, pai = (radius_p * torch.cos(phase_p)).to(u.dtype), (radius_p * torch.sin(phase_p)).to(u.dtype)
    out_x = par * chunk_in_x.unsqueeze(2) - pai * chunk_in_y.unsqueeze(2) + local_x
    out_y = par * chunk_in_y.unsqueeze(2) + pai * chunk_in_x.unsqueeze(2) + local_y
    return torch.stack((out_x, out_y), dim=-1).reshape(batch, length, p.nu.numel(), 2)


def sparc_decode(u: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], p: SparcParameters, *, factorized: bool, precomputed=None):
    if precomputed is None:
        wr, wi = sparc_write(u, p)
        c, d = sparc_controller(u, p)
    else:
        wr, wi, c, d = precomputed
    transition = _transition_factorized if factorized else _transition_direct
    rho, cp, sp = transition(c, d, p, u.dtype)
    x, y = _apply(rho * cp, rho * sp, state[0], state[1], wr, wi)
    return x, y


def compose_complex_affine(left, right):
    left_lambda, left_write = left
    right_lambda, right_write = right
    return (
        right_lambda * left_lambda,
        right_lambda * left_write + right_write,
    )


def sparc_sequential_from_shared(
    e: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    nu: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    write: torch.Tensor,
    *,
    segment_pos: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    reset_first: bool = False,
):
    batch, length, modes = write.shape
    state = (
        torch.zeros(
            batch, modes, device=write.device, dtype=torch.complex64
        )
        if initial_state is None
        else initial_state.to(torch.complex64)
    )
    values = []
    for time in range(length):
        radius = torch.exp(-nu * e[:, time, None])
        transition = torch.complex(
            radius * (
                cos_theta * c[:, time, None]
                - sin_theta * s[:, time, None]
            ),
            radius * (
                sin_theta * c[:, time, None]
                + cos_theta * s[:, time, None]
            ),
        )
        if segment_pos is not None:
            reset = segment_pos[:, time].eq(0)[:, None]
        elif reset_first and time == 0:
            reset = torch.ones(
                batch, 1, device=write.device, dtype=torch.bool
            )
        else:
            reset = torch.zeros(
                batch, 1, device=write.device, dtype=torch.bool
            )
        transition = torch.where(
            reset, torch.zeros_like(transition), transition
        )
        state = transition * state + write[:, time]
        values.append(state)
    return torch.stack(values, dim=1), state


def ordered_chunk_summary(
    transitions: torch.Tensor, writes: torch.Tensor
):
    summary_lambda = torch.ones_like(transitions[..., 0, :])
    summary_write = torch.zeros_like(writes[..., 0, :])


    for time in range(transitions.shape[-2]):
        summary_lambda, summary_write = compose_complex_affine(
            (summary_lambda, summary_write),
            (transitions[..., time, :], writes[..., time, :]),
        )
    return summary_lambda, summary_write
