from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _complex_affine_combine(
    left_pr, left_pi, left_qr, left_qi,
    right_pr, right_pi, right_qr, right_qi,
):
    product_r = right_pr * left_pr - right_pi * left_pi
    product_i = right_pi * left_pr + right_pr * left_pi
    write_r = right_pr * left_qr - right_pi * left_qi + right_qr
    write_i = right_pi * left_qr + right_pr * left_qi + right_qi
    return product_r, product_i, write_r, write_i


@triton.jit
def _local_affine_summary(
    e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
    summary_pr, summary_pi, summary_qr, summary_qi,
    length: tl.constexpr, modes: tl.constexpr, chunks: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    STEPS: tl.constexpr, MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    step = tl.arange(0, STEPS)[:, None]
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)[None, :]
    token = chunk * STEPS + step
    valid_token = token < length
    valid_mode = lane < modes
    mask = valid_token & valid_mode
    token_offset = batch * length + token
    mode = lane
    shared_e = tl.load(e + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    shared_c = tl.load(c + token_offset, mask=valid_token, other=1.0).to(tl.float32)
    shared_s = tl.load(s + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    mode_nu = tl.load(nu + mode, mask=valid_mode, other=0.0).to(tl.float32)
    mode_c = tl.load(cos_theta + mode, mask=valid_mode, other=1.0).to(tl.float32)
    mode_s = tl.load(sin_theta + mode, mask=valid_mode, other=0.0).to(tl.float32)
    radius = tl.exp(-mode_nu * shared_e)
    transition_r = radius * (mode_c * shared_c - mode_s * shared_s)
    transition_i = radius * (mode_s * shared_c + mode_c * shared_s)
    if HAS_SEGMENTS:
        reset = valid_token & (
            tl.load(segment_pos + token_offset, mask=valid_token, other=1) == 0
        )
    elif RESET_FIRST:
        reset = valid_token & (token == 0)
    else:
        reset = False
    transition_r = tl.where(reset, 0.0, transition_r)
    transition_i = tl.where(reset, 0.0, transition_i)
    transition_r = tl.where(mask, transition_r, 1.0)
    transition_i = tl.where(mask, transition_i, 0.0)
    position = token_offset * modes + lane
    write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
    write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)
    prefix_pr, prefix_pi, prefix_qr, prefix_qi = tl.associative_scan(
        (transition_r, transition_i, write_r, write_i),
        axis=0,
        combine_fn=_complex_affine_combine,
    )
    last = step == STEPS - 1
    result_pr = tl.sum(tl.where(last, prefix_pr, 0.0), axis=0)
    result_pi = tl.sum(tl.where(last, prefix_pi, 0.0), axis=0)
    result_qr = tl.sum(tl.where(last, prefix_qr, 0.0), axis=0)
    result_qi = tl.sum(tl.where(last, prefix_qi, 0.0), axis=0)
    summary_offset = chunk_program * modes + (
        mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    )
    summary_mask = summary_offset - chunk_program * modes < modes
    tl.store(summary_pr + summary_offset, result_pr, mask=summary_mask)
    tl.store(summary_pi + summary_offset, result_pi, mask=summary_mask)
    tl.store(summary_qr + summary_offset, result_qr, mask=summary_mask)
    tl.store(summary_qi + summary_offset, result_qi, mask=summary_mask)


@triton.jit
def _serial_outer_prefix(
    summary_pr, summary_pi, summary_qr, summary_qi,
    h0_r, h0_i, boundary_r, boundary_i, last_r, last_i,
    modes: tl.constexpr, chunks: tl.constexpr, HAS_H0: tl.constexpr,
    MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    batch = tl.program_id(1)
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    mask = lane < modes
    if HAS_H0:
        state_r = tl.load(h0_r + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
        state_i = tl.load(h0_i + batch * modes + lane, mask=mask, other=0.0).to(tl.float32)
    else:
        state_r = tl.zeros((MODE_BLOCK,), tl.float32)
        state_i = tl.zeros((MODE_BLOCK,), tl.float32)

    for chunk in tl.range(0, chunks, 1, num_stages=1):
        offset = (batch * chunks + chunk) * modes + lane
        tl.store(boundary_r + offset, state_r, mask=mask)
        tl.store(boundary_i + offset, state_i, mask=mask)
        pr = tl.load(summary_pr + offset, mask=mask, other=1.0)
        pi = tl.load(summary_pi + offset, mask=mask, other=0.0)
        qr = tl.load(summary_qr + offset, mask=mask, other=0.0)
        qi = tl.load(summary_qi + offset, mask=mask, other=0.0)
        next_r = pr * state_r - pi * state_i + qr
        next_i = pi * state_r + pr * state_i + qi
        state_r, state_i = next_r, next_i
    tl.store(last_r + batch * modes + lane, state_r, mask=mask)
    tl.store(last_i + batch * modes + lane, state_i, mask=mask)


@triton.jit
def _local_affine_replay(
    e, c, s, nu, cos_theta, sin_theta, br, bi, segment_pos,
    boundary_r, boundary_i, out_r, out_i,
    length: tl.constexpr, modes: tl.constexpr, chunks: tl.constexpr,
    HAS_SEGMENTS: tl.constexpr, RESET_FIRST: tl.constexpr,
    STEPS: tl.constexpr, MODE_BLOCK: tl.constexpr,
):
    mode_block = tl.program_id(0)
    chunk_program = tl.program_id(1)
    batch = chunk_program // chunks
    chunk = chunk_program - batch * chunks
    step = tl.arange(0, STEPS)[:, None]
    lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)[None, :]
    token = chunk * STEPS + step
    valid_token = token < length
    valid_mode = lane < modes
    mask = valid_token & valid_mode
    token_offset = batch * length + token
    shared_e = tl.load(e + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    shared_c = tl.load(c + token_offset, mask=valid_token, other=1.0).to(tl.float32)
    shared_s = tl.load(s + token_offset, mask=valid_token, other=0.0).to(tl.float32)
    mode_nu = tl.load(nu + lane, mask=valid_mode, other=0.0).to(tl.float32)
    mode_c = tl.load(cos_theta + lane, mask=valid_mode, other=1.0).to(tl.float32)
    mode_s = tl.load(sin_theta + lane, mask=valid_mode, other=0.0).to(tl.float32)
    radius = tl.exp(-mode_nu * shared_e)
    transition_r = radius * (mode_c * shared_c - mode_s * shared_s)
    transition_i = radius * (mode_s * shared_c + mode_c * shared_s)
    if HAS_SEGMENTS:
        reset = valid_token & (
            tl.load(segment_pos + token_offset, mask=valid_token, other=1) == 0
        )
    elif RESET_FIRST:
        reset = valid_token & (token == 0)
    else:
        reset = False
    transition_r = tl.where(reset, 0.0, transition_r)
    transition_i = tl.where(reset, 0.0, transition_i)
    transition_r = tl.where(mask, transition_r, 1.0)
    transition_i = tl.where(mask, transition_i, 0.0)
    position = token_offset * modes + lane
    write_r = tl.load(br + position, mask=mask, other=0.0).to(tl.float32)
    write_i = tl.load(bi + position, mask=mask, other=0.0).to(tl.float32)

    prefix_pr, prefix_pi, prefix_qr, prefix_qi = tl.associative_scan(
        (transition_r, transition_i, write_r, write_i),
        axis=0,
        combine_fn=_complex_affine_combine,
    )
    summary_lane = mode_block * MODE_BLOCK + tl.arange(0, MODE_BLOCK)
    summary_mask = summary_lane < modes
    summary_offset = chunk_program * modes + summary_lane
    initial_r = tl.load(boundary_r + summary_offset, mask=summary_mask, other=0.0)
    initial_i = tl.load(boundary_i + summary_offset, mask=summary_mask, other=0.0)
    state_r = prefix_pr * initial_r[None, :] - prefix_pi * initial_i[None, :] + prefix_qr
    state_i = prefix_pi * initial_r[None, :] + prefix_pr * initial_i[None, :] + prefix_qi
    tl.store(out_r + position, state_r, mask=mask)
    tl.store(out_i + position, state_i, mask=mask)


@triton.jit
def _reduce_three_shared(
    partial_e, partial_c, partial_s, grad_e, grad_c, grad_s,
    rows: tl.constexpr, mode_blocks: tl.constexpr, REDUCE: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, REDUCE)
    mask = lane < mode_blocks
    offset = row * mode_blocks + lane
    tl.store(grad_e + row, tl.sum(tl.load(partial_e + offset, mask=mask, other=0.0), axis=0))
    tl.store(grad_c + row, tl.sum(tl.load(partial_c + offset, mask=mask, other=0.0), axis=0))
    tl.store(grad_s + row, tl.sum(tl.load(partial_s + offset, mask=mask, other=0.0), axis=0))




def sparc_affine_tile_scan(
    eta: torch.Tensor, delta: torch.Tensor,
    nu_log: torch.Tensor, theta_log: torch.Tensor,
    br: torch.Tensor, bi: torch.Tensor, *,
    segment_pos: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    steps: int = 16, mode_block: int = 32,
    num_warps: int = 4,
    reset_first: bool = False,
):
    from backward.affine import _SPARCAffineTileScan
    e = torch.exp(eta)
    c, s = torch.cos(delta), torch.sin(delta)
    nu = torch.exp(nu_log)
    theta = torch.exp(theta_log)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    empty_segment = eta.new_empty((0,), dtype=torch.int32)
    empty_state = br.new_empty((0,), dtype=torch.float32)
    if initial_state is None:
        h0_r = h0_i = empty_state
    else:
        h0_r, h0_i = initial_state
    return _SPARCAffineTileScan.apply(
        e, c, s, nu, cos_theta, sin_theta, br, bi,
        empty_segment if segment_pos is None else segment_pos,
        h0_r, h0_i, int(steps), int(mode_block), int(num_warps),
        bool(reset_first),
    )
