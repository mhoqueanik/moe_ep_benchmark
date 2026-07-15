"""Shared inputs, routing, and weight sources for MoE-EP microbenchmarks.

Every benchmark script derives backend-specific dtypes from the same per-rank
bf16 hidden states, top-k routing, and bf16 expert weights so comparisons are
not skewed by different random inputs.
"""

from __future__ import annotations

import dataclasses

import torch


HIDDEN_SEED_BASE = 7
ROUTING_SEED_BASE = 17
WEIGHT_SEED_BASE = 13
HIDDEN_SCALE = 10.0
WEIGHT_SCALE = 15.0


@dataclasses.dataclass(frozen=True)
class BenchProblem:
    """Per-rank problem slice shared across all backends."""

    hidden_states: torch.Tensor  # bf16 [m, hidden]
    topk_weights: torch.Tensor   # fp32 [m, top_k]
    topk_ids: torch.Tensor       # int64 [m, top_k]
    w13_bf16: torch.Tensor       # bf16 [num_local, 2*inter, hidden]
    w2_bf16: torch.Tensor        # bf16 [num_local, hidden, inter]


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def make_problem(
    rank: int,
    *,
    num_tokens: int,
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    device: torch.device | int,
) -> BenchProblem:
    hidden_g = torch.Generator(device=device).manual_seed(HIDDEN_SEED_BASE + rank)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=hidden_g,
        )
        / HIDDEN_SCALE
    )

    routing_g = torch.Generator(device=device).manual_seed(ROUTING_SEED_BASE + rank)
    scores = torch.randn(
        num_tokens,
        num_experts,
        dtype=torch.float32,
        device=device,
        generator=routing_g,
    )
    topk_weights, topk_ids = torch.topk(
        scores, top_k, dim=-1, largest=True, sorted=False
    )

    w13_bf16, w2_bf16 = make_expert_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
        device=device,
    )

    return BenchProblem(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.to(torch.int64),
        w13_bf16=w13_bf16,
        w2_bf16=w2_bf16,
    )


def make_expert_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-rank bf16 expert weights (same stream as make_problem).

    Any rank can regenerate any other rank's expert weights from the seed, so
    the accuracy reference can cover ALL experts locally without gathering the
    multi-GB weight set over NCCL.
    """
    weight_g = torch.Generator(device=device).manual_seed(WEIGHT_SEED_BASE + rank)
    w13_bf16 = (
        torch.randn(
            num_local_experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_g,
        )
        / WEIGHT_SCALE
    )
    w2_bf16 = (
        torch.randn(
            num_local_experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_g,
        )
        / WEIGHT_SCALE
    )
    return w13_bf16, w2_bf16


def compute_dense_moe_reference(
    problem: BenchProblem,
    *,
    world_size: int,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device | int,
    gate_up_clamp: float | None = None,
) -> torch.Tensor:
    """fp32 dense-MoE ground truth for this rank's tokens (all experts).

    Mega-backend convention: canonical ``w13 = [gate; up]``, activation
    ``silu(clamp(gate, max=c)) * clamp(up, -c, c)``, output
    ``Σ_k topk_weight · fc2(fc12(x))`` — the topk-weight application point
    (in-fc1 vs post-fc2) is equivalent in exact arithmetic, so one reference
    serves deep_gemm and both cutedsl dtypes.  The measured gap to this
    reference is the full quantization + kernel-arithmetic accuracy loss
    (weight quant, activation quant, fc1-out requant, combine wire).
    """
    x = problem.hidden_states.float()
    topk_ids = problem.topk_ids
    topk_w = problem.topk_weights.float()
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)

    for src_rank in range(world_size):
        w13, w2 = make_expert_weights(
            src_rank,
            num_local_experts=num_local_experts,
            hidden=hidden,
            intermediate=intermediate,
            device=device,
        )
        for local_e in range(num_local_experts):
            global_e = src_rank * num_local_experts + local_e
            routing_mask = topk_ids == global_e
            if not routing_mask.any():
                continue
            routed = routing_mask.nonzero(as_tuple=False)
            tokens, slots = routed[:, 0], routed[:, 1]

            g1 = x[tokens] @ w13[local_e].float().transpose(0, 1)  # (R, 2I)
            gate = g1[:, :intermediate]
            up = g1[:, intermediate:]
            if gate_up_clamp is not None:
                limit = abs(float(gate_up_clamp))
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = torch.nn.functional.silu(gate) * up
            g2 = act @ w2[local_e].float().transpose(0, 1)  # (R, hidden)
            out.index_put_(
                (tokens,),
                g2 * topk_w[tokens, slots].unsqueeze(-1),
                accumulate=True,
            )
        del w13, w2

    return out


def make_split_fp8_weights(
    w13_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    block_shape: list[int],
    use_ue8m0: bool,
):
    """vLLM split-path block-fp8 weights from shared bf16 experts."""
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    def q(w: torch.Tensor):
        qs, ss = [], []
        for e in range(w.size(0)):
            wq, ws = per_block_cast_to_fp8(w[e], block_shape, use_ue8m0=use_ue8m0)
            qs.append(wq)
            ss.append(ws)
        return torch.stack(qs), torch.stack(ss)

    w1, w1_s = q(w13_bf16)
    w2, w2_s = q(w2_bf16)
    return w1, w2, w1_s, w2_s


def make_mega_fp4_weights(w13_bf16: torch.Tensor, w2_bf16: torch.Tensor):
    """fp4 int8 weights + fp32 block-32 scales from shared bf16 experts."""
    from deep_gemm.utils import per_token_cast_to_fp4

    num_local = w13_bf16.size(0)
    hidden = w13_bf16.size(2)
    inter = w2_bf16.size(2)
    device = w13_bf16.device

    w13 = torch.empty(
        num_local, 2 * inter, hidden // 2, dtype=torch.int8, device=device
    )
    w2 = torch.empty(num_local, hidden, inter // 2, dtype=torch.int8, device=device)
    w13_sf = torch.empty(
        num_local, 2 * inter, hidden // 32, dtype=torch.float32, device=device
    )
    w2_sf = torch.empty(
        num_local, hidden, inter // 32, dtype=torch.float32, device=device
    )
    for e in range(num_local):
        q, sf = per_token_cast_to_fp4(w13_bf16[e], use_ue8m0=True, gran_k=32)
        w13[e].copy_(q)
        w13_sf[e].copy_(sf)
        q, sf = per_token_cast_to_fp4(w2_bf16[e], use_ue8m0=True, gran_k=32)
        w2[e].copy_(q)
        w2_sf[e].copy_(sf)
    return w13, w2, w13_sf, w2_sf


def fp32_sf_to_ue8m0_uint8(sf: torch.Tensor) -> torch.Tensor:
    """Pack fp32 block scales into DeepSeek/vLLM ue8m0 uint8 layout."""
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)
