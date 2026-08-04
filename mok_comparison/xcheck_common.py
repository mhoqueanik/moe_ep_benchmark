"""Deterministic same-input protocol for the fi moe_ep vs MoK cross-check.

Both sides call :func:`xcheck_tensors` with the same (rank, world, geometry)
and get bit-identical inputs: tokens, routing, per-rank local expert weights,
and replicated shared-expert weights. Generation is seeded per rank / per
tensor on the CUDA generator, so identical torch builds on identical GPUs
(the same container image) reproduce the same values.

Conventions (match both MoK and the fi mega bench):
  - expert e is owned by rank e // (experts // world), experts contiguous
  - topk_weights = softmax over the top-k router logits, fp32
  - weights bf16, scaled by fan-in**-0.5 like MoK's tests/utils
  - fi w13 = cat([gate, up], dim=1)  (canonical [gate; up])
"""

import torch


def xcheck_tensors(rank, world, tokens, hidden, inter, experts, topk, device):
    num_local = experts // world
    g = torch.Generator(device=device).manual_seed(11234 + rank)

    router_logits = torch.randn(tokens, experts, generator=g, device=device)
    topk_vals, topk_ids = torch.topk(router_logits, topk, dim=1)
    topk_w = torch.softmax(topk_vals.float(), dim=-1)
    x = torch.randn(tokens, hidden, generator=g, device=device, dtype=torch.bfloat16)

    gs = torch.Generator(device=device).manual_seed(777)
    w_shared_gate = torch.randn(inter, hidden, generator=gs, device=device, dtype=torch.bfloat16) * hidden**-0.5
    w_shared_up = torch.randn(inter, hidden, generator=gs, device=device, dtype=torch.bfloat16) * hidden**-0.5
    w_shared_down = torch.randn(hidden, inter, generator=gs, device=device, dtype=torch.bfloat16) * inter**-0.5

    # Per-global-expert seeds so any rank can regenerate any expert's weights.
    gates, ups, downs = [], [], []
    for j in range(num_local):
        ge = rank * num_local + j
        ge_gen = torch.Generator(device=device).manual_seed(90000 + ge)
        gates.append(torch.randn(inter, hidden, generator=ge_gen, device=device, dtype=torch.bfloat16) * hidden**-0.5)
        ups.append(torch.randn(inter, hidden, generator=ge_gen, device=device, dtype=torch.bfloat16) * hidden**-0.5)
        downs.append(torch.randn(hidden, inter, generator=ge_gen, device=device, dtype=torch.bfloat16) * inter**-0.5)

    return {
        "x": x,
        "topk_ids": topk_ids.to(torch.int64),
        "topk_w": topk_w,
        "w_gate": torch.stack(gates),   # [E_local, I, H]
        "w_up": torch.stack(ups),       # [E_local, I, H]
        "w_down": torch.stack(downs),   # [E_local, H, I]
        "w_shared_gate": w_shared_gate,  # [I, H]
        "w_shared_up": w_shared_up,      # [I, H]
        "w_shared_down": w_shared_down,  # [H, I]
    }
