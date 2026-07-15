"""Self-test for the accuracy-pass oracle (``compute_dense_moe_reference``).

The benchmark's ``acc_loss_pct`` is only as trustworthy as this reference, so
pin it two ways:

1. seed-stream contract: ``make_expert_weights(rank)`` must reproduce the
   exact weights ``make_problem(rank)`` hands the benchmarked layer (the
   accuracy pass regenerates PEER ranks' weights from seeds — if the streams
   ever diverge, the reference silently compares against wrong experts);
2. the vectorized expert-loop reference must equal an independently
   structured naive per-token/per-slot implementation, with and without the
   gate-up clamp.

Runs on CPU or GPU:  pytest tests/test_dense_reference.py
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_common import (  # noqa: E402
    compute_dense_moe_reference,
    make_expert_weights,
    make_problem,
)

WORLD = 2
NUM_LOCAL = 2
NUM_EXPERTS = WORLD * NUM_LOCAL
TOP_K = 2
TOKENS = 16
HIDDEN = 64
INTER = 32


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _problem(rank=0):
    return make_problem(
        rank,
        num_tokens=TOKENS,
        num_local_experts=NUM_LOCAL,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden=HIDDEN,
        intermediate=INTER,
        device=_device(),
    )


def _naive_reference(problem, *, gate_up_clamp):
    """Per-token/per-slot loop, structured unlike the vectorized reference."""
    device = _device()
    w13_all, w2_all = [], []
    for src in range(WORLD):
        w13, w2 = make_expert_weights(
            src,
            num_local_experts=NUM_LOCAL,
            hidden=HIDDEN,
            intermediate=INTER,
            device=device,
        )
        w13_all.append(w13.float())
        w2_all.append(w2.float())

    x = problem.hidden_states.float()
    out = torch.zeros(TOKENS, HIDDEN, dtype=torch.float32, device=device)
    for t in range(TOKENS):
        for k in range(TOP_K):
            e = int(problem.topk_ids[t, k])
            src, local = e // NUM_LOCAL, e % NUM_LOCAL
            g1 = x[t] @ w13_all[src][local].t()  # (2I,)
            gate, up = g1[:INTER], g1[INTER:]
            if gate_up_clamp is not None:
                limit = abs(float(gate_up_clamp))
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = torch.nn.functional.silu(gate) * up
            out[t] += (act @ w2_all[src][local].t()) * float(
                problem.topk_weights[t, k]
            )
    return out


def test_make_expert_weights_matches_make_problem_stream():
    for rank in range(WORLD):
        problem = _problem(rank)
        w13, w2 = make_expert_weights(
            rank,
            num_local_experts=NUM_LOCAL,
            hidden=HIDDEN,
            intermediate=INTER,
            device=_device(),
        )
        torch.testing.assert_close(w13, problem.w13_bf16, atol=0, rtol=0)
        torch.testing.assert_close(w2, problem.w2_bf16, atol=0, rtol=0)


@pytest.mark.parametrize("clamp", [None, 0.5])
def test_dense_reference_matches_naive(clamp):
    problem = _problem(0)
    ref = compute_dense_moe_reference(
        problem,
        world_size=WORLD,
        num_local_experts=NUM_LOCAL,
        hidden=HIDDEN,
        intermediate=INTER,
        device=_device(),
        gate_up_clamp=clamp,
    )
    naive = _naive_reference(problem, gate_up_clamp=clamp)
    torch.testing.assert_close(ref, naive, atol=1e-4, rtol=1e-4)


def test_clamp_actually_bites():
    """With a tiny clamp the outputs must differ from the unclamped ones,
    proving the clamp branch is exercised (bench-scale data never clamps)."""
    problem = _problem(0)
    kwargs = dict(
        world_size=WORLD,
        num_local_experts=NUM_LOCAL,
        hidden=HIDDEN,
        intermediate=INTER,
        device=_device(),
    )
    unclamped = compute_dense_moe_reference(problem, gate_up_clamp=None, **kwargs)
    clamped = compute_dense_moe_reference(problem, gate_up_clamp=0.05, **kwargs)
    assert not torch.allclose(unclamped, clamped)
