"""Offline DP-attention bench: DP{N}/TP1 + EP, one LLM process per DP rank.

vLLM 0.25.1 offline data parallel requires the explicit multi-process form
(examples/features/data_parallel/data_parallel_offline.py): each DP rank is a
separate process with VLLM_DP_* env and its own LLM(); MoE layers form an
EP group of size DP*TP across the ranks. This removes the per-layer TP
allreduce entirely — attention is data-parallel, cross-rank traffic collapses
into the mega kernel's own dispatch/combine.

Timing mirrors bench_offline.py: fixed random-token prompts (seed 0) sharded
evenly across ranks, 1 warmup + N timed rounds in lockstep (mp.Barrier), wall
time measured on rank 0 between barriers, total tok/s = all-rank tokens /
round wall. JSON output schema matches bench_offline.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time

DEFAULT_MODEL = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/"
    "deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig"
)


def _rank_main(
    rank: int,
    args,
    ilen: int,
    olen: int,
    barrier,
    result_q,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_port)
    # Do NOT set CUDA_VISIBLE_DEVICES: vLLM's DP mode derives the device from
    # the DP-adjusted local rank and asserts device_count > local_rank.

    import random

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        moe_backend="deep_gemm_mega_moe",
        max_model_len=4096,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype="fp8",
        block_size=256,
        enable_prefix_caching=False,
    )

    # Same fixed prompt set as bench_offline.py (seed 0), sharded by rank so
    # every backend/topology sees identical total work.
    random.seed(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [
        {"prompt_token_ids": [random.randrange(100, vocab - 100) for _ in range(ilen)]}
        for _ in range(args.num_prompts)
    ]
    shard = prompts[rank :: args.dp]
    sp = SamplingParams(temperature=0.0, max_tokens=olen, ignore_eos=True)

    walls = []
    for _ in range(args.rounds + 1):  # +1 warmup, discarded by rank 0
        barrier.wait()
        t0 = time.perf_counter()
        llm.generate(shard, sp, use_tqdm=False)
        barrier.wait()
        walls.append(time.perf_counter() - t0)
    if rank == 0:
        result_q.put(walls)
    barrier.wait()  # keep all ranks alive until results are read


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workload", required=True, help="name:input_len:output_len")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--dp", type=int, default=int(os.environ.get("DP", "4")))
    ap.add_argument(
        "--dp-port",
        type=int,
        default=0,
        help="0 = derive a fresh port per invocation (avoids stale listeners)",
    )
    ap.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("MAX_BATCHED_TOKENS", "4096")),
    )
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("ENFORCE_EAGER", "1") == "1",
    )
    args = ap.parse_args()

    name, ilen, olen = args.workload.split(":")
    ilen, olen = int(ilen), int(olen)
    if args.dp_port == 0:
        args.dp_port = 20000 + (os.getpid() % 20000)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.dp)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_rank_main, args=(r, args, ilen, olen, barrier, result_q)
        )
        for r in range(args.dp)
    ]
    for p in procs:
        p.start()
    import queue as _queue

    walls = None
    while walls is None:
        try:
            walls = result_q.get(timeout=15)
        except _queue.Empty:
            dead = {p.pid: p.exitcode for p in procs if p.exitcode not in (None, 0)}
            if dead:
                for p in procs:
                    p.terminate()
                raise SystemExit(f"DP rank process(es) died during startup: {dead}")
    for p in procs:
        p.join()
    if any(p.exitcode not in (0, None) for p in procs):
        raise SystemExit(f"DP rank failed: {[p.exitcode for p in procs]}")

    total_tokens = args.num_prompts * (ilen + olen)
    rounds = [
        dict(
            round=i,
            warmup=(i == 0),
            seconds=w,
            total_tok_per_s=total_tokens / w,
        )
        for i, w in enumerate(walls)
    ]
    timed = sorted(r["total_tok_per_s"] for r in rounds if not r["warmup"])
    median = timed[len(timed) // 2]
    out = dict(
        tag=args.tag,
        workload=f"{name}",
        dp=args.dp,
        tp=1,
        rounds=rounds,
        median_total_tok_per_s=median,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"bench_offline_dp {args.tag} {name} dp{args.dp}: "
        f"median {median:.1f} tok/s over {len(timed)} rounds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
