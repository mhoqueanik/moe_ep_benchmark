"""Within-engine repeated-round throughput bench.

One engine boot per backend config (env-selected, like smoke_infer.py), then
N timed rounds of the same fixed workload — repeats without engine-restart
variance (prefill cells showed +-35% across restarts with `vllm bench
throughput`, which boots a fresh engine per run).

    FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega \
    python bench_offline.py --tag fi_dg --workload prefill:1024:1 \
        --rounds 5 --out results/offline_fi_dg_prefill.json
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
# NVFP4 cast of the same base weights (see cast_mxfp4_to_nvfp4.log in the
# checkpoint dir) — the format the nvfp4_cutedsl prequantized-weights path
# consumes without dequant->requant.
DEFAULT_MODEL_NVFP4 = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/"
    "nvidia_deepseek-v4-flash-nvfp4/hf/hf-48bfe38_orig"
)


def resolve_model(explicit: str | None) -> str:
    """Two-checkpoint policy (2026-07-19): each backend runs the checkpoint
    format its kernel consumes natively — native/fi_dg the mx-format original,
    fi_nvfp4 the NVFP4 cast. Same base weights, so GSM8K accuracy (see
    eval_gsm8k.py) is the cross-checkpoint fairness gate. Priority:
    --model > MODEL env > per-backend default (MODEL_NVFP4 overrides the
    nvfp4 default)."""
    if explicit:
        return explicit
    if os.environ.get("MODEL"):
        return os.environ["MODEL"]
    if (
        os.environ.get("FI_MOE_EP") == "1"
        and os.environ.get("FI_MOE_EP_MEGAKERNEL") == "nvfp4_cutedsl"
    ):
        return os.environ.get("MODEL_NVFP4", DEFAULT_MODEL_NVFP4)
    return DEFAULT_MODEL


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def assert_expected_dsl() -> None:
    """Refuse to bench on an unintended CuTe-DSL runtime.

    The fi cutedsl kernels' codegen is version-sensitive; since the MR!27
    mainloop WAR (2026-07-22) the validated production runtime is 4.5.2
    (vllm 0.25.1's own pin), so that's the default expectation. Set
    EXPECT_DSL=4.6.1 when benching a DSL_461=1 venv, or EXPECT_DSL="" to
    disable the guard. Complements the cutlass_dsl_version stamp in the
    result JSON (a stamp audits after the fact; this stops a wrong-version
    run before it burns a node-hour)."""
    want = os.environ.get("EXPECT_DSL", "4.5.2")
    if not want:
        return
    got = _pkg_version("nvidia-cutlass-dsl")
    if got != want:
        raise RuntimeError(
            f"nvidia-cutlass-dsl {got!r} != EXPECT_DSL {want!r} — rebuild the "
            "venv (FRESH=1 setup_container.sh) or set EXPECT_DSL to match "
            "(empty string disables)"
        )
    print(f"[bench_offline] DSL guard: nvidia-cutlass-dsl == {got}", flush=True)


def _read_nvlink_counters() -> dict | None:
    """Cumulative NVLink data counters (KiB) per GPU, summed over links.

    Enabled via NVLINK_COUNTERS=1; sampled before the first timed round and
    after the last, so the delta covers exactly the timed window (boot,
    warmup, and capture excluded). GB200 counters via `nvidia-smi nvlink -gt d`.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "nvlink", "-gt", "d"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    gpus: dict = {}
    gpu = None
    for line in out.splitlines():
        m = re.match(r"GPU (\d+):", line)
        if m:
            gpu = int(m.group(1))
            gpus[gpu] = {"tx_kib": 0, "rx_kib": 0}
            continue
        m = re.search(r"Data (Tx|Rx): (\d+) KiB", line)
        if m and gpu is not None:
            gpus[gpu]["tx_kib" if m.group(1) == "Tx" else "rx_kib"] += int(m.group(2))
    return gpus or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workload", default="prefill:1024:1",
                    help="name:input_len:output_len")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None,
                    help="checkpoint path; default resolves per backend "
                         "(see resolve_model)")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--dp", type=int, default=int(os.environ.get("DP", "1")))
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("MAX_BATCHED_TOKENS", "4096")))
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=float(os.environ.get("GPU_MEM_UTIL", "0")) or None,
                    help="engine default when unset; the fi 8192-token bucket "
                         "needs headroom (bigger symm workspace squeezes KV)")
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("MAX_NUM_SEQS", "0")) or None,
                    help="decode concurrency cap (engine default when unset); "
                         "raise together with --num-prompts to bench "
                         "large-batch decode steps")
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("ENFORCE_EAGER", "1") == "1")
    args = ap.parse_args()
    assert_expected_dsl()
    args.model = resolve_model(args.model)
    print(f"[bench_offline] {args.tag}: model = {args.model}", flush=True)

    name, ilen, olen = args.workload.split(":")
    ilen, olen = int(ilen), int(olen)

    import random

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        # V4 models need the deepseek_v4 tokenizer; "auto" resolves V3.2 to
        # its own mode (TOKENIZER_MODE=auto for deepseek_v32 checkpoints).
        tokenizer_mode=os.environ.get("TOKENIZER_MODE", "deepseek_v4"),
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        enable_expert_parallel=True,
        moe_backend=os.environ.get("MOE_BACKEND", "deep_gemm_mega_moe"),
        max_model_len=4096,
        max_num_batched_tokens=args.max_num_batched_tokens,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
        **(
            {"gpu_memory_utilization": args.gpu_memory_utilization}
            if args.gpu_memory_utilization
            else {}
        ),
        enforce_eager=args.enforce_eager,
        # MAX_CAPTURE: extend CUDA-graph capture to prefill-sized batches
        # (default vLLM caps at 512, so 4096-token prefill steps run eager
        # and pay inter-rank arrival skew inside the collective MoE kernel).
        **(
            {
                "compilation_config": {
                    "max_cudagraph_capture_size": int(os.environ["MAX_CAPTURE"]),
                    # CAPTURE_SIZES="256,2048,4096,8192": sparse capture list.
                    # The default dense list (every 16 tokens) costs graph-pool
                    # memory per size — at MAX_CAPTURE=8192 vllm estimated
                    # 310 GiB and KV went negative. Batches round up to the
                    # nearest captured size (padding), so a few sizes suffice
                    # for fixed-shape workloads.
                    **(
                        {
                            "cudagraph_capture_sizes": [
                                int(s)
                                for s in os.environ["CAPTURE_SIZES"].split(",")
                            ]
                        }
                        if os.environ.get("CAPTURE_SIZES")
                        else {}
                    ),
                }
            }
            if os.environ.get("MAX_CAPTURE")
            else {}
        ),
        kv_cache_dtype="fp8",
        block_size=256,
        # Repeat rounds reuse the same prompts: with prefix caching on, every
        # round after warmup is a 100% cache hit and "prefill" measures nothing
        # (observed: 91k tok/s fake prefill). Disable for honest repeats.
        enable_prefix_caching=False,
    )

    # Fixed random-token prompts (seed 0), same for every round and backend.
    random.seed(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [
        {"prompt_token_ids": [random.randrange(100, vocab - 100) for _ in range(ilen)]}
        for _ in range(args.num_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=olen, ignore_eos=True)

    sample_nvlink = os.environ.get("NVLINK_COUNTERS") == "1"
    nvlink_before = None
    nvlink_span_t0 = None

    rounds = []
    for r in range(args.rounds + 1):  # +1 warmup round, discarded
        if sample_nvlink and r == 1:  # first timed round: warmup traffic excluded
            nvlink_before = _read_nvlink_counters()
            nvlink_span_t0 = time.perf_counter()
        if os.environ.get("NSYS_GATE") == "1" and r == 1:
            # Bracket exactly one timed round for nsys
            # --capture-range=cudaProfilerStart: bounds the trace to one
            # round (ungated node-mode traces hit CUPTI event caps + the
            # NCCL watchdog and lost the timed rounds — see RUNS.md run 35/37).
            import torch

            torch.cuda.profiler.start()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        n_out = sum(len(o.outputs[0].token_ids) for o in outs)
        n_total = args.num_prompts * ilen + n_out
        rec = {
            "round": r,
            "warmup": r == 0,
            "elapsed_s": dt,
            "total_tok_per_s": n_total / dt,
            "output_tok_per_s": n_out / dt,
        }
        if os.environ.get("NSYS_GATE") == "1" and r == 1:
            import torch

            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        rounds.append(rec)
        print(f"[bench_offline] {args.tag}/{name} round {r}"
              f"{' (warmup)' if r == 0 else ''}: "
              f"{rec['total_tok_per_s']:.1f} total tok/s", flush=True)

    nvlink = None
    if sample_nvlink and nvlink_before is not None:
        after = _read_nvlink_counters()
        span = time.perf_counter() - nvlink_span_t0
        if after:
            nvlink = {
                "span_s": span,
                "per_gpu": {
                    str(g): {
                        "tx_gb": (after[g]["tx_kib"] - nvlink_before[g]["tx_kib"]) / 1e6,
                        "rx_gb": (after[g]["rx_kib"] - nvlink_before[g]["rx_kib"]) / 1e6,
                    }
                    for g in sorted(after)
                    if g in nvlink_before
                },
            }

    timed = sorted(r["total_tok_per_s"] for r in rounds if not r["warmup"])
    median = timed[len(timed) // 2]
    payload = {
        "tag": args.tag,
        "workload": name,
        "input_len": ilen,
        "output_len": olen,
        "num_prompts": args.num_prompts,
        "eager": args.enforce_eager,
        "model": args.model,
        # Provenance: DSL <4.5.2 compiles the cutedsl kernels 34-54% slower
        # (perf floor 4.5.2 since the 2026-07-22 MR!27 WAR; 4.5.2 == 4.6.1
        # parity); stamp so every result is auditable.
        "cutlass_dsl_version": _pkg_version("nvidia-cutlass-dsl"),
        "fi_moe_ep": os.environ.get("FI_MOE_EP", "0"),
        "fi_megakernel": os.environ.get("FI_MOE_EP_MEGAKERNEL", "deep_gemm_mega"),
        "median_total_tok_per_s": median,
        "rounds": rounds,
        **({"nvlink": nvlink} if nvlink else {}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[bench_offline] median {median:.1f} tok/s -> {args.out}")


if __name__ == "__main__":
    main()
