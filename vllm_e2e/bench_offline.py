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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workload", default="prefill:1024:1",
                    help="name:input_len:output_len")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("MAX_BATCHED_TOKENS", "4096")))
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("ENFORCE_EAGER", "1") == "1")
    args = ap.parse_args()

    name, ilen, olen = args.workload.split(":")
    ilen, olen = int(ilen), int(olen)

    import random

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode="deepseek_v4",
        tensor_parallel_size=args.tp,
        enable_expert_parallel=True,
        moe_backend="deep_gemm_mega_moe",
        max_model_len=4096,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype="fp8",
        block_size=256,
    )

    # Fixed random-token prompts (seed 0), same for every round and backend.
    random.seed(0)
    vocab = llm.get_tokenizer().vocab_size
    prompts = [
        {"prompt_token_ids": [random.randrange(100, vocab - 100) for _ in range(ilen)]}
        for _ in range(args.num_prompts)
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=olen, ignore_eos=True)

    rounds = []
    for r in range(args.rounds + 1):  # +1 warmup round, discarded
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
        rounds.append(rec)
        print(f"[bench_offline] {args.tag}/{name} round {r}"
              f"{' (warmup)' if r == 0 else ''}: "
              f"{rec['total_tok_per_s']:.1f} total tok/s", flush=True)

    timed = sorted(r["total_tok_per_s"] for r in rounds if not r["warmup"])
    median = timed[len(timed) // 2]
    payload = {
        "tag": args.tag,
        "workload": name,
        "input_len": ilen,
        "output_len": olen,
        "num_prompts": args.num_prompts,
        "eager": args.enforce_eager,
        "fi_moe_ep": os.environ.get("FI_MOE_EP", "0"),
        "fi_megakernel": os.environ.get("FI_MOE_EP_MEGAKERNEL", "deep_gemm_mega"),
        "median_total_tok_per_s": median,
        "rounds": rounds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[bench_offline] median {median:.1f} tok/s -> {args.out}")


if __name__ == "__main__":
    main()
