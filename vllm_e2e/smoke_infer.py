"""Correctness smoke: run DeepSeek-V4-Flash through one MoE backend config and
dump greedy generations + per-token logprobs for offline comparison.

The MoE path is selected by env (so runs differ only by env):
  FI_MOE_EP=0                     -> native vLLM deep_gemm_mega_moe
  FI_MOE_EP=1                     -> flashinfer moe_ep, kernel from
  FI_MOE_EP_MEGAKERNEL=deep_gemm_mega|nvfp4_cutedsl|mxfp8_cutedsl

Usage:
    python smoke_infer.py --tag native --out results/smoke_native.json
"""

from __future__ import annotations

import argparse
import json
import os

DEFAULT_MODEL = (
    "/lustre/share/coreai_dlalgo_ci/artifacts/model/"
    "deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig"
)

PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The three laws of thermodynamics state that",
    "Once upon a time, in a village by the sea,",
    "The main difference between TCP and UDP is",
    "To make a good espresso, you need",
    "The theory of general relativity says that gravity",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="label for this config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "4")))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int,
                    default=int(os.environ.get("MAX_BATCHED_TOKENS", "4096")))
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("ENFORCE_EAGER", "1") == "1")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode=os.environ.get("TOKENIZER_MODE", "deepseek_v4"),
        tensor_parallel_size=args.tp,
        enable_expert_parallel=True,
        moe_backend=os.environ.get("MOE_BACKEND", "deep_gemm_mega_moe"),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        **(
            {"compilation_config": {"max_cudagraph_capture_size": int(os.environ["MAX_CAPTURE"])}}
            if os.environ.get("MAX_CAPTURE")
            else {}
        ),
        kv_cache_dtype="fp8",
        block_size=256,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    outputs = llm.generate(PROMPTS, sp)

    records = []
    for prompt, out in zip(PROMPTS, outputs):
        comp = out.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs = []
        for tid, lp in zip(token_ids, comp.logprobs or []):
            entry = lp.get(tid)
            logprobs.append(float(entry.logprob) if entry is not None else None)
        records.append(
            {
                "prompt": prompt,
                "text": comp.text,
                "token_ids": token_ids,
                "logprobs": logprobs,
                "cumulative_logprob": float(comp.cumulative_logprob)
                if comp.cumulative_logprob is not None
                else None,
            }
        )

    payload = {
        "tag": args.tag,
        "fi_moe_ep": os.environ.get("FI_MOE_EP", "0"),
        "fi_megakernel": os.environ.get("FI_MOE_EP_MEGAKERNEL", "deep_gemm_mega"),
        "enforce_eager": args.enforce_eager,
        "model": args.model,
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[smoke_infer] wrote {args.out}")
    for r in records[:3]:
        print(f"  {r['prompt']!r} -> {r['text'][:80]!r}")


if __name__ == "__main__":
    main()
