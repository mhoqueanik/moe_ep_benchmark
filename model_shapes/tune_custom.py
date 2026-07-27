"""Custom-space knob sweep for the nvfp4 cutedsl mega-MoE path.

The stock entry points sweep fixed spaces (``tune.py``: curated-24 or the
schedule pair; ``MEGA_KNOBS=auto``: curated-24). This driver takes the kernel
team's tester-style per-knob value grid (``--use-knob name=value``,
repeatable; values are crossed), unions it with the shim's curated candidate
set, prunes illegal combos with the shim's own ``is_valid``, and runs the
same collective autotune sweep. The winner records into the knob cache
(``FLASHINFER_MOE_EP_KNOB_CACHE``) exactly like ``tune.py``.

Launch (mirrors tune.py; must match the production EP world size / GPU):

    torchrun --nproc_per_node=8 model_shapes/tune_custom.py \\
        --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \\
        --max-tokens 512 \\
        --use-knob mma_tiler_mnk=256,256,256 --use-knob mma_tiler_mnk=256,128,256 \\
        --use-knob flag_batch=1 --use-knob flag_batch=2 ...

``--intermediate`` is the model post-SwiGLU width (the
``*MegaMoeConfig.intermediate_size`` convention), like tune.py.
NOTE: the grid may include ``in_kernel_fc2_reduce=true`` — an ikr winner is
nondeterministic in accumulation order (see shim/autotune.py); the ranked
log always shows the best non-ikr candidate too.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Any, Dict, List


def parse_knob_value(raw: str) -> Any:
    """``256,128,256`` -> tuple; ``true``/``false`` -> bool; ``none`` -> None;
    ints stay ints; anything else is a string (e.g. token_back_mode names)."""
    if "," in raw:
        return tuple(int(x) for x in raw.split(","))
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        return raw.strip()


def build_candidates(
    use_knobs: List[str], *, include_curated: bool, combine_format: str
) -> List[Dict[str, Any]]:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.autotune import (
        nvfp4_candidates,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.tuner import is_valid

    grid: Dict[str, List[Any]] = {}
    for tok in use_knobs:
        name, _, raw = tok.partition("=")
        if not raw:
            raise SystemExit(f"--use-knob needs name=value, got {tok!r}")
        grid.setdefault(name, []).append(parse_knob_value(raw))

    out: List[Dict[str, Any]] = []
    if include_curated:
        out.extend(
            nvfp4_candidates(
                combine_format=combine_format, allow_in_kernel_fc2_reduce=True
            )
        )
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        knobs = dict(zip(names, values, strict=False))
        if is_valid(knobs, combine_format=combine_format):
            out.append(knobs)

    def key(k: Dict[str, Any]) -> str:
        return json.dumps(
            {n: list(v) if isinstance(v, tuple) else v for n, v in k.items()},
            sort_keys=True,
        )

    seen, deduped = set(), []
    for k in out:
        s = key(k)
        if s not in seen:
            seen.add(s)
            deduped.append(k)
    return deduped


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--live-tokens", type=int, default=None)
    p.add_argument("--use-knob", action="append", default=[], dest="use_knobs")
    p.add_argument(
        "--no-curated",
        action="store_true",
        help="sweep ONLY the --use-knob grid (default unions in the shim's curated set)",
    )
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--timed-iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    candidates = build_candidates(
        args.use_knobs,
        include_curated=not args.no_curated,
        combine_format="bf16",
    )
    if not candidates:
        raise SystemExit("empty candidate list after validity pruning")

    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        autotune_nvfp4_mega_moe,
        create_dummy_nvfp4_inputs,
        finalize_dist,
        init_dist,
    )

    rank, world_size = init_dist()
    live_tokens = args.live_tokens if args.live_tokens is not None else args.max_tokens
    symm_buffer = None
    try:
        if rank == 0:
            print(
                f"[tune-custom] {len(candidates)} candidates "
                f"(curated union: {not args.no_curated}) "
                f"max_tokens={args.max_tokens} live_tokens={live_tokens}",
                flush=True,
            )
        y, l1, l2, symm_buffer = create_dummy_nvfp4_inputs(
            rank,
            world_size,
            args.num_experts,
            args.max_tokens,
            live_tokens,
            args.topk,
            args.hidden,
            2 * args.intermediate,
            gate_up_clamp=None,
            seed=args.seed,
        )
        winner = autotune_nvfp4_mega_moe(
            y,
            l1,
            l2,
            symm_buffer,
            num_tokens=live_tokens,
            candidates=candidates,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"[tune-custom] recorded winner: {json.dumps(winner, default=list)}",
                flush=True,
            )
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()
        finalize_dist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
