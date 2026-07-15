"""Standalone MoE Expert-Parallel benchmark for FlashInfer's fused mega path.

Mirrors ``bench_moe_ep_nonmega.py`` geometry (DeepSeek-V4-Flash EP: TP=1, DP=N)
but times the fused mega kernels (dispatch+GEMM+combine in one kernel).

Input staging (bf16 -> fp8/MXFP8/NVFP4) and weight preprocessing are lifted OUT
of the timed region.  Activations are pre-quantized once; the timed loop calls
only ``kernel.compute()``.

Launch (4 GPUs, Blackwell sm_100+):

    CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_moe_ep_mega.py \\
        --world-size 4 --mega-backend mxfp8_cutedsl \\
        --tokens-per-rank 8 --num-experts 256 --top-k 8 \\
        --hidden 7168 --intermediate 2048

Backends: deep_gemm_mega | mxfp8_cutedsl | nvfp4_cutedsl
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import socket
import traceback
from statistics import median

import torch
import torch.distributed as dist

from bench_common import (
    compute_dense_moe_reference,
    make_mega_fp4_weights,
    make_problem,
    next_pow2,
)


@dataclasses.dataclass
class Cfg:
    world_size: int
    mega_backend: str  # deep_gemm_mega | mxfp8_cutedsl | nvfp4_cutedsl
    tokens_per_rank: int
    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    warmup: int
    iters: int
    gate_up_clamp: float
    fast_math: bool
    mxfp8_kind: str  # mxfp8_e4m3 | mxfp8_e5m2
    out_csv: str | None = None


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_entry(local_rank, world_size, init_method, cfg):
    # FlashInfer's moe_ep runtime binds each process to its GPU via
    # os.environ["LOCAL_RANK"] (moe_ep/core/runtime/bootstrap.py: _ensure_cuda_device,
    # _init_nvshmem_after_dist). torch.multiprocessing.spawn inherits the parent's
    # env, so a container-set LOCAL_RANK (e.g. "0") would pin EVERY rank to cuda:0 --
    # causing illegal-address (deep_gemm_mega weight transform) or device-mismatch
    # (nvshmem uid broadcast for cutedsl) failures. Override it per-spawned-rank.
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )
    dist.all_reduce(torch.tensor([local_rank], device=device))
    try:
        _worker(
            ProcessGroupInfo(
                world_size=world_size,
                rank=local_rank,
                local_rank=local_rank,
                device=device,
            ),
            cfg,
        )
    except Exception as ex:
        print(f"[rank {local_rank}] {ex}")
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def parallel_launch(cfg: Cfg):
    from torch.multiprocessing import spawn

    init_method = f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{_get_open_port()}"
    spawn(
        _worker_entry,
        args=(cfg.world_size, init_method, cfg),
        nprocs=cfg.world_size,
        join=True,
    )


# --------------------------------------------------------------------------
# Backend helpers
# --------------------------------------------------------------------------
def _mega_knobs():
    """Cutedsl kernel tuning knobs from MEGA_KNOBS (cutedsl backends only).

    Unset/empty -> None (the shim's token-count heuristic); "auto" -> online
    autotune at the first forward (collective sweep, winner kept for the
    session); otherwise a JSON dict, e.g.
    '{"mma_tiler_mnk": [256, 128, 256], "flag_batch": 4}' (lists become the
    tuples the tuner expects).
    """
    raw = os.environ.get("MEGA_KNOBS", "").strip()
    if not raw:
        return None
    if raw == "auto":
        return "auto"
    knobs = json.loads(raw)
    return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


def _mega_ikr() -> bool:
    """MEGA_IKR=1 -> in_kernel_fc2_reduce (in-flight REDG top-k combine)."""
    return bool(int(os.environ.get("MEGA_IKR", "0")))


def _mega_combine_dtype() -> str:
    """MEGA_COMBINE_DTYPE=bf16|mxfp8|nvfp4 -> cross-rank combine wire format."""
    return os.environ.get("MEGA_COMBINE_DTYPE", "bf16")


def _mega_variant_suffix() -> str:
    """CSV compute_kernel suffix for the MEGA_IKR / MEGA_COMBINE_DTYPE variant."""
    parts = []
    if _mega_ikr():
        parts.append("ikr")
    if _mega_combine_dtype() != "bf16":
        parts.append(f"combine_{_mega_combine_dtype()}")
    return ("+" + "+".join(parts)) if parts else ""


def _build_megakernel_config(cfg: Cfg):
    if cfg.mega_backend == "deep_gemm_mega":
        from flashinfer.moe_ep import DeepGemmMegaMoeConfig

        return DeepGemmMegaMoeConfig(
            intermediate_size=cfg.intermediate,
            top_k=cfg.top_k,
            activation_clamp=cfg.gate_up_clamp,
            fast_math=cfg.fast_math,
        )
    if cfg.mega_backend == "mxfp8_cutedsl":
        from flashinfer.moe_ep import Mxfp8CutedslMegaMoeConfig

        return Mxfp8CutedslMegaMoeConfig(
            intermediate_size=cfg.intermediate,
            top_k=cfg.top_k,
            kind=cfg.mxfp8_kind,
            gate_up_clamp=cfg.gate_up_clamp,
            fast_math=cfg.fast_math,
            in_kernel_fc2_reduce=_mega_ikr(),
            knobs=_mega_knobs(),
        )
    if cfg.mega_backend == "nvfp4_cutedsl":
        from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig

        return Nvfp4CutedslMegaMoeConfig(
            intermediate_size=cfg.intermediate,
            top_k=cfg.top_k,
            gate_up_clamp=cfg.gate_up_clamp,
            fast_math=cfg.fast_math,
            in_kernel_fc2_reduce=_mega_ikr(),
            combine_dtype=_mega_combine_dtype(),
            knobs=_mega_knobs(),
        )
    raise ValueError(f"unknown mega backend: {cfg.mega_backend}")


def _prestage_inputs(cfg, rank, world, device, problem):
    """Pre-quantize activations once; return kwargs for :class:`MoEEpTensors`."""
    m = cfg.tokens_per_rank
    max_m = max(64, next_pow2(m))
    bf16_hidden = problem.hidden_states
    topk_w = problem.topk_weights
    topk_ids = problem.topk_ids

    if cfg.mega_backend == "deep_gemm_mega":
        import deep_gemm

        from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.staging import (
            stage_mega_moe_inputs,
        )

        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            cfg.num_experts,
            max_m,
            cfg.top_k,
            cfg.hidden,
            cfg.intermediate,
        )
        stage_mega_moe_inputs(
            bf16_hidden,
            topk_w,
            topk_ids,
            buf.x[:m],
            buf.x_sf[:m],
            buf.topk_idx[:m],
            buf.topk_weights[:m],
        )
        t_hidden = buf.x[:m].clone()
        t_scales = buf.x_sf[:m].clone()
        buf.destroy()
        return dict(
            hidden_states=t_hidden,
            scales=t_scales,
            topk_ids=topk_ids,
            topk_weights=topk_w,
        )

    if cfg.mega_backend == "mxfp8_cutedsl":
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
            get_symm_buffer_for_mxfp8_mega_moe,
        )
        from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        buf = get_symm_buffer_for_mxfp8_mega_moe(
            cfg.num_experts,
            max_m,
            cfg.top_k,
            cfg.hidden,
            cfg.intermediate,
            rank,
            world,
            kind=cfg.mxfp8_kind,
            gate_up_clamp=cfg.gate_up_clamp,
        )
        stage_mega_moe_inputs(
            bf16_hidden,
            topk_w,
            topk_ids,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            kind=cfg.mxfp8_kind,
        )
        t_hidden = buf.x[:m].clone()
        t_scales = buf.x_sf[:m].clone()
        buf.destroy()
        return dict(
            hidden_states=t_hidden,
            scales=t_scales,
            topk_ids=topk_ids,
            topk_weights=topk_w,
        )

    if cfg.mega_backend == "nvfp4_cutedsl":
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
            get_symm_buffer_for_mega_moe,
        )
        from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        # Identity epilogue scalars (fc1_alpha = fc2_alpha = fc1_norm_const = 1,
        # the symm-buffer defaults): the kernel loads them either way (zero perf
        # difference vs the old random make_dummy_epilogue_params), and identity
        # keeps the output on real model math so the accuracy-loss pass can
        # compare it against the bf16 dense reference.
        buf = get_symm_buffer_for_mega_moe(
            cfg.num_experts,
            max_m,
            cfg.top_k,
            cfg.hidden,
            2 * cfg.intermediate,
            rank,
            world,
            gate_up_clamp=cfg.gate_up_clamp,
        )
        stage_mega_moe_inputs(
            bf16_hidden,
            topk_w,
            topk_ids,
            buf.x[:m],
            buf.x_sf[:m],
            buf.topk_idx[:m],
            buf.topk_weights[:m],
        )
        t_hidden = buf.x[:m].clone()
        t_scales = buf.x_sf[:m].clone()
        buf.destroy()
        return dict(
            hidden_states=t_hidden,
            scales=t_scales,
            topk_ids=topk_ids,
            topk_weights=topk_w,
        )

    raise ValueError(cfg.mega_backend)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _worker(pgi: ProcessGroupInfo, cfg: Cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    device = pgi.device
    bootstrap = BootstrapConfig(
        world_size=pgi.world_size, rank=pgi.rank, auto_bootstrap=False
    )
    ensure_moe_ep_cuda_device(bootstrap)

    world, rank = pgi.world_size, pgi.rank
    m = cfg.tokens_per_rank
    max_m = max(64, next_pow2(m))
    num_local = cfg.num_experts // world

    assert cfg.num_experts % world == 0
    assert cfg.hidden % 128 == 0 and cfg.intermediate % 128 == 0

    megakernel_cfg = _build_megakernel_config(cfg)
    kernel = create_mega_kernel(megakernel_cfg)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, kernel.runtime_requirements(bootstrap)
    )

    mega = None
    try:
        problem = make_problem(
            rank,
            num_tokens=m,
            num_local_experts=num_local,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            hidden=cfg.hidden,
            intermediate=cfg.intermediate,
            device=device,
        )

        if cfg.mega_backend == "deep_gemm_mega":
            from flashinfer.moe_ep import MoEWeightPack

            w13, w2, w13_sf, w2_sf = make_mega_fp4_weights(
                problem.w13_bf16, problem.w2_bf16
            )
            weights = MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_sf, w2_scale=w2_sf)
        else:
            weights = MoEWeightPack(w13=problem.w13_bf16, w2=problem.w2_bf16)

        mega = MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=FleetParams(
                num_experts=cfg.num_experts,
                max_tokens_per_rank=max_m,
                token_hidden_size=cfg.hidden,
            ),
            weights=weights,
            backend=MegaConfig(
                megakernel=megakernel_cfg,
                quantize_input=False,
                preprocess_weights=True,
            ),
        )

        tensor_kwargs = _prestage_inputs(cfg, rank, world, device, problem)
        t = MoEEpTensors(**tensor_kwargs)

        workspace = mega._ensure_workspace()
        transformed = mega._transformed
        mega._kernel.stage_inputs(t, workspace, quantize_input=False)

        y = torch.empty(m, cfg.hidden, dtype=torch.bfloat16, device=device)

        def run():
            mega._kernel.compute(workspace, transformed, output=y)
            return y

        # MEGA_TIMING selects the timed region:
        #   e2e (default) - the full FI forward path (arg prep, workspace
        #     reset, kernel, sync, output copy), each iter launched from a
        #     global barrier + idle GPU (cold-start latency).
        #   e2e_pipelined - the SAME full FI forward path, but iters enqueued
        #     back-to-back with no per-iter barrier/sync (steady-state, like a
        #     serving pipeline).  e2e minus e2e_pipelined isolates the
        #     barrier-cold collective start skew; e2e_pipelined minus kernel
        #     isolates the per-call host/copy cost.
        #   kernel - tester parity (cutedsl_megamoe tester/solver.py
        #     perf_run): a prebuilt bare kernel launch, iters enqueued
        #     back-to-back with no host sync between, per-iter events, 300MB
        #     L2 flush outside the event window (steady-state kernel time).
        #     cutedsl backends only; deep_gemm_mega falls back to the same
        #     loop around compute() (no thunk API), so its "kernel" time
        #     still includes the FI wrapper overhead.
        timing_mode = os.environ.get("MEGA_TIMING", "e2e")
        if timing_mode not in ("e2e", "e2e_pipelined", "kernel"):
            raise ValueError(
                f"MEGA_TIMING must be e2e|e2e_pipelined|kernel, got {timing_mode!r}"
            )

        thunk = None
        if timing_mode == "kernel":
            # One full forward first so MEGA_KNOBS=auto (autotune at first
            # compute()) fires before the thunk snapshots the winning compile.
            run()
            if cfg.mega_backend == "nvfp4_cutedsl":
                from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                    nvfp4_mega_launch_thunk,
                )

                thunk = nvfp4_mega_launch_thunk(
                    transformed[0], transformed[1], workspace
                )
            elif cfg.mega_backend == "mxfp8_cutedsl":
                from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                    mxfp8_mega_launch_thunk,
                )

                thunk = mxfp8_mega_launch_thunk(
                    transformed[0], transformed[1], workspace
                )
        timed = thunk if thunk is not None else run

        for _ in range(cfg.warmup):
            timed()
        torch.cuda.synchronize()
        dist.barrier()

        samples: list[float] = []
        if timing_mode in ("kernel", "e2e_pipelined"):
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(cfg.iters)
            ]
            for ev0, ev1 in events:
                # L2 flush outside the event window (matches tester perf_run).
                _ = torch.randn(300 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
                ev0.record()
                timed()
                ev1.record()
            torch.cuda.synchronize()
            samples = [ev0.elapsed_time(ev1) * 1e3 for ev0, ev1 in events]
        else:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            for _ in range(cfg.iters):
                dist.barrier()
                torch.cuda.synchronize()
                ev0.record()
                timed()
                ev1.record()
                torch.cuda.synchronize()
                samples.append(ev0.elapsed_time(ev1) * 1e3)  # ms -> us

        us = median(samples)
        us_min, us_max = min(samples), max(samples)
        dist.barrier()

        # Accuracy-loss pass (MEGA_ACC=0 disables): one un-timed full forward
        # compared against the fp32 dense-MoE ground truth over ALL experts
        # (each rank regenerates peer weights from their deterministic seeds).
        # Reported separately from the latency/speedup numbers as a global
        # rel-L2 percentage — it captures the full low-precision cost of the
        # path (weight quant + activation quant + fc1-out requant + combine
        # wire) relative to bf16 model math.
        acc_loss_pct = float("nan")
        if bool(int(os.environ.get("MEGA_ACC", "1"))):
            run()
            torch.cuda.synchronize()
            y_val = y.float()
            y_ref = compute_dense_moe_reference(
                problem,
                world_size=world,
                num_local_experts=num_local,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                device=device,
                gate_up_clamp=cfg.gate_up_clamp,
            )
            sums = torch.stack(
                [(y_val - y_ref).square().sum(), y_ref.square().sum()]
            )
            dist.all_reduce(sums)
            acc_loss_pct = 100.0 * (sums[0] / sums[1].clamp_min(1e-30)).sqrt().item()
            dist.barrier()

        if rank == 0:
            tokens_total = m * world
            tok_s = tokens_total / (us * 1e-6) if us > 0 else float("nan")

            comm_backend = "fused_symm_mega"
            compute_kernel = cfg.mega_backend + (
                _mega_variant_suffix()
                if cfg.mega_backend in ("nvfp4_cutedsl", "mxfp8_cutedsl")
                else ""
            )

            if cfg.mega_backend == "deep_gemm_mega":
                weight_dtype = "fp4_int8_block32"
                act_compute_dtype = "fp8_e4m3fn_block32"
            elif cfg.mega_backend == "mxfp8_cutedsl":
                weight_dtype = f"mxfp8_{cfg.mxfp8_kind}_block32"
                act_compute_dtype = f"mxfp8_{cfg.mxfp8_kind}_block32"
            else:
                weight_dtype = "nvfp4_block16"
                act_compute_dtype = "nvfp4_block16"

            header = (
                "path,algo,comm_backend,compute_kernel,quant_timed,weight_dtype,"
                "input_dtype,act_compute_dtype,tokens_per_rank,gpus,num_experts,"
                "top_k,hidden,inter,e2e_us_p50,e2e_us_min,e2e_us_max,tok_s,"
                "acc_loss_pct"
            )
            row = (
                f"mega,mega,{comm_backend},{compute_kernel},no,"
                f"{weight_dtype},bfloat16,{act_compute_dtype},"
                f"{cfg.tokens_per_rank},{world},{cfg.num_experts},{cfg.top_k},"
                f"{cfg.hidden},{cfg.intermediate},"
                f"{us:.1f},{us_min:.1f},{us_max:.1f},{tok_s:.1f},"
                f"{acc_loss_pct:.3f}"
            )

            print(
                "\n=== FlashInfer MoE-EP (mega) result ===\n"
                f"  mega backend     : {compute_kernel}\n"
                f"  staging          : excluded (pre-quantized once; timed = compute only)\n"
                f"  weight prep      : excluded (preprocessed at layer init)\n"
                f"  geometry         : {world} GPUs (DP={world}, EP={world}, TP=1), "
                f"{cfg.num_experts} experts, top-{cfg.top_k}, hidden={cfg.hidden}, "
                f"inter={cfg.intermediate}, {cfg.tokens_per_rank} tokens/rank\n"
                f"  timing mode      : {timing_mode} "
                + "("
                + {
                    "kernel": "tester-parity bare kernel launch",
                    "e2e_pipelined": "full FI forward, back-to-back enqueued",
                    "e2e": "full FI forward, barrier-cold",
                }[timing_mode]
                + ")\n"
                f"  E2E latency (us) : p50={us:.1f}  min={us_min:.1f}  max={us_max:.1f}  "
                f"({cfg.iters} iters, {cfg.warmup} warmup, CUDA-event timed)\n"
                f"  throughput       : {tok_s:.1f} tok/s\n"
                f"  accuracy loss    : {acc_loss_pct:.3f}% rel-L2 vs bf16 dense "
                f"reference (all-rank; MEGA_ACC=0 to skip)\n"
                f"BENCH_CSV,{row}",
                flush=True,
            )

            if cfg.out_csv:
                write_header = (
                    not os.path.exists(cfg.out_csv)
                    or os.path.getsize(cfg.out_csv) == 0
                )
                with open(cfg.out_csv, "a") as f:
                    if write_header:
                        f.write(header + "\n")
                    f.write(row + "\n")
    finally:
        if mega is not None:
            with contextlib.suppress(Exception):
                mega.destroy()
        finalize_moe_ep_runtime(runtime)


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
def _parse() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument(
        "--mega-backend",
        required=True,
        choices=["deep_gemm_mega", "mxfp8_cutedsl", "nvfp4_cutedsl"],
    )
    p.add_argument("--tokens-per-rank", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--gate-up-clamp", type=float, default=10.0)
    p.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--mxfp8-kind",
        default="mxfp8_e4m3",
        choices=["mxfp8_e4m3", "mxfp8_e5m2"],
    )
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()
    return Cfg(
        world_size=a.world_size,
        mega_backend=a.mega_backend,
        tokens_per_rank=a.tokens_per_rank,
        num_experts=a.num_experts,
        top_k=a.top_k,
        hidden=a.hidden,
        intermediate=a.intermediate,
        warmup=a.warmup,
        iters=a.iters,
        gate_up_clamp=a.gate_up_clamp,
        fast_math=a.fast_math,
        mxfp8_kind=a.mxfp8_kind,
        out_csv=a.out_csv,
    )


def main():
    import importlib

    cfg = _parse()

    if cfg.mega_backend in ("mxfp8_cutedsl", "nvfp4_cutedsl"):
        importlib.import_module(
            "flashinfer.moe_ep.kernel_src.cutedsl_megamoe"
        )
    if cfg.mega_backend == "deep_gemm_mega":
        import deep_gemm  # noqa: F401

    parallel_launch(cfg)


if __name__ == "__main__":
    main()
