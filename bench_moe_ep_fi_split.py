"""Standalone MoE Expert-Parallel benchmark for FlashInfer's SPLIT path.

Mirrors ``bench_moe_ep_mega.py`` geometry (DeepSeek-V4-Flash EP: TP=1, DP=N)
but times FlashInfer's non-fused pipeline — NCCL-EP dispatch → local fused_moe
compute → NCCL-EP combine — via :class:`flashinfer.moe_ep.MoEEpSplitLayer`,
so it is the FI-side counterpart of ``bench_moe_ep_nonmega.py`` (which times
vLLM's DeepEP + DeepGEMM split path) and the split-vs-fused comparison point
for ``bench_moe_ep_mega.py``.

Weight preprocessing (bf16 -> backend-native, e.g. NVFP4 + BlockMajorK
shuffles) happens once at layer init and is OUT of the timed region.  The
per-iteration activation staging (the dispatch-layout bridge, and for nvfp4
the activation quant) is part of the compute stage and IS timed — unlike the
mega benchmark, the split path has no prestage hook to lift it out.

Launch (8 GPUs, Blackwell sm_100+):

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python bench_moe_ep_fi_split.py \\
        --world-size 8 --algorithm ll --quant bf16 \\
        --tokens-per-rank 8 --num-experts 256 --top-k 8 \\
        --hidden 7168 --intermediate 2048

Variants: --quant {bf16, nvfp4, identity}; identity times the comm-only
dispatch/combine roundtrip (no expert compute).  --algorithm {ll, ht} selects
the NCCL-EP algorithm; --layout {expert_major, rank_major} the LL receive
layout (HT always uses FLAT).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import socket
import traceback
from statistics import median

import torch
import torch.distributed as dist

from bench_common import compute_dense_moe_reference, make_problem

@dataclasses.dataclass
class Cfg:
    world_size: int
    algorithm: str  # ll | ht
    layout: str  # expert_major | rank_major (ll only; ht forces FLAT)
    quant: str  # bf16 | nvfp4 | identity
    tokens_per_rank: int
    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    warmup: int
    iters: int
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
    # Same LOCAL_RANK override as bench_moe_ep_mega.py: FlashInfer's moe_ep
    # runtime binds each process to its GPU via os.environ["LOCAL_RANK"], and
    # torch.multiprocessing.spawn inherits the parent's env — a container-set
    # LOCAL_RANK would pin every rank to cuda:0.
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
def _build_fused_moe_config(cfg: Cfg, rank: int, compute_max_tokens: int):
    """MoEConfig for the inner fused_moe kernel (mirrors flashinfer
    benchmarks/bench_moe_ep.py::_build_compute, minus the weight creation —
    weights come from bench_common's shared problem)."""
    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    num_local = cfg.num_experts // cfg.world_size
    routing = RoutingConfig(num_experts=cfg.num_experts, top_k=cfg.top_k)
    experts = ExpertConfig(
        intermediate_size=cfg.intermediate,
        local_expert_offset=rank * num_local,
        local_num_experts=num_local,
    )
    execution = ExecutionConfig(tune_max_num_tokens=compute_max_tokens)

    if cfg.quant == "nvfp4":
        # Exactly ONE candidate: materialize_fused_moe_weights prepares the
        # weight view for the first matching backend only, while the MoELayer
        # winner selection probes every candidate — a two-candidate list (as
        # in flashinfer's own benchmarks/bench_moe_ep.py) dies with
        # "Weights not prepared for backend 'trtllm_fp4_routed'" at the first
        # forward. FI_SPLIT_NVFP4_BACKEND=trtllm selects the trtllm-gen
        # kernel instead of cutedsl (the default).
        nvfp4_backend = (
            TrtllmFp4Config()
            if os.environ.get("FI_SPLIT_NVFP4_BACKEND", "cutedsl") == "trtllm"
            else CuteDslConfig()
        )
        return MoEConfig(
            routing=routing,
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=experts,
            backend=BackendOptions(candidates=(nvfp4_backend,)),
            execution=execution,
        )
    return MoEConfig(
        routing=routing,
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=experts,
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=execution,
    )


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _worker(pgi: ProcessGroupInfo, cfg: Cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        FusedMoeKernelConfig,
        IdentityConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        NcclEpConfig,
        SplitConfig,
        dummy_moe_weights,
    )
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    device = pgi.device
    world, rank = pgi.world_size, pgi.rank
    m = cfg.tokens_per_rank
    num_local = cfg.num_experts // world

    assert cfg.num_experts % world == 0

    ep_algorithm = (
        EpAlgorithm.HIGH_THROUGHPUT
        if cfg.algorithm == "ht"
        else EpAlgorithm.LOW_LATENCY
    )
    ep_layout = (
        EpLayout.RANK_MAJOR
        if cfg.layout == "rank_major"
        else EpLayout.EXPERT_MAJOR
    )
    if ep_algorithm is EpAlgorithm.HIGH_THROUGHPUT:
        # HT always uses the library's FLAT layout; the field is LL-only.
        ep_layout = EpLayout.EXPERT_MAJOR

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

    # Post-dispatch token budget the inner kernel must be tuned for (mirrors
    # flashinfer benchmarks/bench_moe_ep.py): RANK_MAJOR receives each token
    # once per source rank; EXPERT_MAJOR / HT-FLAT pad per local expert.
    if ep_layout is EpLayout.RANK_MAJOR:
        compute_max_tokens = m * world
    else:
        compute_max_tokens = num_local * m * world

    if cfg.quant == "identity":
        layer_backend = SplitConfig(comm=NcclEpConfig(), kernel=IdentityConfig())
        layer_weights = dummy_moe_weights(
            num_local_experts=num_local, hidden=cfg.hidden, device=device
        )
    else:
        layer_backend = SplitConfig(
            comm=NcclEpConfig(),
            kernel=FusedMoeKernelConfig(
                moe_config=_build_fused_moe_config(cfg, rank, compute_max_tokens)
            ),
        )
        layer_weights = MoEWeightPack(w13=problem.w13_bf16, w2=problem.w2_bf16)

    bootstrap = BootstrapConfig(
        world_size=world,
        rank=rank,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    fleet_params = FleetParams(
        num_experts=cfg.num_experts,
        max_tokens_per_rank=m,
        token_hidden_size=cfg.hidden,
        dtype_bytes=2,
        algorithm=ep_algorithm,
        layout=ep_layout,
    )

    layer: MoEEpSplitLayer | None = None
    try:
        layer = MoEEpLayer(
            bootstrap, fleet_params, weights=layer_weights, backend=layer_backend
        )
        t = MoEEpTensors(
            hidden_states=problem.hidden_states,
            topk_ids=problem.topk_ids,
            topk_weights=problem.topk_weights,
        )

        def run():
            return layer.forward(t)

        for _ in range(cfg.warmup):
            run()
        torch.cuda.synchronize()
        dist.barrier()

        # Main loop: barrier-cold e2e CUDA-event timing, identical protocol to
        # the vllm_split / mega "e2e" mode so the sections compare cell for
        # cell. Per-stage timing stays OFF here — enable_timing adds a device
        # sync inside forward().
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(cfg.iters):
            dist.barrier()
            torch.cuda.synchronize()
            ev0.record()
            run()
            ev1.record()
            torch.cuda.synchronize()
            samples.append(ev0.elapsed_time(ev1) * 1e3)  # ms -> us

        us = median(samples)
        us_min, us_max = min(samples), max(samples)
        dist.barrier()

        # Stage breakdown (dispatch / compute / combine) from the layer's
        # opt-in CUDA events — a second, separately-reported loop so the sync
        # it inserts never contaminates the e2e numbers above.
        layer.enable_timing = True
        disp_us: list[float] = []
        comp_us: list[float] = []
        comb_us: list[float] = []
        for _ in range(cfg.iters):
            dist.barrier()
            torch.cuda.synchronize()
            run()
            tm = layer.last_timings_ms
            disp_us.append(tm.get("dispatch", 0.0) * 1e3)
            comp_us.append(tm.get("compute", 0.0) * 1e3)
            comb_us.append(tm.get("combine", 0.0) * 1e3)
        layer.enable_timing = False
        d_us, cp_us, cb_us = median(disp_us), median(comp_us), median(comb_us)
        dist.barrier()

        # Accuracy-loss pass (SPLIT_ACC=0 disables; skipped for identity): one
        # un-timed forward vs the fp32 dense-MoE ground truth over ALL experts.
        # The FI split fused_moe kernels use the trtllm-gen gated-act order
        # silu(second_half) * first_half — gate_second_half selects it.
        acc_loss_pct = float("nan")
        if cfg.quant != "identity" and bool(int(os.environ.get("SPLIT_ACC", "1"))):
            y_val = run().float()
            torch.cuda.synchronize()
            y_ref = compute_dense_moe_reference(
                problem,
                world_size=world,
                num_local_experts=num_local,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                device=device,
                gate_up_clamp=None,
                gate_second_half=True,
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

            layout_name = "ht_flat" if cfg.algorithm == "ht" else cfg.layout
            comm_backend = f"nccl_ep_{layout_name}"
            if cfg.quant == "identity":
                compute_kernel = "identity"
                weight_dtype = "none"
                act_compute_dtype = "bfloat16"
                quant_timed = "no"
            elif cfg.quant == "nvfp4":
                compute_kernel = "fused_moe_nvfp4"
                weight_dtype = "nvfp4_block16"
                act_compute_dtype = "nvfp4_block16"
                quant_timed = "yes"  # act quant runs inside the compute stage
            else:
                compute_kernel = "fused_moe_bf16"
                weight_dtype = "bfloat16"
                act_compute_dtype = "bfloat16"
                quant_timed = "no"

            header = (
                "path,algo,comm_backend,compute_kernel,quant_timed,weight_dtype,"
                "input_dtype,act_compute_dtype,tokens_per_rank,gpus,num_experts,"
                "top_k,hidden,inter,e2e_us_p50,e2e_us_min,e2e_us_max,tok_s,"
                "dispatch_us_p50,compute_us_p50,combine_us_p50,acc_loss_pct"
            )
            row = (
                f"fi_split,{cfg.algorithm},{comm_backend},{compute_kernel},"
                f"{quant_timed},{weight_dtype},bfloat16,{act_compute_dtype},"
                f"{cfg.tokens_per_rank},{world},{cfg.num_experts},{cfg.top_k},"
                f"{cfg.hidden},{cfg.intermediate},"
                f"{us:.1f},{us_min:.1f},{us_max:.1f},{tok_s:.1f},"
                f"{d_us:.1f},{cp_us:.1f},{cb_us:.1f},{acc_loss_pct:.3f}"
            )

            print(
                "\n=== FlashInfer MoE-EP (split) result ===\n"
                f"  dispatch/combine : NCCL-EP {cfg.algorithm.upper()} "
                f"({layout_name} layout)\n"
                f"  local MoE compute: {compute_kernel}\n"
                f"  weight prep      : excluded (preprocessed at layer init)\n"
                f"  activation quant : "
                + (
                    "included (per-iter, inside compute stage)"
                    if quant_timed == "yes"
                    else "n/a (bf16 wire and compute)"
                )
                + "\n"
                f"  geometry         : {world} GPUs (DP={world}, EP={world}, TP=1), "
                f"{cfg.num_experts} experts, top-{cfg.top_k}, hidden={cfg.hidden}, "
                f"inter={cfg.intermediate}, {cfg.tokens_per_rank} tokens/rank\n"
                f"  E2E latency (us) : p50={us:.1f}  min={us_min:.1f}  max={us_max:.1f}  "
                f"({cfg.iters} iters, {cfg.warmup} warmup, CUDA-event timed, "
                f"barrier-cold)\n"
                f"  stage p50 (us)   : dispatch={d_us:.1f}  compute={cp_us:.1f}  "
                f"combine={cb_us:.1f}  (separate timing loop)\n"
                f"  throughput       : {tok_s:.1f} tok/s\n"
                f"  accuracy loss    : {acc_loss_pct:.3f}% rel-L2 vs bf16 dense "
                f"reference (all-rank; SPLIT_ACC=0 to skip)\n"
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
        if layer is not None:
            with contextlib.suppress(Exception):
                layer.destroy()


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
def _parse() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument(
        "--algorithm",
        choices=["ht", "ll"],
        default="ll",
        help="NCCL-EP algorithm: ht=high-throughput (FLAT), ll=low-latency",
    )
    p.add_argument(
        "--layout",
        choices=["expert_major", "rank_major"],
        default="expert_major",
        help="LL receive layout (ll only; ht always uses FLAT)",
    )
    p.add_argument(
        "--quant",
        choices=["bf16", "nvfp4", "identity"],
        default="bf16",
        help="inner kernel: fused_moe at bf16/nvfp4, or the comm-only "
        "identity baseline",
    )
    p.add_argument("--tokens-per-rank", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()
    return Cfg(
        world_size=a.world_size,
        algorithm=a.algorithm,
        layout=a.layout,
        quant=a.quant,
        tokens_per_rank=a.tokens_per_rank,
        num_experts=a.num_experts,
        top_k=a.top_k,
        hidden=a.hidden,
        intermediate=a.intermediate,
        warmup=a.warmup,
        iters=a.iters,
        out_csv=a.out_csv,
    )


def main():
    parallel_launch(_parse())


if __name__ == "__main__":
    main()
