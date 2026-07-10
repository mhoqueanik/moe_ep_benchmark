"""Standalone MoE Expert-Parallel benchmark for vLLM's fused deep_gemm mega path.

Mirrors ``bench_moe_ep_mega.py``: shared inputs/weights/routing from
``bench_common``, input staging and weight finalize outside the timed region,
timed loop calls only ``deep_gemm.fp8_fp4_mega_moe``.

Launch (4 GPUs, Blackwell sm_100+):

    CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_moe_ep_vllm_mega.py \\
        --world-size 4 --tokens-per-rank 8 --num-experts 256 --top-k 8 \\
        --hidden 7168 --intermediate 2048
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

from bench_common import (
    fp32_sf_to_ue8m0_uint8,
    make_mega_fp4_weights,
    make_problem,
    next_pow2,
)


@dataclasses.dataclass
class Cfg:
    world_size: int
    tokens_per_rank: int
    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    warmup: int
    iters: int
    gate_up_clamp: float
    fast_math: bool
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
            init_method,
        )
    except Exception as ex:
        print(f"[rank {local_rank}] {ex}")
        traceback.print_exc()
        raise
    finally:
        # _worker's finally already runs vLLM's cleanup_dist_env_and_memory(),
        # which destroys the process group. Guard against the double-destroy
        # (raises "Process group cannot be None") so a clean run exits 0.
        if dist.is_initialized():
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


def _worker(pgi: ProcessGroupInfo, cfg: Cfg, init_method: str):
    import vllm.third_party.deep_gemm as deep_gemm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.deepseek_v4 import (
        DeepseekV4MegaMoEExperts,
        _stage_deepseek_v4_mega_moe_inputs,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    device = pgi.device
    init_workspace_manager(device)

    world, rank, local_rank = pgi.world_size, pgi.rank, pgi.local_rank
    m = cfg.tokens_per_rank
    max_m = max(64, next_pow2(m))
    num_local = cfg.num_experts // world
    experts_start = rank * num_local

    assert cfg.num_experts % world == 0
    assert cfg.hidden % 128 == 0 and cfg.intermediate % 128 == 0

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.tensor_model_parallel_size = 1
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.scheduler_config.max_num_batched_tokens = max_m

    experts = None
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=world,
                rank=rank,
                distributed_init_method=init_method,
                local_rank=local_rank,
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

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

            # DeepseekV4MegaMoEExperts allocates its weight parameters with
            # bare torch.zeros(...) (no device), which default to CPU. In a
            # real vLLM run the layer is built inside a CUDA device context;
            # here we must supply one, otherwise finalize_weights() rejects the
            # CPU-resident weights ("expert weights must be loaded on CUDA").
            with torch.device(device):
                experts = DeepseekV4MegaMoEExperts(
                    vllm_config,
                    num_experts=cfg.num_experts,
                    num_local_experts=num_local,
                    experts_start_idx=experts_start,
                    top_k=cfg.top_k,
                    hidden_size=cfg.hidden,
                    intermediate_size=cfg.intermediate,
                    prefix="bench_moe_ep_vllm_mega",
                )

            w13, w2, w13_sf, w2_sf = make_mega_fp4_weights(
                problem.w13_bf16, problem.w2_bf16
            )
            experts.w13_weight.data.copy_(w13.view(torch.uint8))
            experts.w2_weight.data.copy_(w2.view(torch.uint8))
            experts.w13_weight_scale.data.copy_(fp32_sf_to_ue8m0_uint8(w13_sf))
            experts.w2_weight_scale.data.copy_(fp32_sf_to_ue8m0_uint8(w2_sf))
            experts.finalize_weights()

            symm_buffer = experts.get_symm_buffer()
            num_tokens = problem.hidden_states.shape[0]
            _stage_deepseek_v4_mega_moe_inputs(
                problem.hidden_states,
                problem.topk_weights,
                problem.topk_ids.to(torch.int32),
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens],
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
            )
            torch.cuda.synchronize()

            assert experts._transformed_l1_weights is not None
            assert experts._transformed_l2_weights is not None
            transformed_l1 = experts._transformed_l1_weights
            transformed_l2 = experts._transformed_l2_weights

            y = torch.empty(
                num_tokens, cfg.hidden, dtype=torch.bfloat16, device=device
            )

            def run():
                deep_gemm.fp8_fp4_mega_moe(
                    y,
                    transformed_l1,
                    transformed_l2,
                    symm_buffer,
                    activation_clamp=cfg.gate_up_clamp,
                    fast_math=cfg.fast_math,
                )
                return y

            for _ in range(cfg.warmup):
                run()
            torch.cuda.synchronize()
            dist.barrier()

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
                samples.append(ev0.elapsed_time(ev1) * 1e3)

            us = median(samples)
            us_min, us_max = min(samples), max(samples)
            dist.barrier()

            if rank == 0:
                tokens_total = m * world
                tok_s = tokens_total / (us * 1e-6) if us > 0 else float("nan")

                header = (
                    "path,algo,comm_backend,compute_kernel,quant_timed,weight_dtype,"
                    "input_dtype,act_compute_dtype,tokens_per_rank,gpus,num_experts,"
                    "top_k,hidden,inter,e2e_us_p50,e2e_us_min,e2e_us_max,tok_s"
                )
                row = (
                    "vllm_mega,mega,fused_symm_mega,vllm_deep_gemm_mega,no,"
                    "fp4_int8_block32,bfloat16,fp8_e4m3fn_block32,"
                    f"{cfg.tokens_per_rank},{world},{cfg.num_experts},{cfg.top_k},"
                    f"{cfg.hidden},{cfg.intermediate},"
                    f"{us:.1f},{us_min:.1f},{us_max:.1f},{tok_s:.1f}"
                )

                print(
                    "\n=== vLLM MoE-EP (mega) result ===\n"
                    "  mega backend     : vllm_deep_gemm_mega\n"
                    "  staging          : excluded (pre-quantized once; timed = compute only)\n"
                    "  weight prep      : excluded (finalized at layer init)\n"
                    f"  geometry         : {world} GPUs (DP={world}, EP={world}, TP=1), "
                    f"{cfg.num_experts} experts, top-{cfg.top_k}, hidden={cfg.hidden}, "
                    f"inter={cfg.intermediate}, {cfg.tokens_per_rank} tokens/rank\n"
                    f"  E2E latency (us) : p50={us:.1f}  min={us_min:.1f}  max={us_max:.1f}  "
                    f"({cfg.iters} iters, {cfg.warmup} warmup, CUDA-event timed)\n"
                    f"  throughput       : {tok_s:.1f} tok/s\n"
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
        with contextlib.suppress(Exception):
            cleanup_dist_env_and_memory()


def _parse() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--tokens-per-rank", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--gate-up-clamp", type=float, default=10.0)
    p.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()
    return Cfg(
        world_size=a.world_size,
        tokens_per_rank=a.tokens_per_rank,
        num_experts=a.num_experts,
        top_k=a.top_k,
        hidden=a.hidden,
        intermediate=a.intermediate,
        warmup=a.warmup,
        iters=a.iters,
        gate_up_clamp=a.gate_up_clamp,
        fast_math=a.fast_math,
        out_csv=a.out_csv,
    )


def main():
    import deep_gemm  # noqa: F401

    parallel_launch(_parse())


if __name__ == "__main__":
    main()
