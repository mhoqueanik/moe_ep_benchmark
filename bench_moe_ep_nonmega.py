"""Standalone MoE Expert-Parallel benchmark for vLLM's NON-mega (modular) path.

Simulates the MoE portion of a served DeepSeek-V4-Flash run configured with

    --tensor-parallel-size 1 --data-parallel-size N --enable-expert-parallel

i.e. pure Expert Parallel across N GPUs, each rank owning num_experts / N
experts. Every rank plays one DP replica and one EP rank at once (TP=1).

Non-mega path = vLLM's FusedMoEKernel (a "modular kernel") =
  all-to-all dispatch/combine (DeepEP HT or LL) + experts (DeepGEMM or trtllm-fp8).

    all-to-all              experts (fp8 block-quant)
    -----------             -------------------------
    DeepEP-HT   (Standard)  DeepGemmExperts | TrtLlmFp8ExpertsModular
    DeepEP-LL   (Batched)   BatchedDeepGemmExperts

The mega path (--moe-backend deep_gemm_mega_moe -> deep_gemm.fp8_fp4_mega_moe)
fuses dispatch+GEMM+combine into one kernel and is NOT simulated here.

Launch (single node, 4 GPUs -- matches CUDA_VISIBLE_DEVICES=0,1,2,3, DP=4):

    CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_moe_ep_nonmega.py \
        --world-size 4 --algorithm ll --experts-backend deepgemm \
        --tokens-per-rank 8 --num-experts 256 --top-k 8 \
        --hidden 7168 --intermediate 2048

Rank 0 prints one CSV row. Weights/inputs are random but correctly shaped and
typed -- this measures latency, not accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import traceback
from statistics import median

import torch
import torch.distributed as dist

from bench_common import make_problem, make_split_fp8_weights


@dataclasses.dataclass
class Cfg:
    world_size: int
    algorithm: str  # "ht" | "ll"
    experts_backend: str  # "deepgemm" | "trtllm"
    tokens_per_rank: int
    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    warmup: int
    iters: int
    routing_method: str  # RoutingMethodType member name (trtllm branch only)
    exclude_quant: bool = False  # lift input activation quant out of timed region (HT)
    out_csv: str | None = None  # append the result row (with header) to this CSV


# --------------------------------------------------------------------------
# Multi-process launch (mirrors vllm/tests/kernels/moe/parallel_utils.py)
# --------------------------------------------------------------------------
@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


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
    dist.all_reduce(torch.tensor([local_rank], device=device))  # warm comms
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

    from vllm.utils.network_utils import get_open_port

    init_method = f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}"
    spawn(
        _worker_entry,
        args=(cfg.world_size, init_method, cfg),
        nprocs=cfg.world_size,
        join=True,
    )


# --------------------------------------------------------------------------
# DeepEP all-to-all builders (mirror parallel_utils.make_deepep_*_a2a)
# --------------------------------------------------------------------------
def make_deepep_ht_a2a(pg, rank: int, world_size: int, num_local_experts: int):
    import deep_ep

    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )

    # NVLink-only (single node). For multi-node HT set num_rdma_bytes > 0.
    buffer = deep_ep.Buffer(
        group=pg,
        num_nvl_bytes=1024 * 1024 * 1024,
        num_rdma_bytes=0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )
    return DeepEPHTPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=world_size,
        dp_size=1,
        rank_expert_offset=rank * num_local_experts,
    )


def make_prequant_ht_a2a(pg, rank: int, world_size: int, num_local_experts: int):
    """DeepEP-HT prepare/finalize that skips the per-iter input activation quant.

    For the HT block-quant path, vLLM quantizes the input `a1` (hidden states) to
    fp8 *before* dispatch (deepep_ht.prepare_async). Since the benchmark input is
    static, that quantization is identical every iteration, so we compute it once
    (set_prequant) and reuse the cached (a1q, a1q_scale) here -- lifting the quant
    kernel out of the timed region while keeping dispatch/GEMM/combine intact.
    """
    import deep_ep

    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
        DeepEPHTPrepareAndFinalize,
    )

    class _PrequantDeepEPHT(DeepEPHTPrepareAndFinalize):
        _a1q: torch.Tensor | None = None
        _a1q_scale: torch.Tensor | None = None

        def set_prequant(self, a1q, a1q_scale):
            self._a1q = a1q
            self._a1q_scale = a1q_scale

        def prepare_async(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                          apply_router_weight_on_input, quant_config,
                          defer_input_quant=False):
            assert not apply_router_weight_on_input
            assert self._a1q is not None, "call set_prequant() before timing"
            # Reuse cached fp8 activations instead of quantizing a1 every call.
            return self._do_dispatch(
                tokens=self._a1q,
                token_scales=self._a1q_scale,
                rank_topk_ids=topk_ids,
                rank_topk_weights=topk_weights,
                num_experts=num_experts,
                a1_scale=None,
                quant_config=quant_config,
                defer_input_quant=defer_input_quant,
            )

    buffer = deep_ep.Buffer(
        group=pg,
        num_nvl_bytes=1024 * 1024 * 1024,
        num_rdma_bytes=0,
        low_latency_mode=False,
        num_qps_per_rank=1,
    )
    return _PrequantDeepEPHT(
        buffer=buffer,
        num_dispatchers=world_size,
        dp_size=1,
        rank_expert_offset=rank * num_local_experts,
    )


def make_deepep_ll_a2a(pg, world_size, max_tokens_per_rank, hidden, num_experts,
                       use_fp8_dispatch):
    import deep_ep

    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DeepEPLLPrepareAndFinalize,
    )

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_tokens_per_rank, hidden, world_size, num_experts
    )
    buffer = deep_ep.Buffer(
        group=pg,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_experts // world_size,
    )
    return DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=world_size,
        max_tokens_per_rank=max_tokens_per_rank,
        use_fp8_dispatch=use_fp8_dispatch,
    )


# --------------------------------------------------------------------------
# Weights / config helpers
# --------------------------------------------------------------------------
def make_block_fp8_weights(w13_bf16, w2_bf16, block_shape, use_ue8m0):
    """Block-fp8 weights from shared bf16 experts."""
    return make_split_fp8_weights(w13_bf16, w2_bf16, block_shape, use_ue8m0)


def make_moe_config(cfg: Cfg, rank: int):
    """Real FusedMoEConfig (trtllm needs true dims; deepgemm ignores most)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.utils.math_utils import next_power_of_2

    num_local = cfg.num_experts // cfg.world_size
    parallel = FusedMoEParallelConfig(
        tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
        dp_size=cfg.world_size, dp_rank=rank,
        ep_size=cfg.world_size, ep_rank=rank, sp_size=1,
        use_ep=True,
        all2all_backend=(
            "deepep_low_latency" if cfg.algorithm == "ll" else "deepep_high_throughput"
        ),
        enable_eplb=False,
    )
    routing = getattr(RoutingMethodType, cfg.routing_method, RoutingMethodType.TopK)
    return FusedMoEConfig(
        num_experts=cfg.num_experts,
        experts_per_token=cfg.top_k,
        hidden_dim=cfg.hidden,
        intermediate_size_per_partition=cfg.intermediate,
        num_local_experts=num_local,
        num_logical_experts=cfg.num_experts,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=routing,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        max_num_tokens=max(64, next_power_of_2(cfg.tokens_per_rank)),
    )


def make_dummy_moe_config():
    """Minimal config for kernels (DeepGEMM) that ignore most fields."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    return FusedMoEConfig(
        num_experts=1, experts_per_token=1, hidden_dim=1,
        intermediate_size_per_partition=1, num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU, in_dtype=torch.bfloat16,
        device="cuda", routing_method=RoutingMethodType.TopK, max_num_tokens=512,
    )


def build_experts(cfg: Cfg, moe_config, quant_config, max_tokens: int, world: int):
    if cfg.experts_backend == "deepgemm":
        if cfg.algorithm == "ll":
            from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (  # noqa: E501
                BatchedDeepGemmExperts,
            )
            return BatchedDeepGemmExperts(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
                max_num_tokens=max_tokens,
                num_dispatchers=world,
            )
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmExperts,
        )
        return DeepGemmExperts(
            moe_config=make_dummy_moe_config(), quant_config=quant_config
        )

    # trtllm-fp8: Standard layout, HT only. NOTE: TRT-LLM-gen may expect a
    # specific (swizzled) weight layout; treat this branch as the "alternative
    # expert kernel" path -- latency is meaningful, numerics are not validated.
    assert cfg.algorithm == "ht", "trtllm-fp8 modular experts require --algorithm ht"
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        TrtLlmFp8ExpertsModular,
    )
    return TrtLlmFp8ExpertsModular(moe_config=moe_config, quant_config=quant_config)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _worker(pgi: ProcessGroupInfo, cfg: Cfg):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.utils.deep_gemm import (
        get_mk_alignment_for_contiguous_layout,
        is_deep_gemm_e8m0_used,
    )
    from vllm.utils.math_utils import next_power_of_2
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    dev_idx = torch.accelerator.current_device_index()
    world = pgi.world_size
    num_local = cfg.num_experts // world
    block_shape = get_mk_alignment_for_contiguous_layout()  # e.g. [128, 128]
    use_ue8m0 = is_deep_gemm_e8m0_used()

    assert cfg.num_experts % world == 0, "num_experts must divide world_size"
    assert cfg.hidden % block_shape[0] == 0 and cfg.intermediate % block_shape[0] == 0

    m = cfg.tokens_per_rank
    problem = make_problem(
        pgi.rank,
        num_tokens=m,
        num_local_experts=num_local,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden=cfg.hidden,
        intermediate=cfg.intermediate,
        device=dev_idx,
    )
    w1, w2, w1_s, w2_s = make_block_fp8_weights(
        problem.w13_bf16, problem.w2_bf16, block_shape, use_ue8m0
    )
    s, e = pgi.rank * num_local, (pgi.rank + 1) * num_local

    # trtllm-fp8 (TRT-LLM-gen) expects the DeepSeek block-scale weights shuffled
    # into 4D BlockMajorK layout (and w13 -> w31 order), exactly what the real fp8
    # MoE layer does in process_weights_after_loading via prepare_fp8_moe_layer_for_fi.
    # deepgemm consumes the plain (E, 2N, K)/(E, K, N) block-fp8 weights directly.
    if cfg.experts_backend == "trtllm":
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            _shuffle_deepseek_fp8_moe_weights,
            swap_w13_to_w31,
        )

        w1 = swap_w13_to_w31(w1.contiguous()).contiguous()
        w1_s = swap_w13_to_w31(w1_s.contiguous()).contiguous()
        w1_s.clamp_(min=1e-10)
        w2_s = w2_s.clamp(min=1e-10).contiguous()
        # -> 4D BlockMajorK (E, K/128, Mn, 128) for both projections.
        w1, w2 = _shuffle_deepseek_fp8_moe_weights(w1, w2.contiguous())

    m, k = cfg.tokens_per_rank, cfg.hidden
    x = problem.hidden_states
    topk_ids = problem.topk_ids
    topk_w = problem.topk_weights

    expert_map = torch.full((cfg.num_experts,), -1, dtype=torch.int32, device=dev_idx)
    expert_map[s:e] = torch.arange(num_local, device=dev_idx, dtype=torch.int32)

    pg = dist.new_group(list(range(world)))
    max_tokens = max(64, next_power_of_2(m))

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=None,  # LL cannot dispatch scales; HT quantizes on dispatch
        block_shape=block_shape,
    )

    # --exclude-quant lifts the input activation fp8 quantization out of the timed
    # region. Only the HT block-quant path quantizes the *input* pre-dispatch, so
    # it's cleanly cacheable; LL quantizes post-dispatch, so we can't (warn + keep).
    exclude_quant = cfg.exclude_quant and cfg.algorithm == "ht"
    if cfg.exclude_quant and cfg.algorithm == "ll" and pgi.rank == 0:
        print(
            "[warn] --exclude-quant is only supported for --algorithm ht "
            "(LL quantizes post-dispatch); timing will INCLUDE quantization.",
            flush=True,
        )

    if cfg.algorithm == "ll":
        a2a = make_deepep_ll_a2a(
            pg, world, max_tokens, k, cfg.num_experts, use_fp8_dispatch=False
        )
    elif exclude_quant:
        a2a = make_prequant_ht_a2a(pg, pgi.rank, world, num_local)
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )

        # Pre-quantize the (static) input once, matching deepep_ht.prepare_async's
        # block-quant branch, and hand the cached fp8 tokens to the dispatch.
        a1q, a1q_scale = moe_kernel_quantize_input(
            x,
            quant_config.a1_scale,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )
        if a1q_scale is not None and a1q_scale.numel() == 1:
            a1q_scale = a1q_scale.view(1, 1)
        a2a.set_prequant(a1q, a1q_scale)
    else:
        a2a = make_deepep_ht_a2a(pg, pgi.rank, world, num_local)

    with set_current_vllm_config(VllmConfig()):
        moe_config = make_moe_config(cfg, pgi.rank)
        experts = build_experts(cfg, moe_config, quant_config, max_tokens, world)
        mk = FusedMoEKernel(prepare_finalize=a2a, fused_experts=experts, inplace=False)

        fwd_cfg = VllmConfig()
        fwd_cfg.parallel_config.data_parallel_size = world
        fwd_cfg.parallel_config.enable_expert_parallel = True
        num_tokens_across_dp = torch.tensor([m] * world, device="cpu", dtype=torch.int)

        def run():
            with set_forward_context(
                None, fwd_cfg, num_tokens=m, num_tokens_across_dp=num_tokens_across_dp
            ):
                return mk.apply(
                    hidden_states=x, w1=w1, w2=w2,
                    topk_weights=topk_w, topk_ids=topk_ids,
                    activation=MoEActivation.SILU,
                    global_num_experts=cfg.num_experts,
                    expert_map=expert_map,
                    apply_router_weight_on_input=False,
                )

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
            samples.append(ev0.elapsed_time(ev1) * 1e3)  # ms -> us

    us = median(samples)
    us_min, us_max = min(samples), max(samples)
    dist.barrier()
    if pgi.rank == 0:
        tokens_total = m * world
        tok_s = tokens_total / (us * 1e-6) if us > 0 else float("nan")

        # Human-readable names for the two kernels being benchmarked.
        if cfg.algorithm == "ll":
            comm_backend = "deepep_low_latency"
            comm_desc = "DeepEP low-latency all2all (NVSHMEM dispatch/combine)"
        else:
            comm_backend = "deepep_high_throughput"
            comm_desc = "DeepEP high-throughput all2all (NVSHMEM dispatch/combine)"

        if cfg.experts_backend == "trtllm":
            compute_kernel = "trtllm_fp8_blockscale"
            compute_desc = "TRT-LLM-gen fp8 block-scale grouped GEMM"
        elif cfg.algorithm == "ll":
            compute_kernel = "deepgemm_batched"
            compute_desc = "DeepGEMM masked grouped GEMM (fp8 block, batched layout)"
        else:
            compute_kernel = "deepgemm_contiguous"
            compute_desc = "DeepGEMM contiguous grouped GEMM (fp8 block)"

        quant_timed = "no" if exclude_quant else "yes"

        # Actual dtypes used this run (derived from the tensors).
        def _dt(t: torch.Tensor) -> str:
            return str(t.dtype).replace("torch.", "")

        weight_dtype = f"{_dt(w1)}_block{block_shape[0]}"   # e.g. float8_e4m3fn_block128
        input_dtype = _dt(x)                                # hidden states at API boundary
        act_compute_dtype = f"fp8_e4m3fn_block{block_shape[1]}"  # activations fed to the GEMM

        header = (
            "path,algo,comm_backend,compute_kernel,quant_timed,weight_dtype,input_dtype,"
            "act_compute_dtype,tokens_per_rank,gpus,num_experts,top_k,hidden,inter,"
            "e2e_us_p50,e2e_us_min,e2e_us_max,tok_s"
        )
        row = (
            f"split,{cfg.algorithm},{comm_backend},{compute_kernel},{quant_timed},"
            f"{weight_dtype},{input_dtype},{act_compute_dtype},"
            f"{cfg.tokens_per_rank},{world},{cfg.num_experts},{cfg.top_k},"
            f"{cfg.hidden},{cfg.intermediate},"
            f"{us:.1f},{us_min:.1f},{us_max:.1f},{tok_s:.1f}"
        )

        quant_note = (
            "excluded (input pre-quantized once, reused)"
            if exclude_quant
            else "included in timed region"
        )
        print(
            "\n=== vLLM MoE-EP (non-mega) result ===\n"
            f"  dispatch/combine : {comm_desc}\n"
            f"  local MoE compute: {compute_desc}\n"
            f"  dtypes           : weights={_dt(w1)} (block {block_shape[0]}x"
            f"{block_shape[1]}), input(hidden)={input_dtype}, "
            f"activations@compute={act_compute_dtype}\n"
            f"  activation quant : {quant_note}\n"
            f"  geometry         : {world} GPUs (DP={world}, EP={world}, TP=1), "
            f"{cfg.num_experts} experts, top-{cfg.top_k}, hidden={cfg.hidden}, "
            f"inter={cfg.intermediate}, {cfg.tokens_per_rank} tokens/rank\n"
            f"  E2E latency (us) : p50={us:.1f}  min={us_min:.1f}  max={us_max:.1f}  "
            f"({cfg.iters} iters, {cfg.warmup} warmup, CUDA-event timed, eager)\n"
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


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------
def _parse() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=4, help="GPUs = DP size = EP size")
    p.add_argument("--algorithm", choices=["ht", "ll"], default="ll",
                   help="all-to-all: ht=DeepEP high-throughput, ll=low-latency")
    p.add_argument("--experts-backend", choices=["deepgemm", "trtllm"],
                   default="deepgemm")
    p.add_argument("--tokens-per-rank", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--routing-method", default="DeepseekV4",
                   help="RoutingMethodType name (trtllm branch metadata only)")
    p.add_argument("--exclude-quant", action=argparse.BooleanOptionalAction, default=True,
                   help="lift input activation fp8 quant out of the timed region "
                        "(HT block-quant path only; default: excluded)")
    p.add_argument("--out-csv", default=None,
                   help="append the result row (with header) to this CSV file")
    a = p.parse_args()
    return Cfg(
        world_size=a.world_size, algorithm=a.algorithm,
        experts_backend=a.experts_backend, tokens_per_rank=a.tokens_per_rank,
        num_experts=a.num_experts, top_k=a.top_k, hidden=a.hidden,
        intermediate=a.intermediate, warmup=a.warmup, iters=a.iters,
        routing_method=a.routing_method, exclude_quant=a.exclude_quant,
        out_csv=a.out_csv,
    )


def main():
    cfg = _parse()

    from vllm.utils.import_utils import has_deep_ep, has_deep_gemm

    if not has_deep_ep():
        raise RuntimeError(
            "deep_ep is not installed -- required for the all-to-all path."
        )
    if cfg.experts_backend == "deepgemm" and not has_deep_gemm():
        raise RuntimeError(
            "deep_gemm is not installed -- required for --experts-backend deepgemm"
        )

    parallel_launch(cfg)


if __name__ == "__main__":
    main()