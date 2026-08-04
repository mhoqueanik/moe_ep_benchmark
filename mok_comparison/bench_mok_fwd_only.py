"""Forward-only (inference) variant of bench_mok.py.

Runs the BF16 and MXFP8 forward passes only: correctness check of the
forward output against the BF16 reference, then latency/TFLOP/s via the
same benchmark_fwd harness as the stock benchmark.
"""

import torch.distributed as dist

from benchmarks.bench_mok import (
    BF16_BWD_COMM_SMS,
    BF16_FWD_COMM_SMS,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    MACROBATCH_SIZE,
    MINIBATCH_SIZE,
    MXFP8_BWD_COMM_SMS,
    MXFP8_FWD_COMM_SMS,
    NUM_EXPERTS,
    NUM_LOCAL_TOKENS,
    TOPK,
    MoKBenchmark,
)
from benchmarks.utils import benchmark_fwd, get_num_local_experts, get_tflops, init_distributed
from mok import functional
from tests.utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16


def main() -> None:
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    benchmark = MoKBenchmark(inputs)

    if rank == 0:
        print(f"tokens/rank={NUM_LOCAL_TOKENS} experts={NUM_EXPERTS} ({num_local_experts}/rank) topk={TOPK} H={HIDDEN_DIM} I={INTERMEDIATE_DIM} EP={world_size}")
        print(f"BF16 comm SMs={BF16_FWD_COMM_SMS}/{BF16_BWD_COMM_SMS}, MXFP8 comm SMs={MXFP8_FWD_COMM_SMS}/{MXFP8_BWD_COMM_SMS}, minibatch={MINIBATCH_SIZE}, macrobatch={MACROBATCH_SIZE}")

    variants = (
        ("BF16", benchmark.run_bf16_fwd, BF16_TOLERANCE),
        ("MXFP8", benchmark.run_mxfp8_fwd, MXFP8_TOLERANCE),
    )

    reference_output = run_reference_bf16(*inputs)[0]
    for precision, run_fwd, tolerance in variants:
        output, _ = run_fwd()
        check_correctness(f"{precision}/output", reference_output, output, tolerance, print_stats=rank == 0)
    del reference_output

    for precision, run_fwd, _ in variants:
        fwd_ms = benchmark_fwd(run_fwd, device)
        if rank == 0:
            fwd_tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
            print(f"{precision}: forward {fwd_ms:.3f} ms, {fwd_tflops:.1f} TFLOP/s")

    dist.barrier()
    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
