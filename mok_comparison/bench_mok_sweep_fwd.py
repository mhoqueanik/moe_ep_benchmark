"""MXFP8 forward-only sweep over MoK's comm-SM / minibatch knobs.

No correctness pass (covered by bench_mok_fwd_only.py); prints one line per
config. Sweep values via env: SWEEP_COMM_SMS="4 8 16 24 36 52",
SWEEP_MINIBATCH="2048 4096 8192".
"""

import os

import torch
import torch.distributed as dist

from benchmarks.bench_mok import HIDDEN_DIM, INTERMEDIATE_DIM, NUM_EXPERTS, NUM_LOCAL_TOKENS, TOPK
from benchmarks.utils import benchmark_fwd, get_num_local_experts, get_tflops, init_distributed
from mok import functional, ops
from tests.utils import generate_inputs


def main() -> None:
    rank, world_size, device = init_distributed()
    num_local_experts = get_num_local_experts(NUM_EXPERTS, world_size)
    inputs = generate_inputs(rank, device, NUM_EXPERTS, num_local_experts, TOPK, NUM_LOCAL_TOKENS, HIDDEN_DIM, INTERMEDIATE_DIM)
    (x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down,
     w_routed_gate, w_routed_up, w_routed_down, _) = inputs

    w_gate_q = ops.mxfp8_quantize(w_routed_gate, True, True)[:2]
    w_up_q = ops.mxfp8_quantize(w_routed_up, True, True)[:2]
    w_down_q = ops.mxfp8_quantize(w_routed_down, True, True)[:2]

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        print(f"device {props.name}: {props.multi_processor_count} SMs", flush=True)

    comm_sms_list = [int(v) for v in os.environ.get("SWEEP_COMM_SMS", "4 8 16 24 36 52").split()]
    minibatch_list = [int(v) for v in os.environ.get("SWEEP_MINIBATCH", "4096").split()]

    for minibatch in minibatch_list:
        for comm_sms in comm_sms_list:
            config = functional.MoKConfig(
                fwd_num_comm_sms=comm_sms,
                bwd_num_comm_sms=comm_sms,
                minibatch_size=minibatch,
                macrobatch_size=32 * minibatch,
            )
            workspace = functional.get_workspace(
                config, dist.group.WORLD, device=device,
                num_local_tokens=NUM_LOCAL_TOKENS, hidden_size=HIDDEN_DIM, topk=TOPK,
            )

            def run_fwd():
                schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
                return functional.forward(
                    config, workspace, schedule, x, router_weights,
                    w_shared_gate, w_shared_up, w_shared_down,
                    w_gate_q, w_up_q, w_down_q,
                )

            fwd_ms = benchmark_fwd(run_fwd, device)
            if rank == 0:
                tflops = get_tflops(fwd_ms, NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM, INTERMEDIATE_DIM)
                print(
                    f"SWEEP mxfp8 fwd_comm_sms={comm_sms} minibatch={minibatch}: "
                    f"{fwd_ms:.3f} ms, {tflops:.1f} TFLOP/s",
                    flush=True,
                )
            dist.barrier()
            # Symmetric workspaces are large; don't accumulate them across configs.
            workspace = None
            functional.clear_workspace_cache()
            torch.cuda.empty_cache()

    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
