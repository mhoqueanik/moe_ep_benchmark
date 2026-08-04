"""MoK side of the fi-vs-MoK same-input cross-check.

Launch with torchrun from the MoK repo root (mok package importable, SM100
patch applied, _C built), with the xcheck dir on PYTHONPATH:

    PYTHONPATH=$PWD:$XDIR torchrun --standalone --nproc-per-node=4 \
        $XDIR/xcheck_mok.py

Loads the deterministic shared inputs (xcheck_common), runs MoK's MXFP8
forward (routed + shared expert, same call pattern as benchmarks/bench_mok.py),
and saves each rank's output to $MEGA_XCHECK_DIR/out_mok_rank{r}.pt.
"""

import os

import torch
import torch.distributed as dist

from mok import functional, ops
from xcheck_common import xcheck_tensors

TOKENS = int(os.environ.get("TOKENS", 2048))
HIDDEN = int(os.environ.get("HIDDEN", 7168))
INTER = int(os.environ.get("INTER", 3072))
EXPERTS = int(os.environ.get("NUM_EXPERTS", 384))
TOPK = int(os.environ.get("TOPK", 6))
XDIR = os.environ["MEGA_XCHECK_DIR"]


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world, device_id=device)

    xt = xcheck_tensors(rank, world, TOKENS, HIDDEN, INTER, EXPERTS, TOPK, device)

    config = functional.MoKConfig(
        fwd_num_comm_sms=36,
        bwd_num_comm_sms=36,
        minibatch_size=4096,
        macrobatch_size=32 * 4096,
    )
    workspace = functional.get_workspace(
        config, dist.group.WORLD, device=device,
        num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK,
    )

    w_gate_q = ops.mxfp8_quantize(xt["w_gate"], True, True)
    w_up_q = ops.mxfp8_quantize(xt["w_up"], True, True)
    w_down_q = ops.mxfp8_quantize(xt["w_down"], True, True)

    schedule = functional.build_schedule(
        workspace, config, xt["topk_ids"], num_local_experts=EXPERTS // world
    )
    output, _ = functional.forward(
        config,
        workspace,
        schedule,
        xt["x"],
        xt["topk_w"],
        xt["w_shared_gate"],
        xt["w_shared_up"],
        xt["w_shared_down"],
        w_gate_q[:2],
        w_up_q[:2],
        w_down_q[:2],
    )
    torch.cuda.synchronize()
    torch.save(output.cpu(), f"{XDIR}/out_mok_rank{rank}.pt")
    dist.barrier()
    if rank == 0:
        print(f"[xcheck] saved out_mok_rank*.pt to {XDIR}", flush=True)

    functional.clear_workspace_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
