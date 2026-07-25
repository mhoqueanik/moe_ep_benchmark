# FlashInfer moe_ep on vLLM 0.25.1 — one 8-GPU SM100 node (validated)

Self-contained setup + validation for the FlashInfer `moe_ep` expert path on a
**single node with 8 SM100 GPUs**. This is the executed, numbers-carrying
executed 8-GPU procedure for this branch. Every
number below was measured from scratch on **1x8 B200** on 2026-07-25.

> **Provenance.** Fresh clone → self-built container → venv → EP8 sweep. Jobs:
> microbenchmark 2337199, e2e sweep 2337204 (DeepSeek-V4-Flash), and the V4-Pro
> e2e sweep (§6). vLLM 0.25.1, flashinfer `4_5_2-perf-fix` @ `1ee41bcd`,
> cutlass-dsl 4.5.2, 8x B200.

---

## 0. What you need

| | |
|---|---|
| GPUs | 8x SM100 (cc 10.0), one node, NVLink. B200 or GB200 NVL8. |
| CUDA | torch built for CUDA 13 (EP runtime wheels are cu13-only) |
| NCCL | >= 2.30.7 |
| vLLM | 0.25.1 |
| FlashInfer | a build exposing `flashinfer.moe_ep` (branch `4_5_2-perf-fix`) |
| DSL | nvidia-cutlass-dsl **4.5.2** (vLLM 0.25.1's own pin; version-sensitive) |
| Checkpoints | DeepSeek-V4-Flash mx + NVFP4 (see §3) |

The full from-scratch setup (clone, build the container image, venv, patch) is
identical to `RUNBOOK_REPRO.md` §1–§3 — **including its fixes**: mount the
flashinfer repo at `/host/flashinfer` when building the image, and
`HF_HUB_DISABLE_XET=1` for the checkpoint pull. Only the world size and the
EP8-specific notes below differ.

## 1. The two backends

| `--moe-backend` | megakernel | checkpoint | native baseline |
|---|---|---|---|
| `flashinfer_moe_ep_mega_deep_gemm` | `deep_gemm_mega` | mx (verbatim) | `deep_gemm_mega_moe` |
| `flashinfer_moe_ep_mega_cutedsl` | `nvfp4_cutedsl` | NVFP4 prequantized (or mx requantized at load) | `deep_gemm_mega_moe` |

`FI_MOE_EP` / `FI_MOE_EP_MEGAKERNEL` are a hard error — selection is by backend
string. EPLB is rejected for the fi backends.

## 2. Set the world size to 8

`bench_offline.py` reads `TP` from the environment (`tensor_parallel_size=TP`,
`DP=1`), so **EP = world = TP = 8**:

```bash
export TP=8
```

The e2e sweep script for this node, `job_vllm_pr_runbook_sweep_ep8.sh`, sets
`TP=8` internally and submits its own exclusive 8-GPU node. It is byte-identical
to the validated 4-GPU sweep except: `export TP=8`, the fi_cutedsl knob cache is
the EP8-retuned one (§4), and outputs are `results/sweep_ep8_*.json`.

## 3. Checkpoints

Same policy as EP4: native/fi_dg consume the **mx** original; fi_cutedsl the
**NVFP4** cast (prequantized, no dequant→requant). Pass `MODEL_NVFP4` to the
pinned NVFP4 download and **leave `MODEL` unset** (see `RUNBOOK_REPRO.md` §5c:
setting `MODEL` forces fi_cutedsl onto the mx dequant path). The CI mirror's
NVFP4 copy is off-pin (post-rewrite schema) — download the pinned revision.

## 4. EP8-specific: retune the knobs (required)

The shipped knob caches are **EP4-tuned**. At EP8 each rank holds 32 of 256
experts (not 64), which halves tokens-per-expert and moves the winning tile.
Retune once (synthetic weights, ~10 min), then point the sweep at the result:

```bash
FLASHINFER_MOE_EP_KNOB_CACHE=$PWD/results/knob_cache_ep8.json \
torchrun --nproc_per_node=8 -m flashinfer.moe_ep.tune --dtype nvfp4 \
    --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 --max-tokens 8192
```

(Note: use `--nproc_per_node=8`, not the `-np 8` in the older 8-GPU doc.) The
retune measurably matters: with it, fi_cutedsl prefill-8k reaches **1.199x** at
EP8 (vs understated numbers on the EP4 cache).

## 5. Expected results — DeepSeek-V4-Flash, 1x8 B200

### 5a. Kernel microbenchmark (`e2e_pipelined` p50 µs; job 2337199)

`deep_gemm_mega` vs `nvfp4_cutedsl` at the DSV4-Flash MoE geometry, EP8. The
DeepGEMM↔CuteDSL crossover moves **earlier** than EP4 (~1024 vs ~2048 tok/rank),
because each rank holds half as many experts:

| tok/rank | 8 | 64 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| `deep_gemm_mega` | 108.6 | 125.0 | 155.7 | 237.7 | 381.0 | 693.4 | 1321.4 |
| `nvfp4_cutedsl` | 121.8 | 134.2 | 191.5 | 232.4 | 336.7 | 578.6 | 1100.8 |
| **nvfp4 vs dg** | 0.89x | 0.93x | 0.81x | **1.02x** | 1.13x | 1.20x | 1.20x |

### 5b. e2e throughput (median total tok/s; job 2337204)

| cell | native | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k `prefill:1024:1` x256 | 39012 | 39824 (1.021x) | **46774 (1.199x)** |
| decode-1k `decode:128:256` x1024 | 31837 | 19750 (**0.620x** ⚠) | 33667 (1.057x) |
| 100K ISL / 1K, 32 conc | 29642 | 30265 (1.021x) | **32940 (1.111x)** |
| 32K ISL / 32, 32 conc | 35714 | 36604 (1.025x) | **42078 (1.178x)** |

> **⚠ fi_dg decode-1k regresses at EP8 (0.62x native, consistent across all
> timed rounds).** The flashinfer DeepGEMM-mega decode path's 8-way dispatch/
> combine overhead dominates the small decode workload, and fi_dg (unlike
> fi_cutedsl) has no knob retune. fi_dg prefill/long-context are unaffected
> (~1.02x). Use `deep_gemm_mega_moe` (native) or `fi_cutedsl` for EP8 decode.

fi_cutedsl is the consistent EP8 win (1.06–1.20x), strongest at prefill and
narrowing with context, same trend as EP4. Absolute tok/s run below EP4 for a
fixed workload (e.g. prefill native 39012 vs 44656) — expected for TP8 (more
all-reduce, MLA KV replicated per rank).

### 5c. Correctness

Config check (`test_backend_registration.py`) → 15/15. GSM8K (200 q) on this
stack: native 0.960, fi_dg 0.960, fi_cutedsl 0.955 — all in band. The logprob
smoke confirms *routing* (`[fi_moe_ep] ... world=8` banner, one per rank; absent
on native), not bit-exactness.

## 6. Expected results — DeepSeek-V4-Pro, 1x8 B200

<!-- PRO_EP8_RESULTS -->
_(filled from the V4-Pro EP8 sweep; geometry 7168/3072/384 experts/top-6, its own
EP8 knob retune. native/fi_dg on mx `hf-0366e4e`, fi_cutedsl on pinned NVFP4
`9e7e88ee`.)_

## 7. Files in this branch

Everything from `vllm-pr` (the full harness) plus:
- `vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` — the 1x8 Flash e2e sweep
- `vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` — the 1x8 V4-Pro e2e sweep
- `vllm_e2e/RUNBOOK_1x8.md` — this document
- `model_shapes/RESULTS_EP8.md` — the EP8 microbenchmark table
