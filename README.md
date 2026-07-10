# MoE-EP Benchmark

Single-node MoE expert-parallel microbenchmarks. Each variant launches one process per GPU (`torch.multiprocessing`), mirroring DP=N + EP (TP=1).

## Subsections

| Section | Script | Backends |
|---------|--------|----------|
| `vllm_split` | `bench_moe_ep_nonmega.py` | DeepEP dispatch/combine + local compute (`deepgemm`, `trtllm`) |
| `vllm_mega` | `bench_moe_ep_vllm_mega.py` | vLLM fused mega MoE (`vllm_deep_gemm_mega`) |
| `fi_mega` | `bench_moe_ep_mega.py` | FlashInfer fused mega (`deep_gemm_mega`, `mxfp8_cutedsl`, `nvfp4_cutedsl`) |

## Output files

Results land in `results/` (override with `OUT_DIR`).

**`run.sh`** — one row per variant at a fixed `TOKENS`:

```
results/bench_<stamp>_vllm_split.{log,csv}
results/bench_<stamp>_vllm_mega.{log,csv}
results/bench_<stamp>_fi_mega.{log,csv}
```

**`run_sweep.sh`** — same variants swept over sequence length; rows append per section:

```
results/sweep_<stamp>_vllm_split.{log,csv}
results/sweep_<stamp>_vllm_mega.{log,csv}
results/sweep_<stamp>_fi_mega.{log,csv}
```

---

## `run.sh`

Run from the repo root (or adjust paths).

```bash
# Full suite (all three sections)
./moe_ep_benchmark/run.sh

# One section only
SECTION=vllm_split ./moe_ep_benchmark/run.sh
SECTION=vllm_mega  ./moe_ep_benchmark/run.sh
SECTION=fi_mega    ./moe_ep_benchmark/run.sh

# Decode path (DeepEP low-latency all2all)
ALGO=ll ./moe_ep_benchmark/run.sh

# Prefill path (DeepEP high-throughput; default)
ALGO=ht ./moe_ep_benchmark/run.sh

# Include input activation quant in split-path timing (HT only)
EXCLUDE_QUANT=0 ./moe_ep_benchmark/run.sh

# Split path only: skip vllm_mega when running all sections
VLLM_MEGA=0 ./moe_ep_benchmark/run.sh

# Split path only: single expert backend
EXPERTS_LIST=deepgemm ./moe_ep_benchmark/run.sh
EXPERTS_LIST=trtllm   ./moe_ep_benchmark/run.sh

# FI mega only: subset of backends
MEGA_LIST=deep_gemm_mega ./moe_ep_benchmark/run.sh
MEGA_LIST="mxfp8_cutedsl nvfp4_cutedsl" SECTION=fi_mega ./moe_ep_benchmark/run.sh

# Problem size / hardware
GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./moe_ep_benchmark/run.sh
TOKENS=64 NUM_EXPERTS=256 TOPK=8 HIDDEN=7168 INTER=2048 ./moe_ep_benchmark/run.sh

# Timing
WARMUP=20 ITERS=50 ./moe_ep_benchmark/run.sh

# Python interpreters (vLLM vs FlashInfer envs)
PYTHON=python FI_PYTHON=python ./moe_ep_benchmark/run.sh

# Custom output directory
OUT_DIR=/tmp/moe_bench ./moe_ep_benchmark/run.sh
```

**Tip:** To compare local expert GEMMs under the same all-to-all, use `ALGO=ht` so DeepGEMM and TRT-LLM both run on DeepEP high-throughput.

---

## `run_sweep.sh`

Sweeps `TOKENS` (tokens per rank) and reuses all `run.sh` knobs. Default points are powers of two from 1 to 8192.

```bash
# Full sweep (all sections, default seq lengths)
./moe_ep_benchmark/run_sweep.sh

# One section only
SECTION=vllm_split ./moe_ep_benchmark/run_sweep.sh
SECTION=vllm_mega  ./moe_ep_benchmark/run_sweep.sh
SECTION=fi_mega    ./moe_ep_benchmark/run_sweep.sh

# Explicit sweep points
SEQ_LENS="1 8 64 512 4096" ./moe_ep_benchmark/run_sweep.sh

# Auto-generated powers of two in a range (default: 1 .. 8192)
SEQ_LEN_MIN=1 SEQ_LEN_MAX=4096 ./moe_ep_benchmark/run_sweep.sh

# Combined with run.sh options
ALGO=ht GPUS=4 EXCLUDE_QUANT=0 ./moe_ep_benchmark/run_sweep.sh
SECTION=fi_mega MEGA_LIST=deep_gemm_mega SEQ_LENS="8 128 1024" ./moe_ep_benchmark/run_sweep.sh
VLLM_MEGA=0 SEQ_LENS="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192" ./moe_ep_benchmark/run_sweep.sh
```

---

## Environment reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SECTION` | `all` | `vllm_split`, `vllm_mega`, `fi_mega`, or `all` |
| `GPUS` | `4` | World size (= DP = EP) |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPU device list |
| `ALGO` | `ht` | `ht` (prefill) or `ll` (decode); split path only |
| `TOKENS` | `8` | Tokens per rank (`run.sh` only; sweep overrides) |
| `NUM_EXPERTS` | `256` | Total experts |
| `TOPK` | `8` | Top-k routing |
| `HIDDEN` | `7168` | Hidden size |
| `INTER` | `2048` | Expert intermediate size |
| `WARMUP` | `20` | Warmup iterations |
| `ITERS` | `50` | Timed iterations |
| `EXPERTS_LIST` | `deepgemm trtllm` | Split-path expert backends |
| `MEGA_LIST` | `deep_gemm_mega mxfp8_cutedsl nvfp4_cutedsl` | FI mega backends |
| `VLLM_MEGA` | `1` | Include `vllm_mega` when `SECTION=all` |
| `EXCLUDE_QUANT` | `1` | Exclude input quant from split-path timing |
| `PYTHON` | `python` | Interpreter for vLLM scripts |
| `FI_PYTHON` | `$PYTHON` | Interpreter for FlashInfer mega script |
| `OUT_DIR` | `moe_ep_benchmark/results` | Output directory |
| `SEQ_LENS` | *(auto)* | Space-separated sweep points (`run_sweep.sh`) |
| `SEQ_LEN_MIN` | `1` | Sweep range start (`run_sweep.sh`) |
| `SEQ_LEN_MAX` | `8192` | Sweep range end (`run_sweep.sh`) |
