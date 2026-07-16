# vLLM 0.25.1 e2e: DeepSeek-V4-Flash with native deep_gemm mega vs flashinfer moe_ep

End-to-end vLLM inference benchmark comparing:

- **native** — vLLM 0.25.1's built-in `deep_gemm_mega_moe` MoE backend
  (`DeepseekV4MegaMoEExperts`, `deep_gemm.fp8_fp4_mega_moe`)
- **fi_dg** — flashinfer `moe_ep` mega path with the `deep_gemm_mega` kernel
  (same deep_gemm kernel underneath, flashinfer runtime/layer on top)
- **fi_nvfp4 / fi_mxfp8** — flashinfer `moe_ep` CuTeDSL mega kernels
  (`nvfp4_cutedsl` / `mxfp8_cutedsl`; checkpoint fp4 weights are dequantized
  to bf16 at load and requantized with the kernel's own recipe)

Model: DeepSeek-V4-Flash (hidden 4096, moe_inter 2048, 256 experts, top-6,
43 layers) from
`/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-ai_deepseek-v4-flash/hf/hf-6e76323_orig`.
4x GB200 (SM100), TP=4 + expert parallel (EP=4).

## 0. Layout

```
vllm_e2e/
├── patch_0251/            # fi integration for the vLLM 0.25.1 wheel
│   ├── model.py           # 0.25.1 deepseek_v4/nvidia/model.py + 5 fi hunks
│   ├── fi_utils.py        # flashinfer moe_ep glue (wrapper experts class)
│   ├── apply.sh / reset.sh
├── setup_container.sh     # venv + vllm 0.25.1 + flashinfer branch + patch
├── in_container.sh        # run a command in the held-node container
├── smoke_infer.py         # correctness: greedy gen + logprob dump
├── compare_outputs.py     # diff two smoke dumps
├── bench_throughput.sh    # vllm bench throughput sweep -> CSV
├── results/               # CSVs, JSONs, smoke dumps
└── logs/
```

## 1. Get a node + container

```bash
ROOT=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
W=$ROOT/moe_ep_benchmark/vllm_e2e
# 4h hold job (node with 4 GPUs)
JOBID=$(sbatch --parsable -A coreai_libraries_cudnn -p batch -N1 \
    --ntasks-per-node=1 --time=04:00:00 \
    -J coreai_libraries_cudnn-fi.vllm_e2e.hold \
    --output=$W/logs/hold_%j.log --wrap "sleep 14400")
# every command below goes through:
JOBID=$JOBID bash $W/in_container.sh '<command>'
```

`in_container.sh` uses the flashinfer-ep image
(`$ROOT/flashinfer-ep-pt2605-mega_moe_ep-20260712.sqsh`), mounts `$ROOT` and
`/lustre/share` (model, read-only), and reuses the named container `fivllm`
across calls.

## 2. One-time setup (per venv; survives container/node restarts)

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash setup_container.sh'
```

What it does (see the script):
1. venv (`--system-site-packages`) at `vllm_e2e/venv0251` on lustre
2. `pip install vllm==0.25.1` (wheel; brings its pinned torch)
3. editable flashinfer branch install (JIT, `BUILD_NIXL_EP=0`)
4. `nvidia-cutlass-dsl >= 4.6.1` (vllm pins 4.5.2, which is a 34-54% perf
   regression for the cutedsl mega kernels — see
   `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/TUNING.md`)
5. applies `patch_0251/` to the installed vllm; sanity import checks
   (every line should print PASS)

## 3. Backend selection (env-only A/B)

All runs use `--moe-backend deep_gemm_mega_moe`; the fi path is opted in
per-run with env:

| config   | env |
|----------|-----|
| native   | `FI_MOE_EP=0` (or unset) |
| fi_dg    | `FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega` |
| fi_nvfp4 | `FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl` |
| fi_mxfp8 | `FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=mxfp8_cutedsl` |

Optional: `FI_MOE_EP_KNOBS=auto` (online autotune of cutedsl kernel knobs at
first forward) or a JSON dict of explicit knobs.

## 4. Correctness smoke

```bash
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && FI_MOE_EP=0 python smoke_infer.py --tag native --out results/smoke_native.json'
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && FI_MOE_EP=1 FI_MOE_EP_MEGAKERNEL=deep_gemm_mega python smoke_infer.py --tag fi_dg --out results/smoke_fi_dg.json'
JOBID=$JOBID bash $W/in_container.sh \
  'source venv0251/bin/activate && python compare_outputs.py results/smoke_native.json results/smoke_fi_dg.json'
```

Expectation: **fi_dg vs native is greedy-token EXACT** (identical weight
transform + identical deep_gemm kernel; only staging/launch plumbing differs).
fi_nvfp4 / fi_mxfp8 are requantized — compare logprob divergence, not exact
tokens.

## 5. Throughput benchmark

```bash
JOBID=$JOBID bash $W/in_container.sh 'bash bench_throughput.sh'
# knobs: BACKENDS="native fi_dg fi_nvfp4" WORKLOADS="prefill:1024:1 decode:128:256 mixed:512:128" NUM_PROMPTS=256 EAGER=0
```

Results land in `results/bench_<stamp>.csv` (+ per-run JSON from
`vllm bench throughput --output-json`).

## 6. Results

See `FINDINGS.md`.

## 7. Known caveats

- vLLM 0.25.1 pins `nvidia-cutlass-dsl==4.5.2`; the setup force-upgrades to
  >=4.6.1 (pip prints a resolver warning — expected). Only the fi cutedsl
  kernels consume it.
- flashinfer branch = 0.6.15 vs vllm pin 0.6.13 (editable install overrides;
  resolver warning expected).
- cutedsl megakernels get the checkpoint's fp4/ue8m0-32 weights via a
  dequant-to-bf16 + requant (double quantization) — small accuracy delta vs
  native is expected and reported in FINDINGS.md.
- EPLB is not supported on the fi path (wrapper stubs the EPLB hooks).
