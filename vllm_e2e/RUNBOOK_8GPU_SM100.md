# FlashInfer moe_ep on vLLM 0.25.1 — one 8-GPU SM100 node

Self-contained setup and validation for the FlashInfer `moe_ep` expert path on
a **single node with 8 SM100 GPUs** (B200 / GB200 NVL8). Nothing here depends
on the other documents in this directory.

> **Status: written, not executed.** Everything measured for this integration
> so far ran on **4x GB200**. The commands below are the 4-GPU procedure with
> the world size changed, and the places where 8-way genuinely differs are
> called out in §7. Treat the first run as a bring-up, not a reproduction.

---

## 1. What you need

| | |
|---|---|
| GPUs | 8x SM100 (compute capability 10.0), one node, NVLink |
| CUDA | torch built for **CUDA 13** — the EP runtime wheels (nccl4py, nvidia-nccl-cu13, nixl-cu13) are CUDA-13 only |
| NCCL | >= 2.30.7 on Blackwell (older releases fail NCCL-EP group-create) |
| vLLM | 0.25.1 (this patch is ported to that release) |
| FlashInfer | a build exposing `flashinfer.moe_ep` |
| Model | DeepSeek-V4-Flash, MXFP4 or NVFP4 weights (see §3) |

Roughly 190 GB of weights, so ~25 GB/GPU at 8-way plus KV. NVIDIA-internal
note: the container image and checkpoints referenced below are internal
artifacts; outside NVIDIA you will need your own equivalents.

## 2. The two backends

| `--moe-backend` | megakernel | consumes | extra runtime |
|---|---|---|---|
| `flashinfer_moe_ep_mega_deep_gemm` | `deep_gemm_mega` | MXFP4 verbatim | torch.distributed |
| `flashinfer_moe_ep_mega_cutedsl` | `nvfp4_cutedsl` | NVFP4 prequantized, or MXFP4 requantized at load | + NVSHMEM |

Neither name carries an arch or a dtype. The arch is checked against the live
device (all mega kernels are Blackwell-only) and the weight path is derived
from the checkpoint. The native, non-FlashInfer comparison point is
`deep_gemm_mega_moe`.

Both require expert parallel and are DeepSeek-V4 only. EPLB is rejected at
startup — the FlashInfer experts do not apply the logical-to-physical expert
map, so a rebalance would move weights without moving routing.

## 3. Checkpoints

Two formats exist and they are not interchangeable:

* **MXFP4** — expert weights `e2m1` packed two-per-byte, scales `F8_E8M0`, one
  per 32 elements. This is the stock DeepSeek-V4-Flash release. Consumed
  verbatim by `deep_gemm_mega`.
* **NVFP4** — `e2m1` weights, `e4m3` scales per 16, plus a per-tensor
  `weight_scale_2`; `hf_quant_config.json` reports `quant_algo: NVFP4`.
  Consumed prequantized by `nvfp4_cutedsl`, with no dequant/requant round
  trip.

`flashinfer_moe_ep_mega_cutedsl` runs on either — it just requantizes MXFP4 at
load. For the best CuteDSL numbers use the NVFP4 checkpoint.

## 4. Install

```bash
python -m venv --system-site-packages venv0251 && source venv0251/bin/activate
python -m pip install --upgrade pip          # 24.0's resolver crashes on NGC metadata
python -m pip install vllm==0.25.1
# FlashInfer branch with moe_ep, replacing the wheel vLLM pulled in
python -m pip uninstall -y flashinfer-python
BUILD_NIXL_EP=0 python -m pip install --no-build-isolation --no-deps -e /path/to/flashinfer
python -m pip install "nvidia-cutlass-dsl[cu13]==4.5.2"   # vLLM 0.25.1's own pin
```

Keep the JIT cache on shared storage, or every fresh container pays the full
compile again (the TRT-LLM MoE module alone is ~30 min):

```bash
export FLASHINFER_WORKSPACE_BASE=/path/on/shared/storage
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

Sanity-check the pieces:

```bash
python - <<'PY'
import importlib
for m in ("torch", "vllm", "deep_gemm", "flashinfer", "flashinfer.moe_ep",
          "cutlass", "nvshmem.core"):
    try:
        importlib.import_module(m); print(f"ok   {m}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
PY
```

## 5. Apply the patch

The integration is not in vLLM 0.25.1, so it is copied over the installed
wheel. Three files:

```bash
VLLM=$(python -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')
cp patch_0251/flashinfer_moe_ep.py "$VLLM/utils/flashinfer_moe_ep.py"
cp patch_0251/model.py             "$VLLM/models/deepseek_v4/nvidia/model.py"
bash patch_0251/apply.sh    # does the above, plus the kernel.py registration
```

The **`kernel.py` registration is required, not cosmetic**: `KernelConfig`
validates `moe_backend` against the `MoEBackend` literal and rejects an
unknown value before the model is ever built, so the two backend names have to
be added to that literal. `apply.sh` inserts them in place (a whole-file copy
would pin the rest of the config module to this patch's vLLM vintage) and
keeps `*.orig` backups.

If your FlashInfer predates 0.6.17 the backends refuse to start. For a
pre-release build:

```bash
export FI_MOE_EP_SKIP_VERSION_CHECK=1
```

## 6. Validate

**Config level (~1 min, no model).** Confirms the backends are registered, the
table is consistent, and the guards fire:

```bash
python test_backend_registration.py     # expect: ALL CHECKS PASSED
```

**End to end (8 GPUs).** Run native first, then each FlashInfer backend, and
compare greedy output:

```bash
export ENFORCE_EAGER=1 TP=8
MOE_BACKEND=deep_gemm_mega_moe \
  python smoke_infer.py --tag native --out results/native.json
MOE_BACKEND=flashinfer_moe_ep_mega_deep_gemm \
  python smoke_infer.py --tag fi_dg --out results/fi_dg.json
python compare_outputs.py results/native.json results/fi_dg.json
```

**Check the bootstrap banner**, which is the only positive proof the backend
string reached the kernel — it must appear once per EP rank:

```
[fi_moe_ep] ep_rank=0 world=8 cuda.current_device=0 megakernel=deep_gemm_mega
... one line per rank, world=8
```

The **native** run must print no `[fi_moe_ep]` line at all. On 4x GB200,
`flashinfer_moe_ep_mega_deep_gemm` was bit-exact against native (8/8 prompts,
mean |dlogprob| 0.0000) and the CuteDSL path landed at 1/8 exact with
|dlogprob| 0.016-0.13, the expected requantization band.

## 7. What actually differs at 8-way

Read this before comparing against any recorded 4-GPU number.

**Retune the kernel knobs.** The shipped caches (`knob_cache_dsv4_8k.json`,
`knob_cache_dsv4_dec2k.json`) were tuned at **EP4**. At EP8 each rank holds 32
of the 256 experts instead of 64, which halves tokens-per-expert and changes
the winning tile/scheduling choice. Quoting 4-GPU numbers against an untuned
8-GPU run will understate FlashInfer:

```bash
torchrun -np 8 -m flashinfer.moe_ep.tune --dtype nvfp4 \
    --hidden 4096 --intermediate 2048 --num-experts 256 --topk 6 \
    --max-tokens 8192            # bucket must match max_num_batched_tokens
export FLASHINFER_MOE_EP_KNOB_CACHE=$PWD/results/knob_cache_ep8.json
```

**KV memory does not improve with more TP.** DeepSeek-V4 uses MLA, and vLLM
returns one KV head per GPU regardless of TP size, so the latent KV cache is
*replicated* on every rank. Eight-way TP gives you more compute and more
aggregate HBM for weights, but per-GPU KV for a given concurrency is unchanged.
Budget it as `max_num_seqs * max_model_len * ~24.8 KB` per GPU (fp8 KV, 576
latent dims x 43 layers), on top of the weight shard.

**Expert divisibility.** 256 experts / 8 ranks = 32 per rank, which is fine.
A world size that does not divide the expert count is rejected.

**Capture every recurring shape.** Under CUDA graphs, leaving prefill chunks
eager makes the FlashInfer path look falsely slow: its eager host path
generates more inter-rank launch skew, which the collective mega kernel
absorbs as spin. Capture the decode batch size *and* the prefill chunk size:

```bash
export ENFORCE_EAGER=0 MAX_CAPTURE=8192 CAPTURE_SIZES=32,256,2048,8192
```

At `MAX_CAPTURE=8192` the sparse `CAPTURE_SIZES` list is mandatory — the dense
default made vLLM estimate 310 GiB of graph pool and drove the KV cache
negative.

## 8. Throughput

Run every backend of a regime in **one session**: the native decode number
drifts round-over-round within a session, so cross-session ratios are not
trustworthy.

```bash
export TP=8 ENFORCE_EAGER=0 MAX_CAPTURE=8192 MAX_BATCHED_TOKENS=8192 \
       CAPTURE_SIZES=32,256,2048,8192
CELL="--workload prefill:1024:1 --num-prompts 256 --rounds 3"

MOE_BACKEND=deep_gemm_mega_moe python bench_offline.py --tag native $CELL \
  --out results/ep8_native.json
MOE_BACKEND=flashinfer_moe_ep_mega_cutedsl \
  FLASHINFER_MOE_EP_KNOB_CACHE=$PWD/results/knob_cache_ep8.json \
  python bench_offline.py --tag fi_cutedsl $CELL --out results/ep8_fi.json
```

`bench_offline.py` also reports per-request TTFT and inter-token latency when
the cell is an interactivity one (long input, low `--max-num-seqs`). For
reference, on **4x GB200** the CuteDSL path measured 1.175x native on
prefill-8k, 1.070x on decode-1k, and 1.087x at 100K input with 32 concurrent.
**No 8-GPU equivalent has been measured.**

## 9. If something goes wrong

**`ValueError: ... requires flashinfer-python >= 0.6.17`** — upgrade, or set
`FI_MOE_EP_SKIP_VERSION_CHECK=1` for a pre-release build.

**`ValueError: ... is set but no longer selects anything`** — `FI_MOE_EP` or
`FI_MOE_EP_MEGAKERNEL` is still exported from an older workflow. Unset them;
selection is by backend string now, and the guard exists because a stale
export used to silently change which path ran.

**`NotImplementedError: EPLB is not supported`** — expected. Use
`deep_gemm_mega_moe` if you need EPLB.

**`NVFP4-quantized expert checkpoint requires moe_backend=...cutedsl`** — an
NVFP4 checkpoint was paired with the DeepGEMM backend. Either point at the
MXFP4 checkpoint or switch backends.

**Unknown `moe_backend`, rejected before the model loads** — `apply.sh` did not
patch `kernel.py`. Re-run it and check for the `kernel.py: registered ...` line.

**No `[fi_moe_ep]` banner but the run succeeds** — the FlashInfer path was
never taken and you measured native. Check the backend string spelling; an
unregistered name would have been rejected, so this usually means the model
took a non-mega path.

**Tracebacks about `libcudart_stub.so: undefined symbol: cudaDeviceReset`** —
cosmetic, from tilelang during worker shutdown, after results are written.
