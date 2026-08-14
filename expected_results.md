# Expected results — 1x8 SM100, FlashInfer `moe_ep`

Every number here was measured from a scratch clone on one **8x B200** node.
If your run lands outside the tolerances below, something is different — §5
lists the two ways that has actually happened.

**Configuration.** vLLM 0.25.1 (wheel + `vllm_e2e/patch_0251/`), flashinfer
branch `4_5_2-perf-fix` @ `1ee41bcd`, nvidia-cutlass-dsl **4.5.2** (vLLM
0.25.1's own pin).

**EP8 throughout, but the two levels reach it differently.** The e2e sweeps
(§1, §2, §4) run **TP8 + EP8, DP1** — one engine sharded eight ways. The kernel
microbenchmark (§3) runs **DP8 + EP8, TP1** — one process per GPU via
`torch.multiprocessing`, no tensor sharding, since there is no model to shard.
Expert parallelism is 8 in both, which is the axis under test; do not read the
two levels as the same parallel configuration in every respect.

e2e also: kv fp8, block 256, prefix caching off, round 0 discarded as warmup,
median of the timed rounds (3 for every cell except the 100K one, which runs 2).

**The three backends, and what a ratio between them means.**

| name | `moe_backend` string | what it is |
|---|---|---|
| `native` | `deep_gemm_mega_moe` | vLLM's own DeepGEMM MegaMoE. The baseline every speedup below is relative to. |
| `fi_dg` | `flashinfer_moe_ep_mega_deep_gemm` | FlashInfer `moe_ep` DeepGEMM MegaMoE — **the same kernel as native**, reached through different glue: the `moe_ep` wrapper rather than a torch op. So `fi_dg` vs `native` isolates integration overhead, not kernel work, and ~1.00x is the expected answer. |
| `fi_cutedsl` | `flashinfer_moe_ep_mega_cutedsl` | FlashInfer `moe_ep` NVFP4 CuteDSL MegaMoE — a **different kernel**, and the one the work is actually about. Its speedup is the result. |

Read the two columns differently: `fi_dg` at 1.02x says the wrapper costs
nothing (and, at 0.44x, said something was badly wrong — §5.1). `fi_cutedsl`
at 1.32x is the kernel win.

**Two checkpoints, deliberately.** native and fi_dg run the mx original;
fi_cutedsl runs the NVFP4 cast of the same base weights, because that is the
format its kernel consumes. That makes every throughput ratio
cross-checkpoint, which is why §4 exists and is not optional.

---

## 1. vLLM e2e — DeepSeek-V4-Flash, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_ep8.sh` (~1 h).

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 38986 | 40225 (1.032x) | 46584 (**1.195x**) |
| decode-1k | 30845 | 31494 (1.021x) | 32741 (**1.061x**) |
| 100K ISL / 1K | 29632 | 30230 (1.020x) | 32913 (**1.111x**) |
| 32K ISL / 32 | 35700 | 36597 (1.025x) | 42028 (**1.177x**) |

Latency on the interactivity cells (`REQUIRE_LATENCY=1`), fi_cutedsl vs native:
TTFT 42.2s vs 49.3s at 100K, 12.8s vs 15.1s at 32K; ITL p50 51.9ms vs 56.1ms
and 191.2ms vs 226.5ms.

## 2. vLLM e2e — DeepSeek-V4-Pro, EP8

`sbatch vllm_e2e/job_vllm_pr_runbook_sweep_pro.sh` (~2 h).

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15240 | 15630 (1.026x) | 20074 (**1.317x**) |
| decode-1k | 12897 | 13157 (1.020x) | 15368 (**1.192x**) |
| 100K ISL / 1K | 12223 | 12453 (1.019x) | 15053 (**1.231x**) |
| 32K ISL / 32 | 14117 | 14435 (1.023x) | 18250 (**1.293x**) |

Latency, fi_cutedsl vs native: TTFT 95.0s vs 122.3s at 100K and 29.5s vs 38.4s
at 32K; ITL p50 111.8ms vs 134.3ms and 441.3ms vs 573.9ms.

**The fi_cutedsl win grows with model size** — 1.19-1.32x on Pro against
1.06-1.20x on Flash, on identical cells. fi_dg is at parity (1.02x) everywhere,
on both models. If you see fi_dg far from 1.02x, read §5.1 before believing it.

## 2b. vLLM serving mode — server + client, both models

`sbatch vllm_e2e/job_vllm_serving_sweep_ep8.sh` (Flash, ~2.5 h) and
`job_vllm_serving_sweep_pro.sh` (Pro, ~3.7 h); RUNBOOK §3g. One process runs
`vllm serve --moe-backend <be>` per (cell, backend), a second runs
`vllm bench serve` against it. The cells are §1/§2's four workloads verbatim
— same lengths, request counts and per-cell engine settings — so each row
below corresponds 1:1 to an offline row above. Round 0 is discarded as
warmup and each cell reports the **median of the timed client rounds**
(3, except 100K's 2): the native decode baseline drifts round-over-round in
serving just as it does offline (measured: single warm rounds of 26786 vs
28997 tok/s on two nodes — an 8% swing, larger than the fi-vs-native
effect, which is why single-round serving numbers are quoted nowhere in
this file).

Headline is **total token throughput** (input+output, client-measured over
HTTP), the same headline as §1/§2; ratio vs native:

**DeepSeek-V4-Flash** (job 2345223; ctx32k from rerun 2345501):

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 38074 | 39280 (1.032x) | 45643 (**1.199x**) |
| decode-1k | 29419 | 29958 (1.018x) | 30861 (**1.049x**) |
| 100K ISL / 1K | 29457 | 30059 (1.020x) | 32765 (**1.112x**) |
| 32K ISL / 32 | 35661 | 36518 (1.024x) | 41932 (**1.176x**) |

Flash serving lands within 1-3% of the offline absolutes and the fi_cutedsl
ratios match the offline column to ±0.012x on every cell (1.199 vs 1.195,
1.049 vs 1.061, 1.112 vs 1.111, 1.176 vs 1.177) — the two harnesses agree.
The ctx32k row was measured twice, in separate sessions on separate nodes
(2345223 then 2345501): 35583/36463/41888 vs 35661/36518/41932 — ratios
reproduce to 0.001x. Interactivity latency, fi_cutedsl vs native: TTFT p50
42.5 s vs 49.7 s at 100K and 12.9 s vs 15.2 s at 32K; ITL p99 236 ms vs
272 ms and p50 193 ms vs 228 ms.

**DeepSeek-V4-Pro** (job 2345224; ctx32k from rerun 2345502):

| cell | native tok/s | fi_dg | fi_cutedsl |
|---|---|---|---|
| prefill-8k | 15216 | 15609 (1.026x) | 19860 (**1.305x**) |
| decode-1k | 12408 | 12619 (1.017x) | 14518 (**1.170x**) |
| 100K ISL / 1K | 12190 | 12417 (1.019x) | 14909 (**1.223x**) |
| 32K ISL / 32 | 13976 | 14290 (1.022x) | 18052 (**1.292x**) |

Pro agrees with §2 the same way Flash agrees with §1: fi_cutedsl ratios
within ±0.022x of the offline column on every cell (1.305 vs 1.317, 1.170
vs 1.192, 1.223 vs 1.231, 1.292 vs 1.293), fi_dg at wrapper parity
(1.02-1.03x) everywhere, and the fi_cutedsl win growing with model size —
over HTTP exactly as in-process. Like Flash, the ctx32k row was measured in
two sessions (2345224 then 2345502) and its ratios reproduce to 0.001x.
Interactivity latency, fi_cutedsl vs native: TTFT p50 95.5 s vs 122.4 s at
100K and 29.9 s vs 38.9 s at 32K; decode-1k ITL p50 73.8 ms vs 83.5 ms.

**Serving absolutes sit below the offline cells, by design.** The client
measures over HTTP, including tokenize/detokenize, streaming and scheduling
gaps between requests; the offline harness times `llm.generate()`
in-process. Compare serving to serving; the claim that carries across both
harnesses is the fi-vs-native *ratio*, and even that with the caveat that
the serving baseline is noisier — quote the medians and check the printed
min..max spread before reading anything into a small delta.

## 3. Kernel microbenchmark — no vLLM, no checkpoints

How these numbers are made: single node, 8x B200 (EP=DP=8, TP=1) unless a
subsection says otherwise; the six MoE geometries in
`model_shapes/shapes.tsv`; tokens/rank swept per subsection. Mega cells run
`bench_moe_ep_mega.py` and time the `MEGA_TIMING=e2e_pipelined` region;
split cells run `bench_moe_ep_fi_split.py` and time barrier-cold e2e
forwards (dispatch → inner kernel → combine) — CUDA-event timed, p50 of 50
iters after 20 warmups, per-stage `dispatch/compute/combine_us_p50` columns
in the `*_split.csv` files. Weight preprocessing is excluded everywhere;
per-iteration activation quantization is included (inside the compute stage,
or the dispatch stage for packed dispatch). Launch with
`model_shapes/submit_jobs.sh` (one job per shape) or
`model_shapes/run_model_shapes.sh`; `model_shapes/make_tables.py` renders
CSVs into tables. `acc_loss_pct` columns are synthetic-input reconstruction
error (bf16 ~0.3%, mxfp8 ~6.4%, fp4-family 20.6-24.9%), NOT model quality —
that is §4. Cross-session tolerances are §6.

Summary: on the mega path the nvfp4 CuTeDSL kernel is at parity with
`deep_gemm_mega` at small batch and up to 1.7x ahead at 8192 tok/rank; mxfp8
sits at 0.5-0.6x of dg at small batch, near parity at 8192; unquantized bf16
is the floor (mxfp8 is 1.7-2.4x faster than it). The split path is 5-14x
slower than mega end-to-end at every HT point — the gap is the unfused
compute, not the comm (§3.2).

### 3.1 Mega path

#### 3.1a deep_gemm vs nvfp4 CuTeDSL variants

CSVs in `model_shapes/results_ep8/`. `e2e_pipelined` p50 µs per rank; each
CuTeDSL variant's speedup vs `deep_gemm_mega` in brackets (>1.00x = CuTeDSL
ahead). Variants: base nvfp4, `+ikr` (in-kernel fc2 reduce; nondeterministic
accumulation order), `+combine_nvfp4` / `+combine_mxfp8` (quantized combine
wire).

**`deepseek_v4_flash`** — hidden 4096, inter 2048, 256 experts, top-6 — the geometry the §1 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 108.5 | 119.8 (0.91x) | 128.0 (0.85x) | 126.0 (0.86x) | 128.0 (0.85x) |
| 64 | 124.9 | 132.2 (0.94x) | 146.4 (0.85x) | 146.4 (0.85x) | 144.4 (0.86x) |
| 512 | 154.7 | 189.4 (0.82x) | 192.0 (0.81x) | 168.9 (0.92x) | 173.1 (0.89x) |
| 1024 | 233.5 | 232.4 (1.00x) | 237.0 (0.99x) | 197.5 (1.18x) | 205.9 (1.13x) |
| 2048 | 379.1 | 334.8 (1.13x) | 334.8 (1.13x) | 273.4 (1.39x) | 287.7 (1.32x) |
| 4096 | 680.0 | 578.5 (1.18x) | 574.4 (1.18x) | 422.9 (1.61x) | 472.0 (1.44x) |
| 8192 | 1320.4 | 1104.8 (1.20x) | 1091.7 (1.21x) | 772.2 (1.71x) | 887.3 (1.49x) |

**`deepseek_v4_pro`** — hidden 7168, inter 3072, 384 experts, top-6 — the geometry the §2 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 260.1 | 261.1 (1.00x) | 267.3 (0.97x) | 267.2 (0.97x) | 268.6 (0.97x) |
| 64 | 327.7 | 334.7 (0.98x) | 349.1 (0.94x) | 345.0 (0.95x) | 347.1 (0.94x) |
| 512 | 376.9 | 394.3 (0.96x) | 398.3 (0.95x) | 377.8 (1.00x) | 382.0 (0.99x) |
| 1024 | 492.1 | 441.3 (1.12x) | 444.4 (1.11x) | 418.8 (1.18x) | 426.9 (1.15x) |
| 2048 | 899.1 | 626.2 (1.44x) | 664.0 (1.35x) | 572.4 (1.57x) | 586.7 (1.53x) |
| 4096 | 1591.8 | 1023.0 (1.56x) | 1036.2 (1.54x) | 941.1 (1.69x) | 962.0 (1.65x) |
| 8192 | 3144.2 | 1919.0 (1.64x) | 1945.0 (1.62x) | 1716.7 (1.83x) | 1727.9 (1.82x) |

**`deepseek_v3`** — hidden 7168, inter 2048, 256 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 170.8 | 171.0 (1.00x) | 185.3 (0.92x) | 183.2 (0.93x) | 183.3 (0.93x) |
| 64 | 184.4 | 183.3 (1.01x) | 207.9 (0.89x) | 203.7 (0.91x) | 205.9 (0.90x) |
| 512 | 282.6 | 267.3 (1.06x) | 271.2 (1.04x) | 242.8 (1.16x) | 253.0 (1.12x) |
| 1024 | 465.0 | 375.8 (1.24x) | 384.0 (1.21x) | 314.4 (1.48x) | 326.6 (1.42x) |
| 2048 | 809.4 | 576.5 (1.40x) | 598.9 (1.35x) | 490.5 (1.65x) | 517.1 (1.57x) |
| 4096 | 1604.6 | 1045.5 (1.53x) | 1061.9 (1.51x) | 860.1 (1.87x) | 892.9 (1.80x) |
| 8192 | 3236.3 | 2031.6 (1.59x) | 2092.1 (1.55x) | 1576.0 (2.05x) | 1707.0 (1.90x) |

**`kimi_k2_6`** — hidden 7168, inter 2048, 384 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 206.8 | 209.9 (0.99x) | 226.3 (0.91x) | 216.1 (0.96x) | 218.0 (0.95x) |
| 64 | 253.9 | 245.3 (1.04x) | 271.3 (0.94x) | 257.0 (0.99x) | 257.0 (0.99x) |
| 512 | 315.4 | 320.5 (0.98x) | 322.0 (0.98x) | 291.8 (1.08x) | 302.2 (1.04x) |
| 1024 | 450.7 | 408.4 (1.10x) | 410.6 (1.10x) | 342.9 (1.31x) | 355.3 (1.27x) |
| 2048 | 825.3 | 619.5 (1.33x) | 639.9 (1.29x) | 533.6 (1.55x) | 563.1 (1.47x) |
| 4096 | 1664.0 | 1048.5 (1.59x) | 1070.0 (1.56x) | 875.5 (1.90x) | 901.7 (1.85x) |
| 8192 | 3128.4 | 2045.9 (1.53x) | 2101.2 (1.49x) | 1628.6 (1.92x) | 1759.7 (1.78x) |

**`qwen3_5_397b`** — hidden 4096, inter 1024, 512 experts, top-10.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | 112.7 | 125.8 (0.90x) | 150.6 (0.75x) | 138.2 (0.82x) | 141.7 (0.80x) |
| 64 | 131.1 | 142.5 (0.92x) | 181.2 (0.72x) | 162.8 (0.81x) | 166.8 (0.79x) |
| 512 | 194.7 | 209.9 (0.93x) | 214.9 (0.91x) | 179.2 (1.09x) | 189.4 (1.03x) |
| 1024 | 309.2 | 298.0 (1.04x) | 310.3 (1.00x) | 240.6 (1.29x) | 259.2 (1.19x) |
| 2048 | 549.0 | 465.9 (1.18x) | 474.1 (1.16x) | 351.1 (1.56x) | 384.0 (1.43x) |
| 4096 | 1032.2 | 855.1 (1.21x) | 877.6 (1.18x) | 592.9 (1.74x) | 686.0 (1.50x) |
| 8192 | 2022.4 | 1619.0 (1.25x) | 1678.3 (1.21x) | 1098.8 (1.84x) | 1270.8 (1.59x) |

**`gpt_oss_120b`** — hidden 2880, inter 2880, 128 experts, top-4 — `dg` is
`—` throughout: `deep_gemm_mega` requires hidden and intermediate both
divisible by 128, and 2880 is not, so there is no baseline to divide by.

| tok/rank | dg | nvfp4 bf16 | +ikr | +combine_nvfp4 | +combine_mxfp8 |
|---|---|---|---|---|---|
| 8 | — | 93.2 | 95.4 | 97.2 | 97.4 |
| 64 | — | 95.2 | 99.4 | 101.3 | 101.4 |
| 512 | — | 132.2 | 136.2 | 127.9 | 132.0 |
| 1024 | — | 173.1 | 177.2 | 165.0 | 169.0 |
| 2048 | — | 240.7 | 244.8 | 222.1 | 226.4 |
| 4096 | — | 379.9 | 383.8 | 329.6 | 339.1 |
| 8192 | — | 697.4 | 697.2 | 541.7 | 607.3 |

**The crossover sits between 512 and 1024 tok/rank on every shape.** Below it
`deep_gemm_mega` wins; above it the CuteDSL variants pull away, and the
quantized-combine wires (`+combine_nvfp4`, `+combine_mxfp8`) extend the lead
further at large batches.

**V4-Flash is the least favourable geometry of the five that have a baseline.**
The shape the §1 e2e sweep uses tops out at 1.20x on plain `nvfp4_bf16`, where
V4-Pro reaches 1.64x and `deepseek_v3` 1.59x. So §1's 1.06-1.20x end-to-end is
a conservative reading of the kernel, and §2's larger Pro gains follow the
kernel rather than any integration difference.

`gpt_oss_120b`'s empty `dg` column is expected, not a failed run: the harness
attempts `deep_gemm_mega`, hits its `hidden % 128 == 0 && intermediate % 128 == 0`
assertion once per token count, logs it and carries on with the CuteDSL
variants.

`acc_loss_pct` in the CSVs is a synthetic-input reconstruction error (20.6% for
`deep_gemm_mega`, 23.1-24.9% for the CuteDSL variants), **not** a model-quality
number. Model quality is §4.

#### 3.1b deep_gemm vs nvfp4 vs mxfp8 vs bf16

The four mega dtype points side by side; speedup vs `dg` in brackets on
every column (bf16 <1x by construction — it is the unquantized baseline).
`dg`/`nvfp4` are an independent 2026-07-29 remeasurement of §3.1a (jobs
2347073-2347078, v4_flash rerun 2347255, CSVs
`model_shapes/results_ep8_split_20260729/`); every shared point agrees with
§3.1a within the §6 ±0.02x tolerance at 8-2048 tok/rank (8192 moves up to
0.08x with node/thermal state — read that row with a wider error bar). mxfp8
is from the same run; bf16 was measured 2026-08-10 (jobs 2384005-2384012,
CSVs `model_shapes/results_ep8_bf16_20260810/`, flashinfer
`sm100_bf16_implementation` branch — the kernel only exists there — with
cutlass-dsl 4.6.1; same-session `fi_dg`/`fi_fp8` controls agree within ~1%
at 8-2048 tok/rank, worst +3.5%).

**`deepseek_v4_flash`** — hidden 4096, inter 2048, 256 experts, top-6 — the geometry the §1 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | 108.5 | 121.7 (0.89x) | 171.1 (0.63x) | 336.9 (0.32x) |
| 64 | 125.4 | 134.1 (0.94x) | 197.7 (0.63x) | 392.2 (0.32x) |
| 512 | 157.8 | 191.4 (0.82x) | 263.2 (0.60x) | 433.7 (0.36x) |
| 2048 | 383.0 | 335.5 (1.14x) | 437.1 (0.88x) | 822.2 (0.47x) |
| 8192 | 1324.1 | 1101.9 (1.20x) | 1364.0 (0.97x) | 2705.9 (0.49x) |

**`deepseek_v4_pro`** — hidden 7168, inter 3072, 384 experts, top-6 — the geometry the §2 e2e sweep uses.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | 260.2 | 259.1 (1.00x) | 467.9 (0.56x) | 1032.7 (0.25x) |
| 64 | 329.7 | 334.8 (0.98x) | 646.1 (0.51x) | 1449.0 (0.23x) |
| 512 | 374.8 | 394.3 (0.95x) | 732.3 (0.51x) | 1580.0 (0.24x) |
| 2048 | 884.3 | 618.9 (1.43x) | 1166.4 (0.76x) | 2434.1 (0.36x) |
| 8192 | 3176.0 | 1890.2 (1.68x) | 3552.3 (0.89x) | 7117.0 (0.45x) |

**`deepseek_v3`** — hidden 7168, inter 2048, 256 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | 170.6 | 171.1 (1.00x) | 275.6 (0.62x) | 633.9 (0.27x) |
| 64 | 184.4 | 185.2 (1.00x) | 306.2 (0.60x) | 732.3 (0.25x) |
| 512 | 278.5 | 267.2 (1.04x) | 445.4 (0.63x) | 805.9 (0.35x) |
| 2048 | 803.9 | 576.4 (1.39x) | 980.5 (0.82x) | 1931.7 (0.42x) |
| 8192 | 3056.1 | 2021.3 (1.51x) | 3256.8 (0.94x) | 6504.4 (0.47x) |

**`kimi_k2_6`** — hidden 7168, inter 2048, 384 experts, top-8.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | 207.3 | 210.0 (0.99x) | 361.5 (0.57x) | 832.5 (0.25x) |
| 64 | 253.8 | 246.7 (1.03x) | 439.4 (0.58x) | 1047.6 (0.24x) |
| 512 | 314.2 | 320.4 (0.98x) | 529.4 (0.59x) | 1145.9 (0.27x) |
| 2048 | 813.2 | 619.6 (1.31x) | 1082.4 (0.75x) | 2164.7 (0.38x) |
| 8192 | 3185.8 | 2069.5 (1.54x) | 3469.3 (0.92x) | 6678.0 (0.48x) |

**`qwen3_5_397b`** — hidden 4096, inter 1024, 512 experts, top-10.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | 112.7 | 124.0 (0.91x) | 181.1 (0.62x) | 349.2 (0.32x) |
| 64 | 131.1 | 144.2 (0.91x) | 211.3 (0.62x) | 414.7 (0.32x) |
| 512 | 194.7 | 209.4 (0.93x) | 312.3 (0.62x) | 453.6 (0.43x) |
| 2048 | 550.4 | 465.8 (1.18x) | 521.3 (1.06x) | 1023.9 (0.54x) |
| 8192 | 2018.9 | 1622.5 (1.24x) | 1750.0 (1.15x) | 3782.8 (0.53x) |

**`gpt_oss_120b`** — hidden 2880, inter 2880, 128 experts, top-4.

| tok/rank | dg | nvfp4 bf16 | mxfp8 | bf16 |
|---|---|---|---|---|
| 8 | — | 93.1 | 130.0 | — |
| 64 | — | 95.2 | 132.2 | — |
| 512 | — | 132.1 | 167.0 | — |
| 2048 | — | 240.8 | 279.6 | — |
| 8192 | — | 697.3 | 815.1 | — |

`gpt_oss_120b`'s empty `dg` cells are expected: the harness hits
deep_gemm's `hidden % 128 == 0 && intermediate % 128 == 0` assertion and
carries on. Its empty bf16 cells date from an over-strict fleet gate
(`token_hidden_size % 128`) that has since been relaxed (validated
2026-08-10, job 2384696: 0.29% rel-L2; spot numbers 295/324/1500 µs at
8/512/8192 tok/rank, barrier-cold e2e, NOT this table's `e2e_pipelined`).

### 3.2 Split path — by kernel

No speedup-vs-mega brackets here: the split path is not competitive with the
mega path at any point measured — **5-14x slower e2e at every HT cell**
(e.g. v4_flash HT @8 tok/rank: dispatch 93 µs + `fused_moe` compute 539 µs +
combine 114 µs vs the mega path's 108 µs total; the gap is the unfused
compute, not the comm). Values are barrier-cold e2e p50 µs.

Protocols per kernel: **HT** (`ht_flat`, nccl_ep only — nixl_ep has no HT)
sweeps 8-8192 tok/rank; **LL** (`EXPERT_MAJOR`) sweeps 8-512 and compares
nccl_ep vs nixl_ep side by side (`nixl vs nccl` = nccl/nixl, >1.00x = nixl
faster). Provenance — HT nvfp4: 2026-07-29 jobs 2347073-2347078
(`results_ep8_split_20260729/`); HT w4a8/packed: 2026-08-13 jobs
2391067-2391072 (`results_ep8_w4a8_20260813/`, flashinfer
`split_cutedsl_w4a8` d5ad8f00, dsl 4.6.1); HT identity: 2026-08-14 job
2391637 (`results_ep8_ht_id_20260814/`); all LL: 2026-08-14 job 2391254
(`results_ep8_ll_nixl_20260814/`, image
`nixl_ep_ci/fi-nixl-provisioned.sqsh`, nixl-cu13 wheel ==1.3.1, branch
b3655421). Same-session HT controls agree with the July columns within
~2-3% at 512+ tok/rank; 8-64 tok/rank split cells swing ±10-20% between
sessions — read small-batch cross-column deltas as directional.
`qwen3_5_397b` (top-10) has no nccl LL cells (`numTopk <= kNumMaxTopK`,
cap 8 — runbook); `gpt_oss_120b` (hidden 2880) runs only the identity
kernel (nvfp4/w4a8 geometry gates) and was not swept at LL.

#### 3.2a identity (comm-only dispatch/combine roundtrip)

The transport floor with no expert compute. At 8 tok/rank the HT floor is
~210-330 µs vs LL's ~80-130 µs (2-3x) — why HT is the prefill protocol and
LL the decode one. nccl's combine anomaly between 8 and 64 tok/rank shows in
BOTH protocols on the hidden-7168 shapes (HT: ~800 µs @64 vs ~330 @8 and
~390 @512; LL: 560-599 µs @64, a 4.1-6.1x nixl advantage — nixl stays
flat).

HT (nccl_ep):

| shape \ tok/rank | 8 | 64 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| deepseek_v4_flash | 315.9 | 315.2 | 315.9 | 1973.5 | 3154.8 |
| deepseek_v4_pro | 229.1 | 799.2 | 367.3 | 2193.3 | 3953.8 |
| deepseek_v3 | 324.3 | 801.7 | 389.2 | 2258.4 | 4079.8 |
| kimi_k2_6 | 331.3 | 809.4 | 388.0 | 2256.7 | 4314.7 |
| qwen3_5_397b | 212.0 | 342.8 | 377.4 | 2183.2 | 3760.3 |
| gpt_oss_120b | 318.8 | 286.7 | 298.0 | 436.0 | 2578.8 |

LL, nccl_ep vs nixl_ep:

| shape | tok/rank | nccl_ep | nixl_ep | nixl vs nccl |
|---|---|---|---|---|
| deepseek_v4_flash | 8 | 113.0 | 127.8 | 0.88x |
| deepseek_v4_flash | 64 | 144.2 | 87.2 | 1.65x |
| deepseek_v4_flash | 512 | 186.9 | 165.0 | 1.13x |
| deepseek_v4_pro | 8 | 124.4 | 82.7 | 1.50x |
| deepseek_v4_pro | 64 | 599.4 | 98.4 | 6.09x |
| deepseek_v4_pro | 512 | 239.0 | 258.6 | 0.92x |
| deepseek_v3 | 8 | 119.6 | 82.4 | 1.45x |
| deepseek_v3 | 64 | 560.4 | 100.9 | 5.55x |
| deepseek_v3 | 512 | 273.8 | 268.1 | 1.02x |
| kimi_k2_6 | 8 | 117.6 | 85.8 | 1.37x |
| kimi_k2_6 | 64 | 582.0 | 140.9 | 4.13x |
| kimi_k2_6 | 512 | 275.9 | 301.9 | 0.91x |
| qwen3_5_397b | 8 | — | 84.4 | — |
| qwen3_5_397b | 64 | — | 140.7 | — |
| qwen3_5_397b | 512 | — | 226.0 | — |

#### 3.2b nvfp4 trtllm (`fused_moe` split kernel, trtllm-gen backend)

HT only (measured before the LL harness existed); trails the cutedsl
backend at every point.

HT (nccl_ep):

| shape \ tok/rank | 8 | 64 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| deepseek_v4_flash | 1642.2 | 1922.1 | 1643.1 | 4811.5 | 11777.9 |
| deepseek_v4_pro | 2636.0 | 2478.1 | 2526.9 | 7631.7 | 22681.1 |
| deepseek_v3 | 2122.1 | 1831.6 | 2277.2 | 7186.4 | 21387.2 |
| kimi_k2_6 | 2419.1 | 1879.3 | 2347.9 | 7221.1 | 21391.3 |
| qwen3_5_397b | 1594.5 | 1370.1 | 2233.8 | 4867.8 | 12108.0 |

#### 3.2c nvfp4 cutedsl (`fused_moe` split kernel, CuTeDSL backend)

The fastest split kernel measured. At LL @8 tok/rank the nixl comm advantage
carries a 6-13% e2e lead; at 512 compute dominates (>90% of e2e) and the
transports converge.

HT (nccl_ep):

| shape \ tok/rank | 8 | 64 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| deepseek_v4_flash | 906.0 | 912.2 | 1177.2 | 4096.7 | 9414.1 |
| deepseek_v4_pro | 1370.5 | 1540.7 | 1853.7 | 6310.7 | 19928.8 |
| deepseek_v3 | 1624.7 | 1485.8 | 1698.7 | 6083.9 | 19138.8 |
| kimi_k2_6 | 1386.6 | 1466.4 | 1745.3 | 6110.3 | 19226.5 |
| qwen3_5_397b | 884.6 | 961.2 | 1193.6 | 4057.1 | 9784.6 |

LL, nccl_ep vs nixl_ep:

| shape | tok/rank | nccl_ep | nixl_ep | nixl vs nccl |
|---|---|---|---|---|
| deepseek_v4_flash | 8 | 683.0 | 596.1 | 1.15x |
| deepseek_v4_flash | 64 | 845.6 | 787.0 | 1.07x |
| deepseek_v4_flash | 512 | 2910.6 | 2883.9 | 1.01x |
| deepseek_v4_pro | 8 | 851.3 | 807.1 | 1.05x |
| deepseek_v4_pro | 64 | 1613.3 | 1678.5 | 0.96x |
| deepseek_v4_pro | 512 | 9480.9 | 9506.8 | 1.00x |
| deepseek_v3 | 8 | 688.1 | 628.1 | 1.10x |
| deepseek_v3 | 64 | 1370.2 | 1024.4 | 1.34x |
| deepseek_v3 | 512 | 4802.0 | 4757.2 | 1.01x |
| kimi_k2_6 | 8 | 710.6 | 664.8 | 1.07x |
| kimi_k2_6 | 64 | 1560.5 | 1307.2 | 1.19x |
| kimi_k2_6 | 512 | 7103.0 | 6971.7 | 1.02x |
| qwen3_5_397b | 8 | — | 593.5 | — |
| qwen3_5_397b | 64 | — | 838.4 | — |
| qwen3_5_397b | 512 | — | 3180.1 | — |

#### 3.2d w4a8 (`sm100_mxfp8_mxfp4_bf16_cutedsl`)

MXFP8 activations x MXFP4 weights, default kernel tactic (untuned — tracks
nvfp4 cutedsl at 8-64 tok/rank, falls behind at 512+). `w4a8 packed`
(`mxfp8_dispatch=True`) quantizes BEFORE dispatch and sends the packed
fp8+UE8M0 payload — bit-identical output, nccl_ep only. Packed matches
unpacked within noise at >=512 tok/rank under HT (16% faster dispatch stage
at 8192: 695 → 585 µs on v4_flash) and trails at small batch under both
protocols: the pre-dispatch quantize adds ~100-160 µs of launch overhead vs
tens of µs saved on a single-node wire.

HT (nccl_ep):

| shape \ tok/rank | 8 | 64 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| deepseek_v4_flash | 928.4 | 968.4 | 1410.0 | 4949.4 | 13437.1 |
| deepseek_v4_pro | 1424.5 | 1716.3 | 2326.0 | 8096.2 | 27969.3 |
| deepseek_v3 | 1450.6 | 1560.1 | 2124.9 | 7755.7 | 26416.9 |
| kimi_k2_6 | 1451.9 | 1622.5 | 2150.4 | 7711.8 | 26691.1 |
| qwen3_5_397b | 937.9 | 999.0 | 1422.0 | 4933.3 | 13321.3 |

HT packed (nccl_ep):

| shape \ tok/rank | 8 | 64 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|
| deepseek_v4_flash | 1042.7 | 1093.7 | 1484.7 | 4880.4 | 13365.1 |
| deepseek_v4_pro | 1959.0 | 2247.7 | 2356.6 | 7907.6 | 27637.4 |
| deepseek_v3 | 1939.8 | 2049.4 | 2159.7 | 7584.7 | 26429.0 |
| kimi_k2_6 | 1906.0 | 2128.6 | 2182.5 | 7625.9 | 26676.4 |
| qwen3_5_397b | 1042.2 | 1100.3 | 1487.4 | 4842.1 | 13232.3 |

LL, nccl_ep vs nixl_ep:

| shape | tok/rank | nccl_ep | nixl_ep | nixl vs nccl |
|---|---|---|---|---|
| deepseek_v4_flash | 8 | 742.5 | 666.6 | 1.11x |
| deepseek_v4_flash | 64 | 974.9 | 906.0 | 1.08x |
| deepseek_v4_flash | 512 | 4063.7 | 4037.5 | 1.01x |
| deepseek_v4_pro | 8 | 926.1 | 871.3 | 1.06x |
| deepseek_v4_pro | 64 | 2003.6 | 1981.6 | 1.01x |
| deepseek_v4_pro | 512 | 14063.7 | 13937.1 | 1.01x |
| deepseek_v3 | 8 | 756.1 | 696.4 | 1.09x |
| deepseek_v3 | 64 | 1549.8 | 1162.8 | 1.33x |
| deepseek_v3 | 512 | 6648.3 | 6498.1 | 1.02x |
| kimi_k2_6 | 8 | 803.2 | 762.2 | 1.05x |
| kimi_k2_6 | 64 | 1618.4 | 1561.5 | 1.04x |
| kimi_k2_6 | 512 | 9961.7 | 9855.9 | 1.01x |
| qwen3_5_397b | 8 | — | 668.4 | — |
| qwen3_5_397b | 64 | — | 974.6 | — |
| qwen3_5_397b | 512 | — | 4220.3 | — |

LL packed (nccl_ep only):

| shape | tok/rank | nccl_ep |
|---|---|---|
| deepseek_v4_flash | 8 | 840.7 |
| deepseek_v4_flash | 64 | 1073.0 |
| deepseek_v4_flash | 512 | 4457.2 |
| deepseek_v4_pro | 8 | 1010.7 |
| deepseek_v4_pro | 64 | 2066.6 |
| deepseek_v4_pro | 512 | 14523.2 |
| deepseek_v3 | 8 | 840.5 |
| deepseek_v3 | 64 | 1775.6 |
| deepseek_v3 | 512 | 7251.3 |
| kimi_k2_6 | 8 | 899.5 |
| kimi_k2_6 | 64 | 1792.9 |
| kimi_k2_6 | 512 | 10836.0 |

### 3.3 2xB200 EP2 baseline — quantized speedup vs bf16

Externally provided reference run (no SLURM job ID), same harness
(`bench_moe_ep_mega.py`) on **2x B200, EP=DP=2, TP=1**, `deepseek_v3` geometry
(256 experts, top-8, hidden 7168, inter 2048), warmup 20 / iters 50, CUDA-event
p50 µs, staging + weight preprocess excluded, MEGA_ACC=1. This is the framing
the bf16 kernel is for: bf16 as the unquantized baseline, quantized kernels
reported as speedup over it.

**TOKENS=8, MEGA_TIMING=e2e (barrier-cold):**

| backend | p50 (µs) | min | max | tok/s | acc_loss % |
|---|---|---|---|---|---|
| bf16_cutedsl | 1322.1 | 1308.9 | 1470.1 | 12102 | 0.286 |
| mxfp8_cutedsl | 557.3 | 547.0 | 705.4 | 28708 | 6.371 |
| nvfp4_cutedsl | 326.2 | 322.5 | 581.8 | 49044 | 23.324 |

**TOKENS=8, MEGA_TIMING=kernel (tester-parity bare launch):**

| backend | p50 (µs) | min | max | tok/s | acc_loss % |
|---|---|---|---|---|---|
| bf16_cutedsl | 1279.0 | 1262.6 | 1312.0 | 12510 | 0.286 |
| mxfp8_cutedsl | 533.5 | 521.2 | 568.2 | 29993 | 6.371 |
| nvfp4_cutedsl | 289.8 | 287.6 | 313.3 | 55218 | 23.324 |

**Sweep, MEGA_TIMING=e2e_pipelined** — p50 µs, with the quantized backends'
speedup vs bf16 in brackets:

| tok/rank | bf16 | mxfp8 (vs bf16) | nvfp4 (vs bf16) |
|---|---|---|---|
| 8 | 1290.2 | 535.6 (2.41x) | 293.9 (4.39x) |
| 64 | 2675.3 | 1075.3 (2.49x) | 529.3 (5.05x) |
| 256 | 2749.5 | 1139.7 (2.41x) | 551.9 (4.98x) |
| 1024 | 2940.4 | 1285.2 (2.29x) | 656.3 (4.48x) |
| 4096 | 4493.4 | 2259.0 (1.99x) | 1139.1 (3.94x) |

Throughput (tok/s):

| tok/rank | bf16 | mxfp8 | nvfp4 |
|---|---|---|---|
| 8 | 12402 | 29876 | 54448 |
| 64 | 47846 | 119033 | 241809 |
| 256 | 186219 | 449230 | 927778 |
| 1024 | 696515 | 1593506 | 3120505 |
| 4096 | 1823122 | 3626345 | 7191516 |

Accuracy loss (% rel-L2 vs bf16 dense reference) is flat across tokens/rank:
bf16 0.286-0.288, mxfp8 6.357-6.371, nvfp4 23.144-23.324.

**How this compares to the §3.1 EP8 runs** (same `deepseek_v3` geometry; shared
tokens/rank points are 8 and 64):

* **Accuracy reproduces exactly across world sizes**: bf16 0.288 / mxfp8 ~6.36
  / nvfp4 ~23.2 on both EP2 and EP8 — same math, different node counts.
* **The quantized-vs-bf16 ratio also carries over**: mxfp8/bf16 is 2.4x at
  EP2 and 2.3-2.4x at EP8 (633.9/275.5 at 8 tok/rank, 732.3/306.2 at 64),
  decaying toward ~2.0x at the largest batch on both.
* **Absolute latencies do not carry over — EP2 is 2.0x slower at 8 tok/rank
  (bf16 1290.2 vs 633.9; mxfp8 535.6 vs 275.5) and ~3.6x slower at 64
  (2675.3 vs 732.3; 1075.3 vs 306.2).** Per-rank routed work is identical
  (tokens/rank x top_k is world-size independent), but each EP2 rank holds
  4x the experts (128 vs 32), so at small batch — where these kernels are
  weight-bandwidth bound — per-rank weight bytes dominate and scale with
  1/world_size. Compare EP2 numbers to EP2 numbers only; the transferable
  quantities are the ratios and the accuracy losses, not the microseconds.

## 4. Accuracy gate — GSM8K, both checkpoints

`sbatch vllm_e2e/job_gsm8k_flash_pro.sh` (~35 min, both models, both at TP8).

| model | native | fi_dg | fi_cutedsl (NVFP4 cast) | delta |
|---|---|---|---|---|
| Flash | 0.965 | 0.965 | **0.965** | +0.000 |
| Pro | 0.880 | 0.880 | **0.890** | +0.010 |

**The delta is the number that gates a perf claim.** Because fi_cutedsl runs a
different checkpoint, its throughput is only comparable if its accuracy is —
±0.010 on 200 questions is 2 questions, i.e. noise. Both models pass.

**Do not read Pro's 0.880 as a regression.** All three backends agree exactly
(176/176/178 correct; Flash is 193/193/193), so it is a property of the model
and this eval, not of
`moe_ep`. It is not a truncation artifact either: raising `--max-tokens`
512 -> 1024 -> 2048 moves accuracy 0.8800 -> 0.8750 -> 0.8750 while truncated
completions only fall 15 -> 14 -> 13, i.e. a handful never terminate at any
budget. `--min-acc 0.93` is calibrated for Flash; for Pro it will fail and that
failure is expected.

---

## 5. The two ways this has actually gone wrong

Both produced *plausible wrong numbers* rather than errors, which is why they
are documented rather than merely fixed.

### 5.1 A cell that sets `MAX_CAPTURE` without pinning `CAPTURE_SIZES`

The dense default capture ladder makes vLLM's CUDA-graph memory profiler
reserve **~48 GiB/GPU** for the flashinfer backends against a real capture cost
of ~6 GiB — the same ~6 GiB it estimates correctly for native. The phantom
reservation is taken out of the KV cache. Measured by running the V4-Pro
decode cell unpinned on purpose — these rows are not in `results/`, since the
shipped cells all pin `CAPTURE_SIZES`:

| backend | KV available | KV tokens | resident seqs | tok/s |
|---|---|---|---|---|
| native | 48.91 GiB | 95,979 | 1024 of 1024 | 13268 |
| fi_dg | 7.28 GiB | 14,286 | **189** | 5969 (0.45x) |
| fi_cutedsl | 0.07 GiB | — | engine will not start | OOM |

**The tell is a backend that is fast per step and slow overall.** fi_dg's ITL
was *better* than native (42.6 vs 94.4 ms) precisely because its batches were
5x smaller — it reads as a good kernel on a starved engine. Check
`Available KV cache memory` and the scheduler's `Running:`/`Waiting:` counts
before blaming a kernel.

Severity scales with how little KV headroom the model leaves: Pro EP8 held 189
of 1024 requested sequences, Flash EP8 520 — the bigger the weights, the harder
it bites, and on a small enough model it hides entirely. All shipped cells pin
`CAPTURE_SIZES`; doing so costs native ~3% to batch padding, which is already
reflected in §1 and §2.

### 5.2 Exporting `MODEL` around the GSM8K gate

`resolve_model` ranks `--model` > `$MODEL` > per-backend default. A `MODEL=`
in the environment therefore sends *every* backend to that checkpoint,
including fi_cutedsl — the gate then compares the mx weights against
themselves, scores a comfortable pass, and validates nothing — the fi_cutedsl
row carries the mx checkpoint under an NVFP4 label.

`job_gsm8k_flash_pro.sh` passes `--model` per cell and unsets `MODEL` inside
the container. **Check the `model` field in each result JSON** — it records
what was actually loaded, and the job's summary flags any fi_cutedsl row that
did not run an nvfp4 checkpoint.

---

## 6. Tolerances

Ratios are stable to about ±0.02x between sessions; absolute throughput moves
more with node and thermal state. Run all three backends of a cell **in one
session** — native's decode drifts round-over-round, so cross-session ratios
are not trustworthy. GSM8K on 200 questions has a granularity of 0.005, so
treat anything inside ±0.02 as agreement.

The ±0.02x above is measured, not assumed: repeating the whole set on a second
pass reproduced it to within **0.5% on absolute throughput and 0.008x on every
ratio**. Every table was produced by this branch's own scripts against a freshly
built venv.
