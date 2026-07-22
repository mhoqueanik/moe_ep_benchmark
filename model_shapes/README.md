# Model-shape microbenchmark sweep

Runs the fi mega MoE-EP variants across the MoE geometries of real models —
the model list mirrors the
[cudnn-frontend SDPA training benchmark](https://github.com/NVIDIA/cudnn-frontend/tree/develop/benchmark/sdpa_benchmark_training#benchmark-results)
(MoE-capable models only; Llama 3.1 and the video-DiT models are dense).

## Shapes (`shapes.tsv`)

| name | hidden | moe_inter | experts | top-k |
|---|---|---|---|---|
| deepseek_v3 | 7168 | 2048 | 256 | 8 |
| kimi_k2_6 | 7168 | 2048 | 384 | 8 |
| gpt_oss_120b | 2880 | 2880 | 128 | 4 |
| qwen3_5_397b | 4096 | 1024 | 512 | 10 |
| deepseek_v4_flash | 4096 | 2048 | 256 | 6 |
| deepseek_v4_pro | 7168 | 3072 | 384 | 6 |

Geometries verified against local checkpoint configs
(`/lustre/share/coreai_dlalgo_ci/artifacts/model/`) and official HF configs
(2026-07-21).

## Variants

| variant | backend | env |
|---|---|---|
| fi_dg | deep_gemm_mega | — |
| fi_fp4 | nvfp4_cutedsl | — |
| fi_ikr | nvfp4_cutedsl | `MEGA_IKR=1` |
| fi_combine_fp8 | nvfp4_cutedsl | `MEGA_COMBINE_DTYPE=mxfp8` |
| fi_combine_fp4 | nvfp4_cutedsl | `MEGA_COMBINE_DTYPE=nvfp4` |

Timed region defaults to `MEGA_TIMING=e2e_pipelined` (methodology of the
2026-07-15 corrected tables in `kernel_src/cutedsl_megamoe/TUNING.md`).
Default token points: 8 64 512 2048 8192 tokens/rank, 4 GPUs (DP=EP=4).

## Run

```bash
bash submit_jobs.sh                 # one 4h job per shape, parallel nodes
squeue -u $USER                     # watch
python make_tables.py results/model_shapes_*.csv   # -> RESULTS.md
```

Missing cells (job timeout, unsupported shape/kernel combo) are warn-continue;
resubmit that shape (`SHAPE_LIST=<name> bash submit_jobs.sh`) — CuTe-DSL
compiles are cached under `~/.cache/flashinfer`, so re-runs are cheap, and
`make_tables.py` merges CSVs with later files winning.

Compile-cost note: first-time `cute.compile` scales with experts/rank
(~12 min per kernel config at 64 experts/rank; `qwen3_5_397b` at 128/rank is
the worst case and may need a resubmit to finish all variants).
