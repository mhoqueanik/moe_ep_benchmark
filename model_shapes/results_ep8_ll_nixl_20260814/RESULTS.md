# Model-shape microbenchmark results

e2e_pipelined p50 latency in microseconds per variant; fp4-family cells show speedup vs `dg` at the same point. `tok/s` and accuracy-loss columns are in the raw CSVs.

## deepseek_v3 — hidden 7168, inter 2048, 256 experts, top-8

| tok/rank | split nvfp4 cutedsl | split w4a8 | split w4a8 packed | nccl identity | nixl identity | nixl split nvfp4 | nixl split w4a8 |
|---|---|---|---|---|---|---|---|
| 8 | 688.1 | 756.1 | 840.5 | 119.6 | 82.4 | 628.1 | 696.4 |
| 64 | 1370.2 | 1549.8 | 1775.6 | 560.4 | 100.9 | 1024.4 | 1162.8 |
| 512 | 4802.0 | 6648.3 | 7251.3 | 273.8 | 268.1 | 4757.2 | 6498.1 |

## deepseek_v4_flash — hidden 4096, inter 2048, 256 experts, top-6

| tok/rank | split nvfp4 cutedsl | split w4a8 | split w4a8 packed | nccl identity | nixl identity | nixl split nvfp4 | nixl split w4a8 |
|---|---|---|---|---|---|---|---|
| 8 | 683.0 | 742.5 | 840.7 | 113.0 | 127.8 | 596.1 | 666.6 |
| 64 | 845.6 | 974.9 | 1073.0 | 144.2 | 87.2 | 787.0 | 906.0 |
| 512 | 2910.6 | 4063.7 | 4457.2 | 186.9 | 165.0 | 2883.9 | 4037.5 |

## deepseek_v4_pro — hidden 7168, inter 3072, 384 experts, top-6

| tok/rank | split nvfp4 cutedsl | split w4a8 | split w4a8 packed | nccl identity | nixl identity | nixl split nvfp4 | nixl split w4a8 |
|---|---|---|---|---|---|---|---|
| 8 | 851.3 | 926.1 | 1010.7 | 124.4 | 82.7 | 807.1 | 871.3 |
| 64 | 1613.3 | 2003.6 | 2066.6 | 599.4 | 98.4 | 1678.5 | 1981.6 |
| 512 | 9480.9 | 14063.7 | 14523.2 | 239.0 | 258.6 | 9506.8 | 13937.1 |

## kimi_k2_6 — hidden 7168, inter 2048, 384 experts, top-8

| tok/rank | split nvfp4 cutedsl | split w4a8 | split w4a8 packed | nccl identity | nixl identity | nixl split nvfp4 | nixl split w4a8 |
|---|---|---|---|---|---|---|---|
| 8 | 710.6 | 803.2 | 899.5 | 117.6 | 85.8 | 664.8 | 762.2 |
| 64 | 1560.5 | 1618.4 | 1792.9 | 582.0 | 140.9 | 1307.2 | 1561.5 |
| 512 | 7103.0 | 9961.7 | 10836.0 | 275.9 | 301.9 | 6971.7 | 9855.9 |

## qwen3_5_397b — hidden 4096, inter 1024, 512 experts, top-10

| tok/rank | split nvfp4 cutedsl | split w4a8 | split w4a8 packed | nccl identity | nixl identity | nixl split nvfp4 | nixl split w4a8 |
|---|---|---|---|---|---|---|---|
| 8 | — | — | — | — | 84.4 | 593.5 | 668.4 |
| 64 | — | — | — | — | 140.7 | 838.4 | 974.6 |
| 512 | — | — | — | — | 226.0 | 3180.1 | 4220.3 |

