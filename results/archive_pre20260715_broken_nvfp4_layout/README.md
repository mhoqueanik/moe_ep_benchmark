# Archived: pre-2026-07-15 benchmark results (broken nvfp4 weight layout)

All `nvfp4_cutedsl` rows in these CSVs were measured while the FlashInfer
preprocess emitted N-stride-1 fc1/fc2 weights; the kernel compiled its TMA
layout from those strides, so the runs were faster than the correct K-major
layout but numerically wrong (caught by the moe_ep torch-oracle tests on
2026-07-15; fixed in backends/mega/kernel/nvfp4_cutedsl/weights.py).

deep_gemm_mega / mxfp8_cutedsl rows are unaffected (re-verified to reproduce),
but the files are archived wholesale to keep only trustworthy data in
results/.  Corrected sweeps: resweep logs/CSVs dated 2026-07-15 onward.
