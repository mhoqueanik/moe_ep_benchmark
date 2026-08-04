#!/bin/bash
# Build MoK (SM100) in-tree and run the forward-only (inference) benchmark.
set -euo pipefail
cd /lustre/fsw/coreai_libraries_cudnn/mhoqueanik/mixture-of-kittens

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | tail -1
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT=$PWD/mok/_C$EXT_SUFFIX

echo "=== build (ARCH=SM100, OUT=$OUT) ==="
time make ARCH=SM100 PYTHON=python OUT="$OUT"

echo "=== import check ==="
PYTHONPATH=$PWD python -c "import mok; print('mok', mok.__version__)"

echo "=== forward-only benchmark (4 GPUs) ==="
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 -m benchmarks.bench_mok_fwd_only
echo "=== DONE ==="
