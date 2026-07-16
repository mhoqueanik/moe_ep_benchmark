# CuTe-DSL 4.6 compat for vLLM 0.25.1 vendored kernels (installed by
# setup_container.sh as <venv>/lib/python3.12/site-packages/sitecustomize.py).
#
# vLLM 0.25.1 pins nvidia-cutlass-dsl==4.5.2; this venv runs 4.6.1 because the
# flashinfer cutedsl mega kernels are 34-54% slower on 4.5.2. In 4.6 ThrMma
# moved from cutlass.cute.core to cutlass.cute (cute/atom.py); vLLM's vendored
# vllm_flash_attn/cute + third_party/fmha_sm100 still reference the old path.
# Alias it back on first import of cutlass.cute.core.
import importlib
import importlib.abc
import sys


class _CuteCoreCompatLoader(importlib.abc.Loader):
    def __init__(self, orig_loader):
        self._orig = orig_loader

    def create_module(self, spec):
        return self._orig.create_module(spec) if hasattr(self._orig, "create_module") else None

    def exec_module(self, module):
        self._orig.exec_module(module)
        if module.__name__ == "cutlass.cute.core" and not hasattr(module, "ThrMma"):
            try:
                cute = importlib.import_module("cutlass.cute")
                if hasattr(cute, "ThrMma"):
                    module.ThrMma = cute.ThrMma
            except Exception:
                pass


class _CuteCoreCompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "cutlass.cute.core":
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _CuteCoreCompatFinder):
                continue
            spec = getattr(finder, "find_spec", lambda *a, **k: None)(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _CuteCoreCompatLoader(spec.loader)
                return spec
        return None


sys.meta_path.insert(0, _CuteCoreCompatFinder())
