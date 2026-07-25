"""Config-level checks for the flashinfer_moe_ep_mega_* backend strings.

Runs inside the benchmark container against the patched vLLM, but loads no
model and allocates no MoE layer, so it is seconds rather than the ~30 min a
real smoke costs. Covers exactly what the backend-string switch introduced:
the KernelConfig registration that `patch_0251/apply.sh` inserts, the
FI_MOE_EP_BACKENDS table, and the three rejections in
`validate_fi_moe_ep_config` (retired env vars, EPLB, arch floor).

    python test_backend_registration.py

Exits non-zero on the first failure, with the check name.
"""

from __future__ import annotations

import os
import sys
import traceback

BACKENDS = (
    "flashinfer_moe_ep_mega_deep_gemm",
    "flashinfer_moe_ep_mega_cutedsl",
)
NATIVE = "deep_gemm_mega_moe"

_failures: list[str] = []
_passed = 0

# Checks run at decoration time, so print the banner before any of them.
print(f"python {sys.version.split()[0]}")

# validate_fi_moe_ep_config() calls the flashinfer version gate, which this
# venv is deliberately below. Skip it globally so the other checks test what
# they are named for; the gate itself has its own check, which unsets this.
os.environ["FI_MOE_EP_SKIP_VERSION_CHECK"] = "1"


def check(name: str):
    """Decorator: run the function, record pass/fail, never abort the suite."""

    def wrap(fn):
        global _passed
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failure
            _failures.append(name)
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
        else:
            _passed += 1
            print(f"ok    {name}")
        return fn

    return wrap


def expect_raises(exc_types, fn, *, contains: str | None = None) -> None:
    try:
        fn()
    except exc_types as exc:
        if contains and contains not in str(exc):
            raise AssertionError(
                f"raised {type(exc).__name__} but message lacked {contains!r}: {exc}"
            ) from None
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_types}, got {type(exc).__name__}: {exc}"
        ) from None
    else:
        raise AssertionError(f"expected {exc_types}, nothing raised")


# --------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------


@check("flashinfer.moe_ep runtime importable")
def _fi_runtime():
    from flashinfer.moe_ep import (  # noqa: F401
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )


@check("vllm imports and reports a version")
def _vllm():
    import vllm

    print(f"      vllm {vllm.__version__}")


# --------------------------------------------------------------------------
# KernelConfig registration (what apply.sh inserts into kernel.py)
# --------------------------------------------------------------------------


@check("MoEBackend literal contains all three backends")
def _literal():
    from typing import get_args

    from vllm.config.kernel import MoEBackend

    members = set(get_args(MoEBackend))
    missing = [b for b in BACKENDS if b not in members]
    assert not missing, f"not registered in MoEBackend: {missing}"
    assert NATIVE in members, "native deep_gemm_mega_moe disappeared"


@check("KernelConfig validates each backend string")
def _kernelconfig_accepts():
    from vllm.config.kernel import KernelConfig

    for b in BACKENDS:
        cfg = KernelConfig(moe_backend=b)
        assert cfg.moe_backend == b, f"{b} round-tripped as {cfg.moe_backend}"


@check("KernelConfig still rejects an unknown backend")
def _kernelconfig_rejects():
    from vllm.config.kernel import KernelConfig

    expect_raises(Exception, lambda: KernelConfig(moe_backend="not_a_real_backend"))


@check("KernelConfig normalises dashes and case")
def _kernelconfig_normalises():
    from vllm.config.kernel import KernelConfig

    dashed = BACKENDS[1].replace("_", "-").upper()
    assert KernelConfig(moe_backend=dashed).moe_backend == BACKENDS[1]


# --------------------------------------------------------------------------
# fi_utils backend table
# --------------------------------------------------------------------------


@check("every registered backend has a spec, and specs are self-consistent")
def _table():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    assert set(fi_utils.FI_MOE_EP_BACKENDS) == set(BACKENDS), (
        f"table {sorted(fi_utils.FI_MOE_EP_BACKENDS)} != expected {sorted(BACKENDS)}"
    )
    assert fi_utils.FI_MOE_EP_MIN_CAPABILITY == (10, 0), (
        f"arch floor moved: {fi_utils.FI_MOE_EP_MIN_CAPABILITY}"
    )
    for name, spec in fi_utils.FI_MOE_EP_BACKENDS.items():
        # deep_gemm needs only torch.distributed; the cutedsl kernels need NVSHMEM
        assert spec.needs_nvshmem == ("cutedsl" in name), (
            f"{name}: needs_nvshmem={spec.needs_nvshmem}"
        )
        assert fi_utils.fi_moe_ep_backend_spec(name) is spec
        assert fi_utils.fi_spec_for_megakernel(spec.megakernel) is spec


@check("runtime requirements: NVSHMEM only for the cutedsl kernels")
def _requirements():
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST
    from vllm.utils import flashinfer_moe_ep as fi_utils

    for name, spec in fi_utils.FI_MOE_EP_BACKENDS.items():
        reqs = fi_utils.megakernel_runtime_requirements(spec)
        assert TORCH_DIST in reqs, f"{name} lost TORCH_DIST"
        assert (NVSHMEM in reqs) == spec.needs_nvshmem, f"{name}: {reqs}"


@check("predicates: fi backends are mega, native is mega but not fi")
def _predicates():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    for b in BACKENDS:
        assert fi_utils.is_mega_moe_backend(b), f"{b} not treated as mega"
        assert fi_utils.is_fi_moe_ep_backend(b), f"{b} not treated as fi"
    assert fi_utils.is_mega_moe_backend(NATIVE), "native lost its mega path"
    assert not fi_utils.is_fi_moe_ep_backend(NATIVE), "native hijacked onto fi"
    for other in ("triton", "cutlass", "auto"):
        assert not fi_utils.is_mega_moe_backend(other)
        assert not fi_utils.is_fi_moe_ep_backend(other)


@check("unknown megakernel name is rejected")
def _unknown_megakernel():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    expect_raises(ValueError, lambda: fi_utils.fi_spec_for_megakernel("nope_cutedsl"))
    expect_raises(ValueError, lambda: fi_utils.fi_moe_ep_backend_spec(NATIVE))


# --------------------------------------------------------------------------
# validate_fi_moe_ep_config
# --------------------------------------------------------------------------


def _stub_config(backend: str, *, enable_eplb: bool = False):
    """Minimal stand-in: validate_fi_moe_ep_config reads only these two fields."""
    from types import SimpleNamespace

    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=backend),
        parallel_config=SimpleNamespace(enable_eplb=enable_eplb),
    )


@check("a valid fi config passes validation on this device")
def _validate_ok():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    for var in ("FI_MOE_EP", "FI_MOE_EP_MEGAKERNEL"):
        os.environ.pop(var, None)
    for b in BACKENDS:
        fi_utils.validate_fi_moe_ep_config(_stub_config(b))


@check("retired FI_MOE_EP / FI_MOE_EP_MEGAKERNEL are rejected")
def _validate_retired_env():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    for var in ("FI_MOE_EP", "FI_MOE_EP_MEGAKERNEL"):
        for value in ("1", "0", "deep_gemm_mega"):
            os.environ[var] = value
            try:
                # Must fire for the native backend too -- that is the whole
                # point: a stale export used to silently mean "run fi".
                expect_raises(
                    ValueError,
                    lambda: fi_utils.validate_fi_moe_ep_config(_stub_config(NATIVE)),
                    contains=var,
                )
            finally:
                os.environ.pop(var, None)


@check("EPLB is rejected for fi backends but allowed for native")
def _validate_eplb():
    from vllm.utils import flashinfer_moe_ep as fi_utils

    for b in BACKENDS:
        expect_raises(
            NotImplementedError,
            lambda b=b: fi_utils.validate_fi_moe_ep_config(
                _stub_config(b, enable_eplb=True)
            ),
            contains="EPLB",
        )
    # native + EPLB must still be a supported combination
    fi_utils.validate_fi_moe_ep_config(_stub_config(NATIVE, enable_eplb=True))


@check("flashinfer version gate rejects builds below the floor")
def _version_gate():
    from vllm.utils import flashinfer_moe_ep as fi

    floor = fi.FI_MOE_EP_MIN_FLASHINFER
    found = fi._flashinfer_version()
    shown = ".".join(map(str, found)) if found else "undeterminable"
    print(f"      flashinfer {shown}, floor {'.'.join(map(str, floor))}")
    os.environ.pop("FI_MOE_EP_SKIP_VERSION_CHECK", None)
    try:
        if found is not None and found < floor:
            # Below the floor: must refuse, and the escape hatch must work.
            expect_raises(
                ValueError,
                fi.check_flashinfer_version,
                contains=".".join(map(str, floor)),
            )
            os.environ["FI_MOE_EP_SKIP_VERSION_CHECK"] = "1"
            fi.check_flashinfer_version()
        else:
            # At or above the floor (or undeterminable): must not refuse.
            fi.check_flashinfer_version()
    finally:
        os.environ["FI_MOE_EP_SKIP_VERSION_CHECK"] = "1"


@check("arch floor is enforced against the real device capability")
def _validate_arch():
    import torch
    from vllm.utils import flashinfer_moe_ep as fi_utils

    if not torch.cuda.is_available():
        raise AssertionError("no CUDA device visible - run this on a GPU node")
    cc = torch.cuda.get_device_capability()
    print(f"      device capability sm_{cc[0]}{cc[1]}")
    floor = fi_utils.FI_MOE_EP_MIN_CAPABILITY
    assert cc >= floor, (
        f"this node is sm_{cc[0]}{cc[1]} but the backends declare a floor of "
        f"sm_{floor[0]}{floor[1]} -- so _validate_ok above should have failed. "
        "Re-run on Blackwell to exercise the passing path."
    )


def main() -> int:
    total = _passed + len(_failures)
    print(f"\n{_passed}/{total} checks passed")
    if _failures:
        print("failed: " + ", ".join(_failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
