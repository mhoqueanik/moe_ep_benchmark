# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` helpers for DeepSeek V4 vLLM integration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.distributed import get_ep_group

if TYPE_CHECKING:
    from flashinfer.moe_ep import BootstrapConfig, MoEEpMegaLayer
    from vllm.config import VllmConfig

DeepseekV4MegaMoEExpertsFI: type | None = None

MEGA_MOE_BACKENDS = frozenset({"deep_gemm_mega_moe", "fi_moe_ep"})
FI_MEGA_KERNELS = frozenset({"deep_gemm_mega", "nvfp4_cutedsl", "mxfp8_cutedsl"})

_FI_RUNTIME_HANDLE: Any = None
_FI_MOE_EP_RUNTIME_AVAILABLE: bool | None = None


def _has_fi_moe_ep_runtime() -> bool:
    """True when the installed flashinfer exposes the moe_ep runtime helpers."""
    global _FI_MOE_EP_RUNTIME_AVAILABLE
    if _FI_MOE_EP_RUNTIME_AVAILABLE is not None:
        return _FI_MOE_EP_RUNTIME_AVAILABLE
    try:
        from flashinfer.moe_ep import (  # noqa: F401
            bootstrap_moe_ep_runtime,
            ensure_moe_ep_cuda_device,
            finalize_moe_ep_runtime,
        )
    except ImportError:
        _FI_MOE_EP_RUNTIME_AVAILABLE = False
    else:
        _FI_MOE_EP_RUNTIME_AVAILABLE = True
    return _FI_MOE_EP_RUNTIME_AVAILABLE


def is_mega_moe_backend(moe_backend: str) -> bool:
    return moe_backend in MEGA_MOE_BACKENDS


def is_fi_moe_ep_backend(moe_backend: str) -> bool:
    # ``deep_gemm_mega_moe`` stays the only KernelConfig backend string; the
    # flashinfer moe_ep compute path is opted into with FI_MOE_EP=1 so a single
    # install can A/B the native vs flashinfer mega paths run-by-run.
    if moe_backend == "fi_moe_ep" or (
        moe_backend == "deep_gemm_mega_moe"
        and os.environ.get("FI_MOE_EP", "0").lower() in ("1", "true", "yes")
    ):
        if not _has_fi_moe_ep_runtime():
            raise ImportError(
                "FI_MOE_EP=1 requires flashinfer.moe_ep runtime support "
                "(install the flashinfer moe_ep branch), or unset FI_MOE_EP "
                "to use the native deep_gemm_mega_moe path."
            )
        return True
    return False


def resolve_fi_megakernel(vllm_config: "VllmConfig") -> str:
    """Select the flashinfer mega sub-kernel for ``fi_moe_ep``."""
    kernel_config = vllm_config.kernel_config
    megakernel = getattr(kernel_config, "fi_moe_ep_megakernel", None)
    if megakernel is None:
        megakernel = os.environ.get("FI_MOE_EP_MEGAKERNEL", "deep_gemm_mega")
    megakernel = str(megakernel).lower().replace("-", "_")
    if megakernel not in FI_MEGA_KERNELS:
        raise ValueError(
            f"Unsupported fi_moe_ep megakernel {megakernel!r}; "
            f"expected one of {sorted(FI_MEGA_KERNELS)}"
        )
    return megakernel


def make_fi_moe_ep_bootstrap() -> "BootstrapConfig":
    from flashinfer.moe_ep import BootstrapConfig

    ep = get_ep_group()
    return BootstrapConfig(
        world_size=ep.world_size,
        rank=ep.rank_in_group,
        process_group=ep.device_group,
        auto_bootstrap=False,
    )


def megakernel_runtime_requirements(megakernel: str) -> frozenset[str]:
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    if megakernel == "deep_gemm_mega":
        return frozenset({TORCH_DIST})
    if megakernel in ("nvfp4_cutedsl", "mxfp8_cutedsl"):
        return frozenset({TORCH_DIST, NVSHMEM})
    raise ValueError(f"Unsupported fi_moe_ep megakernel {megakernel!r}")


def ensure_fi_moe_ep_runtime(vllm_config: "VllmConfig") -> None:
    """Acquire the process-wide flashinfer moe_ep runtime once per worker."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is not None:
        return

    from flashinfer.moe_ep import bootstrap_moe_ep_runtime

    if not _has_fi_moe_ep_runtime():
        raise ImportError(
            "flashinfer.moe_ep runtime helpers are not available in this "
            "flashinfer build."
        )

    bootstrap = make_fi_moe_ep_bootstrap()
    megakernel = resolve_fi_megakernel(vllm_config)
    # flashinfer's runtime/layer constructors bind the process to
    # cuda:LOCAL_RANK (falling back to bootstrap.rank). vLLM has already bound
    # this worker to its (possibly remapped) visible device, and a mismatched
    # rebind launches the weight transforms on the wrong GPU against another
    # device's pointers (observed as CUDA_ERROR_ILLEGAL_ADDRESS in the
    # deep_gemm transform_sf during load). Pin LOCAL_RANK to the device vLLM
    # chose so every internal set_device is a no-op.
    os.environ["LOCAL_RANK"] = str(torch.cuda.current_device())
    print(
        f"[fi_moe_ep] ep_rank={bootstrap.rank} world={bootstrap.world_size} "
        f"cuda.current_device={torch.cuda.current_device()} "
        f"megakernel={megakernel}",
        flush=True,
    )
    _FI_RUNTIME_HANDLE = bootstrap_moe_ep_runtime(
        bootstrap,
        megakernel_runtime_requirements(megakernel),
    )


def finalize_fi_moe_ep_runtime() -> None:
    """Release the process-wide flashinfer moe_ep runtime."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is None:
        return

    from flashinfer.moe_ep import finalize_moe_ep_runtime

    finalize_moe_ep_runtime(_FI_RUNTIME_HANDLE)
    _FI_RUNTIME_HANDLE = None


_E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequant_fp4_ue8m0_gran32(
    packed: torch.Tensor, sf_ue8m0: torch.Tensor
) -> torch.Tensor:
    """[rows, K//2] packed e2m1 + [rows, K//32] ue8m0-uint8 scales -> bf16 [rows, K]."""
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    vals[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    sf = (sf_ue8m0.to(torch.int32) << 23).view(torch.float32)
    return (vals * sf.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_expert_weights_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """[E, N, K//2] fp4 + [E, N, K//32] ue8m0 -> [E, N, K] bf16 (expert loop)."""
    num_experts, n, k_half = weight.shape
    out = torch.empty(
        num_experts, n, k_half * 2, dtype=torch.bfloat16, device=weight.device
    )
    for e in range(num_experts):
        out[e] = _dequant_fp4_ue8m0_gran32(weight[e], scale[e])
    return out


def mega_moe_weight_pack_from_params(
    w13_weight: nn.Parameter,
    w13_weight_scale: nn.Parameter,
    w2_weight: nn.Parameter,
    w2_weight_scale: nn.Parameter,
    *,
    megakernel: str = "deep_gemm_mega",
):
    from flashinfer.moe_ep import MoEWeightPack

    if megakernel == "deep_gemm_mega":
        # Same fp4-e2m1 + ue8m0-per-32 recipe as the native path: pass verbatim,
        # flashinfer runs the identical deep_gemm transform.
        return MoEWeightPack(
            w13=w13_weight.data,
            w2=w2_weight.data,
            w13_scale=w13_weight_scale.data,
            w2_scale=w2_weight_scale.data,
        )
    # cutedsl kernels quantize with their own recipe (nvfp4 e2m1+e4m3-per-16 /
    # mxfp8 e4m3-per-32): dequantize the checkpoint fp4 to bf16 and let the
    # backend preprocess requantize. Double quantization: outputs are close to
    # but not bit-identical with the native path.
    return MoEWeightPack(
        w13=_dequant_expert_weights_to_bf16(w13_weight.data, w13_weight_scale.data),
        w2=_dequant_expert_weights_to_bf16(w2_weight.data, w2_weight_scale.data),
    )


def build_fi_mega_config(
    *,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    megakernel: str,
    fast_math: bool = True,
):
    from flashinfer.moe_ep import (
        DeepGemmMegaMoeConfig,
        MegaConfig,
        Mxfp8CutedslMegaMoeConfig,
        Nvfp4CutedslMegaMoeConfig,
    )

    knobs: dict | str | None = None
    knobs_env = os.environ.get("FI_MOE_EP_KNOBS", "")
    if knobs_env:
        if knobs_env.strip().startswith("{"):
            import json

            knobs = json.loads(knobs_env)
        else:
            knobs = knobs_env  # e.g. "auto"

    if megakernel == "deep_gemm_mega":
        mk = DeepGemmMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    elif megakernel == "nvfp4_cutedsl":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            knobs=knobs,
        )
    elif megakernel == "mxfp8_cutedsl":
        mk = Mxfp8CutedslMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            knobs=knobs,
        )
    else:
        raise ValueError(f"Unsupported fi_moe_ep megakernel {megakernel!r}")

    return MegaConfig(
        megakernel=mk,
        preprocess_weights=True,
        quantize_input=True,
    )


def build_fi_mega_layer(
    bootstrap: "BootstrapConfig",
    *,
    vllm_config: "VllmConfig",
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    weights,
    fast_math: bool = True,
) -> "MoEEpMegaLayer":
    from flashinfer.moe_ep import FleetParams, MoEEpLayer

    megakernel = resolve_fi_megakernel(vllm_config)
    mega_config = build_fi_mega_config(
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_clamp=activation_clamp,
        megakernel=megakernel,
        fast_math=fast_math,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=hidden_size,
        ),
        weights=weights,
        backend=mega_config,
    )
    from flashinfer.moe_ep import MoEEpMegaLayer

    if not isinstance(layer, MoEEpMegaLayer):
        raise TypeError(
            f"fi_moe_ep expected MoEEpMegaLayer, got {type(layer).__name__}"
        )
    return layer


_MOE_SKIP_PADDING: bool | None = None


def resolve_mega_moe_is_padding(num_tokens: int) -> torch.Tensor | None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    global _MOE_SKIP_PADDING
    if _MOE_SKIP_PADDING is None:
        import vllm.envs as envs

        _MOE_SKIP_PADDING = bool(envs.VLLM_MOE_SKIP_PADDING)
    if not _MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None:
        return None
    return is_padding[:num_tokens]


def apply_mega_moe_routing_preprocess(
    topk_ids: torch.Tensor,
    *,
    is_padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Padding-only routing preprocess (EPLB hooks go here later)."""
    if is_padding is not None:
        topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
    return topk_ids


def make_fi_mega_moe_experts_cls(mega_moe_experts_cls: type[nn.Module]) -> type[nn.Module]:
    """Build ``DeepseekV4MegaMoEExpertsFI`` once the base mega experts class exists."""
    global DeepseekV4MegaMoEExpertsFI

    class _DeepseekV4MegaMoEExpertsFI(mega_moe_experts_cls):
        """Thin wrapper: same weight layout/loader as mega experts, FI compute path."""

        def __init__(
            self,
            vllm_config: "VllmConfig",
            *,
            activation_clamp: float | None = None,
            fast_math: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config, **kwargs)
            self._vllm_config = vllm_config
            self._activation_clamp = activation_clamp
            self._fast_math = fast_math
            self._mega_layer = None
            self._fast_ctx = None

        def finalize_weights(self) -> None:
            if self._mega_layer is not None:
                return
            if self.w13_weight is None:
                return

            self._check_runtime_supported()
            ensure_fi_moe_ep_runtime(self._vllm_config)

            weights = mega_moe_weight_pack_from_params(
                self.w13_weight,
                self.w13_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
                megakernel=resolve_fi_megakernel(self._vllm_config),
            )
            self._mega_layer = build_fi_mega_layer(
                make_fi_moe_ep_bootstrap(),
                vllm_config=self._vllm_config,
                num_experts=self.num_experts,
                max_tokens_per_rank=self.max_num_tokens,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                activation_clamp=self._activation_clamp,
                weights=weights,
                fast_math=self._fast_math,
            )
            # The layer preprocesses at construction (_transformed holds the
            # kernel-ready tensors, which alias or replace the pack) but keeps
            # a reference to the canonical MoEWeightPack. For the cutedsl
            # kernels that pack is a per-layer bf16 DEQUANT copy (~3.2 GB per
            # layer here, 43 MoE layers -> OOM); for deep_gemm it pins the
            # loader-side fp4 originals. Drop it — repeated preprocessing is
            # already guarded by `_transformed is not None`.
            self._mega_layer._weights = None
            del weights
            self.w13_weight = None
            self.w13_weight_scale = None
            self.w2_weight = None
            self.w2_weight_scale = None

        def set_eplb_state(
            self,
            moe_layer_idx: int,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
        ) -> None:
            pass

        def get_expert_weights(self) -> list[torch.Tensor]:
            raise NotImplementedError(
                "EPLB expert weight export is not supported for fi_moe_ep yet."
            )

        def update_expert_map(self) -> None:
            pass

        def forward(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            *,
            activation_clamp: float | None,
            fast_math: bool = True,
        ) -> torch.Tensor:
            if hidden_states.shape[0] > self.max_num_tokens:
                raise ValueError(
                    f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                    f"but the symmetric buffer was sized for {self.max_num_tokens}."
                )

            from flashinfer.moe_ep import MoEEpTensors

            num_tokens = hidden_states.shape[0]
            is_padding = resolve_mega_moe_is_padding(num_tokens)
            topk_ids = apply_mega_moe_routing_preprocess(
                topk_ids,
                is_padding=is_padding,
            )

            # Validated-once fast path (mirrors the microbench's cached-launch
            # loop): MoEEpMegaLayer.forward() re-runs bootstrap/dist checks and
            # input validation on every call, which costs real host time at
            # 43 MoE layers x one call per engine step. After the first
            # successful full forward the layer is immutable, so go straight
            # to the kernel backend's stage_inputs + compute.
            fast = self._fast_ctx
            if fast is not None:
                kernel, workspace, transformed, hidden_size = fast
                t = MoEEpTensors(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
                kernel.stage_inputs(t, workspace, quantize_input=True)
                y = torch.empty(
                    num_tokens,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                return kernel.compute(workspace, transformed, output=y)

            ensure_fi_moe_ep_runtime(self._vllm_config)
            self.finalize_weights()
            assert self._mega_layer is not None

            y = self._mega_layer.forward(
                MoEEpTensors(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
            )
            layer = self._mega_layer
            if hidden_states.dtype == torch.bfloat16:
                self._fast_ctx = (
                    layer._kernel,
                    layer._ensure_workspace(),
                    layer._transformed,
                    layer._fleet_params.token_hidden_size,
                )
            return y

    _DeepseekV4MegaMoEExpertsFI.__name__ = "DeepseekV4MegaMoEExpertsFI"
    _DeepseekV4MegaMoEExpertsFI.__qualname__ = "DeepseekV4MegaMoEExpertsFI"
    _DeepseekV4MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]

    DeepseekV4MegaMoEExpertsFI = _DeepseekV4MegaMoEExpertsFI
    return DeepseekV4MegaMoEExpertsFI


__all__ = [
    "DeepseekV4MegaMoEExpertsFI",
    "FI_MEGA_KERNELS",
    "MEGA_MOE_BACKENDS",
    "apply_mega_moe_routing_preprocess",
    "build_fi_mega_config",
    "build_fi_mega_layer",
    "ensure_fi_moe_ep_runtime",
    "finalize_fi_moe_ep_runtime",
    "is_fi_moe_ep_backend",
    "is_mega_moe_backend",
    "make_fi_mega_moe_experts_cls",
    "make_fi_moe_ep_bootstrap",
    "mega_moe_weight_pack_from_params",
    "megakernel_runtime_requirements",
    "resolve_fi_megakernel",
    "resolve_mega_moe_is_padding",
]
