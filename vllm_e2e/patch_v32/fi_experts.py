"""flashinfer moe_ep RoutedExperts for DeepSeek V3.2 (NVFP4 checkpoint).

V3.2 has no mega-MoE path of its own: its decoder uses the stock
DeepseekV2MoE -> FusedMoE factory -> MoERunner/RoutedExperts pipeline
(monolithic ModelOptNvFp4FusedMoE, MoEPrepareAndFinalizeNoDPEPMonolithic).
Rather than porting the V4 mega experts class, this keeps the stock wiring
(router, runner, weight loading) and swaps only the storage+compute half:

  * install() monkeypatches fused_moe.layer.RoutedExperts (the factory's
    default routed_experts_cls) so every MoE layer gets FiRoutedExpertsV32.
    Gated on FI_MOE_EP=1 by the patched deepseek_v32/nvidia/model.py.
  * process_weights_after_loading is intercepted per-instance: instead of
    converting the raw modelopt params to the native kernel format (and
    doubling weight memory), it builds the fi prequantized NVFP4 pack +
    epilogue alphas (same recipe as the V4 wrapper) and frees the raw params.
  * forward_monolithic routes with the stock router (sigmoid grouped top-k +
    e_score_correction_bias, scaling factor handled by the runner) and runs
    the fi mega layer, reusing the V4 wrapper's fast-path/staging helpers.

Output convention: the fi combine returns the FULL routed sum on every rank,
but this model defers a TP all-reduce into the next layernorm assuming
partial (local-experts) outputs — so scale by 1/tp_world before returning.
The division by a power of two is exact in bf16, and the all-reduce then
reconstructs the full value (shared-expert partials ride the same reduce).
Note the TP-replicated tokens mean the fi dispatch carries world_size copies
of every token — inherent to EP-kernel-in-TP-model; use bench_offline_dp.py
(DP attention) for the fi-favorable topology, as with V4 run 18.
"""

from __future__ import annotations

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size


def fi_v32_enabled() -> bool:
    return os.environ.get("FI_MOE_EP", "0").lower() in ("1", "true", "yes")


def install() -> None:
    """Make the FusedMoE factory build FiRoutedExpertsV32 by default."""
    from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer

    if fused_moe_layer.RoutedExperts is not FiRoutedExpertsV32:
        fused_moe_layer.RoutedExperts = FiRoutedExpertsV32


from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class FiRoutedExpertsV32(RoutedExperts):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.models.deepseek_v4.nvidia.fi_utils import resolve_fi_megakernel

        self._vllm_config = get_current_vllm_config()
        megakernel = resolve_fi_megakernel(self._vllm_config)
        if megakernel != "nvfp4_cutedsl":
            raise ValueError(
                "DeepSeek V3.2 fi path only supports the NVFP4-quantized "
                "checkpoint (FI_MOE_EP_MEGAKERNEL=nvfp4_cutedsl); the fp8 "
                f"checkpoint does not fit one node. Got {megakernel!r}."
            )
        self._mega_layer = None
        self._epilogue_alphas: tuple[torch.Tensor, torch.Tensor] | None = None
        self._fast_ctx = None
        self._fi_router = None  # stitched in by the patched model.py
        self._inv_tp = 1.0 / get_tensor_model_parallel_world_size()

        # Replace the native kernel-format conversion for THIS layer only
        # (quant-method instances are per-layer, but stay defensive).
        qm = self.quant_method
        orig_process = qm.process_weights_after_loading

        def _fi_process_weights(layer, _orig=orig_process):
            if layer is self:
                self._fi_finalize_weights()
            else:
                _orig(layer)

        qm.process_weights_after_loading = _fi_process_weights

    def _fi_finalize_weights(self) -> None:
        if self._mega_layer is not None:
            return
        from vllm.models.deepseek_v4.nvidia.fi_utils import (
            build_fi_mega_layer,
            ensure_fi_moe_ep_runtime,
            make_fi_moe_ep_bootstrap,
            nvfp4_prequant_pack_and_alphas,
        )

        ensure_fi_moe_ep_runtime(self._vllm_config)

        weights, fc1_alpha, fc2_alpha = nvfp4_prequant_pack_and_alphas(
            self.w13_weight.data,
            self.w13_weight_scale.data,
            self.w13_weight_scale_2.data,
            self.w2_weight.data,
            self.w2_weight_scale.data,
            # modelopt allocates this per-tensor scale as (E,) or (E, 1)
            # depending on version; the pack function wants (E,).
            self.w2_weight_scale_2.data.reshape(-1),
            intermediate_size=self.intermediate_size_per_partition,
        )
        self._epilogue_alphas = (fc1_alpha, fc2_alpha)
        self._mega_layer = build_fi_mega_layer(
            make_fi_moe_ep_bootstrap(),
            vllm_config=self._vllm_config,
            num_experts=self.global_num_experts,
            max_tokens_per_rank=self.moe_config.max_num_tokens,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            top_k=self.top_k,
            activation_clamp=None,  # V3.2 has no swiglu_limit
            weights=weights,
        )
        del weights
        self._mega_layer._ensure_workspace()
        # Free the raw checkpoint params — fi holds the preprocessed pack.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w13_weight_scale_2 = None
        self.w13_input_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        self.w2_weight_scale_2 = None
        self.w2_input_scale = None

    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from flashinfer.moe_ep import MoEEpTensors
        from vllm.models.deepseek_v4.nvidia.fi_utils import (
            apply_mega_moe_routing_preprocess,
            resolve_mega_moe_is_padding,
        )

        assert self._fi_router is not None, (
            "FiRoutedExpertsV32 needs its router stitched in "
            "(see the deepseek_v32 model patch)."
        )
        topk_weights, topk_ids = self._fi_router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )
        num_tokens = x.shape[0]
        is_padding = resolve_mega_moe_is_padding(num_tokens)
        topk_ids = apply_mega_moe_routing_preprocess(topk_ids, is_padding=is_padding)

        alphas = self._epilogue_alphas
        t = MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=alphas[0] if alphas is not None else None,
            fc2_alpha=alphas[1] if alphas is not None else None,
        )

        fast = self._fast_ctx
        if fast is not None:
            kernel, workspace, transformed = fast
            kernel.stage_inputs(t, workspace, quantize_input=True)
            y = kernel.compute(workspace, transformed, output=None)
            return y * self._inv_tp

        self._fi_finalize_weights()
        assert self._mega_layer is not None
        y = self._mega_layer.forward(t)
        layer = self._mega_layer
        if x.dtype == torch.bfloat16:
            self._fast_ctx = (
                layer._kernel,
                layer._ensure_workspace(),
                layer._transformed,
            )
        return y * self._inv_tp

    def forward_modular(self, *args, **kwargs) -> torch.Tensor:
        raise AssertionError(
            "FiRoutedExpertsV32 expects the monolithic runner path "
            "(ModelOptNvFp4FusedMoE is monolithic)."
        )
