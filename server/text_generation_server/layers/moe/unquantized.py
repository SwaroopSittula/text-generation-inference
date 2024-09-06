from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from text_generation_server.utils.weights import UnquantizedWeight, Weights


class UnquantizedMoELayer(nn.Module):
    def __init__(
        self,
        *,
        n_experts: int,
        prefix: str,
        weights: Weights,
    ):
        super().__init__()

        gate_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name="gate_proj",
            weights=weights,
            col_parallel=True,
        )
        up_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name="up_proj",
            weights=weights,
            col_parallel=True,
        )

        # First dimension are the experts.
        self.gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)

        self.down_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name="down_proj",
            weights=weights,
            col_parallel=False,
        )

    def forward(
        self, x: torch.Tensor, *, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        return fused_experts(
            x,
            self.gate_up_proj,
            self.down_proj,
            topk_weights,
            topk_ids,
            inplace=True,
        )


def _load_expert_weights(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
    col_parallel: bool,
):
    for i in range(n_experts):
        if col_parallel:
            weight = weights.get_weights_col(
                f"{prefix}.experts.{i}.{name}",
            )
        else:
            weight = weights.get_weights_row(
                f"{prefix}.experts.{i}.{name}",
            )

        assert isinstance(weight, UnquantizedWeight)

        if i == 0:
            all_weight = torch.empty(
                (n_experts,) + weight.weight.shape,
                dtype=weight.weight.dtype,
                device=weight.weight.device,
            )

        all_weight[i] = weight.weight

    return all_weight


# Functions below are from vLLM:
#
# https://github.com/vllm-project/vllm/blob/f7160d946a0a07703e72d81ba9ecf3913f192605/vllm/model_executor/layers/fused_moe/fused_moe.py#L397


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
) -> Dict[str, int]:
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    import triton.language as tl
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs,
        invoke_fused_moe_kernel,
        moe_align_block_size,
    )

    M, _ = hidden_states.shape
    E, N, _ = w1.shape

    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        configs = get_moe_configs(E, w2.shape[2], "float8" if use_fp8 else None)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(
                M, E, N, w1.shape[2], topk_ids.shape[1], "float8" if use_fp8 else None
            )

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=compute_type,
        use_fp8=use_fp8,
    )

    if inplace:
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=hidden_states,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
