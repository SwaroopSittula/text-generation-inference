from dataclasses import dataclass

from moe_kernels import marlin_gemm_moe
import torch

from text_generation_server.utils.weights import Weights
from text_generation_server.layers.marlin.gptq import GPTQMarlinWeight


@dataclass
class GPTQMarlinMoEWeight:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    perm: torch.Tensor


class GPTQMarlinMoE:
    def __init__(
        self,
        *,
        n_experts: int,
        prefix: str,
        weights: Weights,
    ):
        self.gate_proj = _load_expert_weights(
            prefix, n_experts, "gate_proj", weights, col_parallel=True
        )
        self.up_proj = _load_expert_weights(
            prefix, n_experts, "up_proj", weights, col_parallel=True
        )
        self.down_proj = _load_expert_weights(
            prefix, n_experts, "down_proj", weights, col_parallel=True
        )

    def forward(
        self, x: torch.Tensor, *, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor: ...


def _load_expert_weights(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
    col_parallel: bool,
) -> GPTQMarlinMoEWeight:
    for i in range(n_experts):
        if col_parallel:
            weight = weights.get_weights_col(
                f"{prefix}.expert_{i}.{name}",
            )
        else:
            weight = weights.get_weights_row(
                f"{prefix}.expert_{i}.{name}",
            )

        assert weight is GPTQMarlinWeight

        if i == 0:
            qweight = torch.empty(
                (n_experts,) + weight.qweight.shape,
                dtype=weight.qweight.dtype,
                device=weight.qweight.device,
            )
            qzeros = torch.empty(
                (n_experts,) + weight.qzeros.shape,
                dtype=weight.qzeros.dtype,
                device=weight.qzeros.device,
            )
            scales = torch.empty(
                (n_experts,) + weight.scales.shape,
                dtype=weight.scales.dtype,
                device=weight.scales.device,
            )
            g_idx = torch.empty(
                (n_experts,) + weight.g_idx.shape,
                dtype=weight.g_idx.dtype,
                device=weight.g_idx.device,
            )
            perm = torch.empty(
                (n_experts,) + weight.perm.shape,
                dtype=weight.perm.dtype,
                device=weight.perm.device,
            )

        qweight[i] = weight.qweight
        qzeros[i] = weight.qzeros
        scales[i] = weight.scales
        g_idx[i] = weight.g_idx
        perm[i] = weight.perm

    return GPTQMarlinMoEWeight(
        qweight=qweight, qzeros=qzeros, scales=scales, g_idx=g_idx, perm=perm
    )


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    g_idx1: torch.Tensor,
    g_idx2: torch.Tensor,
    rand_perm1: torch.Tensor,
    rand_perm2: torch.Tensor,
    topk: int,
    custom_routing_function: Optional[Callable] = None,
    renormalize: bool = True,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.
    - use_fp8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // 2, "Hidden size mismatch w2"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # TODO fp8 is not implemented yet
    assert not use_fp8

    import triton.language as tl
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs,
        invoke_fused_moe_kernel,
        moe_align_block_size,
    )

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16

    if custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(
            hidden_states, gating_output, topk, renormalize
        )
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize
        )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        "float8" if use_fp8 else None,
        override_config=override_config,
        is_marlin=True,
    )
    config = get_config_func(M)

    block_size_m = config["BLOCK_SIZE_M"]

    sorted_token_ids, _, _ = moe_align_block_size(topk_ids, block_size_m, E)

    max_workspace_size = ((M + 255) // 256) * (max(2 * N, K) // 64) * 16
    workspace = torch.zeros(
        max_workspace_size, dtype=torch.int, device="cuda", requires_grad=False
    )

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    intermediate_cache1 = torch.ops._moe_C.marlin_gemm_moe(
        hidden_states,
        w1,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w1_scale,
        g_idx1,
        rand_perm1,
        workspace,
        M,
        2 * N,
        K,
        True,
        E,
        topk,
        block_size_m,
        True,
        False,
    )

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, 2 * N))

    intermediate_cache3 = torch.ops._moe_C.marlin_gemm_moe(
        intermediate_cache2,
        w2,
        sorted_token_ids,
        topk_weights,
        topk_ids,
        w2_scale,
        g_idx2,
        rand_perm2,
        workspace,
        M,
        K,
        N,
        True,
        E,
        topk,
        block_size_m,
        False,
        True,
    )

    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
