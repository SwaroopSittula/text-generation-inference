import torch

from text_generation_server.utils.weights import Weights
from text_generation_server.layers.marlin.gptq import GPTQMarlinWeight


class GPTQMarlinMoEWeight:
    all_qweight: torch.Tensor
    all_qzeros: torch.Tensor
    all_scales: torch.Tensor
    all_g_idx: torch.Tensor
    all_perm: torch.Tensor

    def __init__(
        self,
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
                    f"{prefix}.expert_{i}.{name}",
                )
            else:
                weight = weights.get_weights_row(
                    f"{prefix}.expert_{i}.{name}",
                )

            assert weight is GPTQMarlinWeight

            if i == 0:
                self.all_qweight = torch.empty(
                    (n_experts,) + weight.qweight.shape,
                    dtype=weight.qweight.dtype,
                    device=weight.qweight.device,
                )
                self.all_qzeros = torch.empty(
                    (n_experts,) + weight.qzeros.shape,
                    dtype=weight.qzeros.dtype,
                    device=weight.qzeros.device,
                )
                self.all_scales = torch.empty(
                    (n_experts,) + weight.scales.shape,
                    dtype=weight.scales.dtype,
                    device=weight.scales.device,
                )
                self.all_g_idx = torch.empty(
                    (n_experts,) + weight.g_idx.shape,
                    dtype=weight.g_idx.dtype,
                    device=weight.g_idx.device,
                )
                self.all_perm = torch.empty(
                    (n_experts,) + weight.perm.shape,
                    dtype=weight.perm.dtype,
                    device=weight.perm.device,
                )

            self.all_qweight[i] = weight.qweight
            self.all_qzeros[i] = weight.qzeros
            self.all_scales[i] = weight.scales
            self.all_g_idx[i] = weight.g_idx
            self.all_perm[i] = weight.perm

    def forward(self, x: torch.Tensor): ...


class GPTQMarlinMoE:
    def __init__(
        self,
        *,
        prefix: str,
        gate_proj: str,
        hidden_size: int,
        moe_intermediate_size: int,
        n_experts: int,
        up_proj: str,
        down_proj: str,
        weights: Weights,
    ):
        gate_proj = GPTQMarlinMoEWeight(
            prefix, n_experts, "gate_proj", weights, col_parallel=True
        )
        up_proj = GPTQMarlinMoEWeight(
            prefix, n_experts, "up_proj", weights, col_parallel=True
        )
        down_proj = GPTQMarlinMoEWeight(
            prefix, n_experts, "down_proj", weights, col_parallel=True
        )
