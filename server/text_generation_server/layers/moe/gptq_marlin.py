from dataclasses import dataclass

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
    self,
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
    col_parallel: bool,
) -> GPQMarlinMoEWeight:
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
