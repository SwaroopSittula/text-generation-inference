from dataclasses import dataclass
from typing import List

from moe_kernels.fused_moe import fused_marlin_moe
import torch
import torch.nn as nn

from text_generation_server.utils.weights import Weights
from text_generation_server.layers.marlin.gptq import GPTQMarlinWeight


@dataclass
class GPTQMarlinMoEWeight:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor
    perm: torch.Tensor


class GPTQMarlinMoE(nn.Module):
    def __init__(
        self,
        *,
        n_experts: int,
        prefix: str,
        weights: Weights,
    ):
        super().__init__()

        self.gate_up_proj = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            names=["gate_proj", "up_proj"],
            weights=weights,
        )

        self.down_proj = _load_expert_weights_row(
            prefix, n_experts, "down_proj", weights, col_parallel=True
        )

    def forward(
        self, x: torch.Tensor, *, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        return fused_marlin_moe(
            x,
            self.gate_up_proj.qweight,
            self.down_proj.qweight,
            topk_weights,
            self.gate_up_proj.g_idx,
            self.down_proj.g_idx,
            self.gate_up_proj.perm,
            self.down_proj.perm,
            topk_ids,
            w1_scale=self.gate_up_proj.scales,
            w2_scale=self.down_proj.scales,
        )


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    names: List[str],
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    weight = None
    for i in range(n_experts):
        weight = weights.get_multi_weights_col(
            [f"{prefix}.{name}" for name in names], 0
        )
        assert weight is GPTQMarlinWeight
        weight = _pack_weight(n_experts=n_experts, expert=i, weight=weight)
    assert weight is not None
    return weight


def _load_expert_weights_row(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    weight = None
    for i in range(n_experts):
        weight = weights.get_weights_row(
            f"{prefix}.expert_{i}.{name}",
        )
        assert weight is GPTQMarlinWeight
        weight = _pack_weight(n_experts=n_experts, expert=i, weight=weight)
    assert weight is not None
    return weight


def _pack_weight(
    *, n_experts: int, expert: int, weight: Optional[GPTQMarlinMoEWeight]
) -> GPTQMarlinMoEWeight:
    if weight is None:
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

        weight = GPTQMarlinMoEWeight(
            qweight=qweight, qzeros=qzeros, scales=scales, g_idx=g_idx, perm=perm
        )

    weight.qweight[expert] = weight.qweight
    weight.qzeros[expert] = weight.qzeros
    weight.scales[expert] = weight.scales
    weight.g_idx[expert] = weight.g_idx
    weight.perm[expert] = weight.perm

    return weight
