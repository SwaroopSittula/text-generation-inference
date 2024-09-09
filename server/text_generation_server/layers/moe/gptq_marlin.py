from dataclasses import dataclass
from typing import List, Optional

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
        n_expert_group: Optional[int],
        n_experts: int,
        prefix: str,
        renormalize: bool,
        topk: int,
        topk_group: Optional[int],
        weights: Weights,
        gate_proj_name: str = "gate_proj",
        up_proj_name: str = "up_proj",
        down_proj_name: str = "down_proj",
    ):
        super().__init__()

        assert (n_expert_group is None) == (
            topk_group is None
        ), "n_expert_group and topk_group must both be None or have some value"

        self.n_expert_group = n_expert_group
        self.topk = topk
        self.topk_group = topk_group
        self.renormalize = renormalize

        # TODO: update these to take prefixes
        self.gate_up_proj = _load_expert_multi_weights_col(
            prefix=prefix,
            n_experts=n_experts,
            names=[gate_proj_name, up_proj_name],
            weights=weights,
        )

        self.down_proj = _load_expert_weights_row(
            prefix=prefix, n_experts=n_experts, name=down_proj_name, weights=weights
        )

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        return fused_marlin_moe(
            x,
            w1=self.gate_up_proj.qweight,
            w2=self.down_proj.qweight,
            g_idx1=self.gate_up_proj.g_idx,
            g_idx2=self.down_proj.g_idx,
            rand_perm1=self.gate_up_proj.perm,
            rand_perm2=self.down_proj.perm,
            w1_scale=self.gate_up_proj.scales,
            w2_scale=self.down_proj.scales,
            gating_output=gating_output,
            topk=self.topk,
            renormalize=self.renormalize,
            # inplace=True,
            use_grouped_topk=self.n_expert_group is not None,
            num_expert_group=self.n_expert_group,
            topk_group=self.topk_group,
        )


def _load_expert_multi_weights_col(
    *,
    prefix: str,
    n_experts: int,
    names: List[str],
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    moe_weight = None
    for i in range(n_experts):
        weight = weights.get_multi_weights_col(
            [f"{prefix}.{i}.{name}" for name in names], 0
        )
        assert isinstance(weight, GPTQMarlinWeight)
        moe_weight = _pack_weight(
            n_experts=n_experts, expert=i, weight=weight, moe_weight=moe_weight
        )
    assert moe_weight is not None
    return moe_weight


def _load_expert_weights_row(
    *,
    prefix: str,
    n_experts: int,
    name: str,
    weights: Weights,
) -> GPTQMarlinMoEWeight:
    moe_weight = None
    for i in range(n_experts):
        weight = weights.get_weights_row(
            f"{prefix}.{i}.{name}",
        )
        assert isinstance(weight, GPTQMarlinWeight)
        moe_weight = _pack_weight(
            n_experts=n_experts, expert=i, weight=weight, moe_weight=moe_weight
        )
    assert moe_weight is not None
    return moe_weight


def _pack_weight(
    *,
    n_experts: int,
    expert: int,
    moe_weight: Optional[GPTQMarlinMoEWeight],
    weight: GPTQMarlinWeight,
) -> GPTQMarlinMoEWeight:
    if moe_weight is None:
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

        moe_weight = GPTQMarlinMoEWeight(
            qweight=qweight, qzeros=qzeros, scales=scales, g_idx=g_idx, perm=perm
        )

    moe_weight.qweight[expert] = weight.qweight
    moe_weight.qzeros[expert] = weight.qzeros
    moe_weight.scales[expert] = weight.scales
    moe_weight.g_idx[expert] = weight.g_idx
    moe_weight.perm[expert] = weight.perm

    return moe_weight
