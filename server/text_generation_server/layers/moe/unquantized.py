from typing import Optional

from moe_kernels.fused_moe import fused_moe
import torch
import torch.nn as nn

from text_generation_server.utils.weights import UnquantizedWeight, Weights


class UnquantizedMoELayer(nn.Module):
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

        gate_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name=gate_proj_name,
            weights=weights,
            col_parallel=True,
        )
        up_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name=up_proj_name,
            weights=weights,
            col_parallel=True,
        )

        # First dimension are the experts.
        self.gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)

        self.down_proj = _load_expert_weights(
            prefix=prefix,
            n_experts=n_experts,
            name=down_proj_name,
            weights=weights,
            col_parallel=False,
        )

    def forward(self, x: torch.Tensor, *, gating_output: torch.Tensor) -> torch.Tensor:
        return fused_moe(
            x,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=gating_output,
            topk=self.topk,
            renormalize=self.renormalize,
            inplace=True,
            use_grouped_topk=self.n_expert_group is not None,
            num_expert_group=self.n_expert_group,
            topk_group=self.topk_group,
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
                f"{prefix}.{i}.{name}",
            )
        else:
            weight = weights.get_weights_row(
                f"{prefix}.{i}.{name}",
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
