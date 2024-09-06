from moe_kernels.fused_moe import fused_experts
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
