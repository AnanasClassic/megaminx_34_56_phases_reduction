from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionClassLinear(nn.Module):
    """Linear(one_hot(position, class)) without materializing the one-hot tensor."""

    def __init__(self, state_size: int, num_classes: int, output_size: int) -> None:
        super().__init__()
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_size = int(output_size)
        self.weight = nn.Parameter(torch.empty(self.state_size * self.num_classes, self.output_size))
        self.bias = nn.Parameter(torch.empty(self.output_size))
        self.register_buffer(
            "position_offsets",
            torch.arange(self.state_size, dtype=torch.int64) * self.num_classes,
            persistent=False,
        )
        bound = 1.0 / math.sqrt(self.state_size * self.num_classes)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.size(1) != self.state_size:
            raise ValueError(f"expected [batch,{self.state_size}] states, got {tuple(states.shape)}")
        if states.dtype != torch.int64:
            states = states.to(torch.int64)
        token_ids = states + self.position_offsets.unsqueeze(0)
        result = F.embedding_bag(token_ids, self.weight, mode="sum")
        return result + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.bn1(self.linear1(inputs)))
        hidden = self.dropout(hidden)
        hidden = self.bn2(self.linear2(hidden))
        return F.silu(inputs + hidden)


class PairQMLP(nn.Module):
    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        actions: int,
        hd1: int = 64,
        hd2: int = 256,
        residual_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(state_size, num_classes, actions, hd1, hd2) <= 0 or residual_blocks < 0:
            raise ValueError("all QMLP dimensions must be positive")
        self.config = {
            "state_size": int(state_size),
            "num_classes": int(num_classes),
            "actions": int(actions),
            "hd1": int(hd1),
            "hd2": int(hd2),
            "residual_blocks": int(residual_blocks),
            "dropout": float(dropout),
        }
        self.input_layer = PositionClassLinear(state_size, num_classes, hd1)
        self.input_bn = nn.BatchNorm1d(hd1)
        self.hidden = nn.Linear(hd1, hd2)
        self.hidden_bn = nn.BatchNorm1d(hd2)
        self.blocks = nn.ModuleList(ResidualBlock(hd2, dropout) for _ in range(residual_blocks))
        self.output_layer = nn.Linear(hd2, actions)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input_bn(self.input_layer(states)))
        hidden = F.silu(self.hidden_bn(self.hidden(hidden)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
