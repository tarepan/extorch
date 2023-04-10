"Transpose nn.Module"

import torch
from torch import nn


class Transpose(nn.Module):
    """torch.transpose equivalent nn.Module."""

    def __init__(self, dim0: int, dim1: int):
        """Arguments of `torch.transpose`."""
        super().__init__()
        assert isinstance(dim0, int), "`dim0` should be Int, but `{dim0}` is {type(dim0)}"
        assert isinstance(dim1, int), "`dim1` should be Int, but `{dim1}` is {type(dim1)}"
        self._dim0, self._dim1 = dim0, dim1

    def forward(self, x: torch.Tensor):
        """Transpose pre-defined dimensions."""
        return torch.transpose(x, self._dim0, self._dim1)
