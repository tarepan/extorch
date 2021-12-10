"Extended Conv1d"

from warnings import warn

from torch import nn
import torch.nn.functional as F


class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    CausalConv + Stride is (naively) supported.
    CausalConv + Dilation is NOT yet supported.
    """
    def __init__(self, *args, padding=0, causal:bool=False, **kwargs):
        kernel_size = args[2]
        self.input_padding = (kernel_size-1, 0) if causal else (0, 0)
        kernel_padding = 0 if causal else padding
        if causal and padding is not 0:
            warn("`padding` is ignored and automatically set because you turn on `causal`.")
        super().__init__(*args, padding=kernel_padding, **kwargs)

    def forward(self, x):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self.input_padding))
