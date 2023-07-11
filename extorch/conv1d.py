"Extended Conv1d"

from typing import Literal, Any

from torch import Tensor, nn
import torch.nn.functional as F


class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    Causal + Stride              is supported (naively).
    Causal + Dilation            is supported.
    Causal + Stride   + Dilation is supported.
    """
    def __init__(self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        *args:        Any,
        stride:       int  = 1,
        padding: Literal["same", "valid", "scale"] | int | tuple[int] = 0,
        dilation:     int  = 1,
        groups:       int  = 1,
        bias:         bool = True,
        padding_mode: str  = "zeros",
        device             = None,
        dtype              = None,
        causal:       bool = False,
    ):
        """All arguments of `nn.Conv1d`, and new `causal` option"""

        # Validation
        if len(args) > 0:
            raise RuntimeError("Conv1dEx needs named arguments for stride and subsequents.")

        # Validation
        if (stride > 1) and (padding == "same"):
            raise RuntimeError("Convolution with stride>1 results in len(ipt) > len(opt), so `padding == 'same'` is not permitted.")

        # input_padding: Padding during Conv1dEx forward explicitly
        # conv_padding:  Padding in nn.Conv1d internally
        padding_total: int = (kernel_size - 1) * dilation
        if causal:
            # Validation
            ## padding amount
            if padding not in ("same", "scale"):
                raise RuntimeError("Conv1dEx with `causal=True` requires `padding=='same'|'scale'`.")
                # If len(opt)<len(ipt) by not-same/scale padding, 'causal or not' is determined by opt usage.
                # For example:
                #     padding=0 & opt[0] as t=k-1     -> causal conv with dropped opt_t0 ~ opt_t{k-2}
                #     padding=0 & opt[0] as t=(k-1)/2 -> normal conv with dropped head and tail
                # `causal` argument explicitly specify the mode, so should avoid this vague interpretation systematically.
            ## padding mode
            if padding_mode != "zeros":
                raise RuntimeError("Currently Conv1dEx support only `padding_mode='zeros'` for causal mode.")

            # Head-only padding - stride do not affect padding
            self._input_padding = (padding_total, 0)
            conv_padding = 0
        else:
            # scale x1 is equal to "same"
            if padding == "scale" and stride == 1:
                padding = "same"

            # Manual padding for strided (and dilated) conv
            if padding == "scale" and stride > 1:
                padding_l = padding_total // 2
                padding_r = padding_total - padding_l
                # NOTE: nn.Conv1d do not support LR-hetero explicit padding
                self._input_padding = (padding_l, padding_r)
                conv_padding = 0
            # Automatic padding for Conv | DilatedConv
            else:
                self._input_padding = (0, 0)
                conv_padding = padding

        super().__init__(in_channels, out_channels, kernel_size, stride, conv_padding, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x: Tensor):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self._input_padding))
