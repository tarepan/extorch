"Extended ConvTranspose1d"

from typing import Literal, Any

from torch import Tensor, nn


class ConvT1dEx(nn.ConvTranspose1d):
    """Extended ConvTranspose1d which support cansal convolution.

    Causal + Stride is supported (naively).
    """
    def __init__(self,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int,
        *args:          Any,
        stride:         int = 1,
        padding: Literal["same", "valid", "scale"] | int | tuple[int] = 0,
        output_padding: int  = 0,
        groups:         int  = 1,
        bias:           bool = True,
        dilation:       int  = 1,
        padding_mode:   str  = "zeros",
        device               = None,
        dtype                = None,
        causal: bool = False,
    ):
        """All arguments of `nn.ConvTranspose1d`, and new `causal` option."""

        # Validation
        if len(args) > 0:
            raise RuntimeError("ConvT1dEx needs named arguments for stride and subsequents.")
        if (stride > 1) and (padding == "same"):
            raise RuntimeError("Transposed convolution with stride>1 results in len(opt) > len(ipt), so `padding == 'same'` is not permitted.")
        if output_padding > 0:
            raise RuntimeError("Currently ConvT1dEx support only `output_padding=0` mode.")

        # input_padding: Padding^-1 during ConvT1dEx forward explicitly
        # conv_padding:  Padding^-1 in nn.ConvTranspose1d internally
        padding_total = (kernel_size - 1) * dilation
        if causal:
            # Validation
            ## padding amount
            if padding not in ("same", "scale"):
                raise RuntimeError("ConvT1dEx with `causal=True` requires `padding=='same'|'scale'`.")
                # If len(opt)!=len(ipt) by not-same/scale padding, 'causal or not' is determined by opt usage.
                # For example:
                #     padding=0 & opt[0] as t=-1 -> normal conv with dropped head and tail
                #     padding=0 & opt[0] as t=0  -> causal conv with dropped tail
                # `causal` argument explicitly specify the mode, so should avoid this vague interpretation systematically.
            ## padding mode
            if padding_mode != "zeros":
                raise RuntimeError("Currently ConvT1dEx support only `padding_mode='zeros'` for causal mode.")

            # Tail cut
            self._input_padding = (0, padding_total)
            conv_padding = 0
        else:
            # Manual padding - nn.ConvTranspose1d do not support string padding arguments
            if padding == "scale" or padding == "same":
                padding_l = padding_total // 2
                padding_r = padding_total - padding_l
                # NOTE: nn.ConvTranspose1d do not support LR-hetero explicit padding
                self._input_padding = (padding_l, padding_r)
                conv_padding = 0
            elif padding == "valid":
                self._input_padding = (0, 0)
                conv_padding = 0
            else:
                self._input_padding = (0, 0)
                conv_padding = padding

        super().__init__(in_channels, out_channels, kernel_size, stride, conv_padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)

    def forward(self, x: Tensor):
        """Forward ConvT1dEx with non-uniform padding^-1"""
        opt_full = super().forward(x)
        return opt_full[..., self._input_padding[0]:] if self._input_padding[1] == 0 else opt_full[..., self._input_padding[0] : -1*self._input_padding[1]]
