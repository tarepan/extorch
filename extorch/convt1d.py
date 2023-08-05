"Extended ConvTranspose1d"

from typing import Literal, Any

from torch import Tensor, nn

from .padding import padding_lr


class ConvT1dEx(nn.ConvTranspose1d):
    """Extended ConvTranspose1d which support cansal convolution.

    Causal + Stride is supported (naively).
    """
    def __init__(self,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int,
        *args:          Any,
        shape:          Literal["delta", "causal"] = "delta",
        stride:         int = 1,
        align:          None | Literal["head", "center", "tail"] = None,
        padding: Literal["same", "valid", "scale"] | int | tuple[int] = 0,
        output_padding: int  = 0,
        groups:         int  = 1,
        bias:           bool = True,
        dilation:       int  = 1,
        padding_mode:   str  = "zeros",
        device               = None,
        dtype                = None,
    ):
        """All arguments of `nn.ConvTranspose1d`, and new `causal` option."""

        # Default values
        if align is not None:
            _align = align
        else:
            _align = "head" if shape == "causal" else "center"

        # Validation
        if len(args) > 0:
            raise RuntimeError("ConvT1dEx needs named arguments for stride and subsequents.")
        if (stride > 1) and (padding == "same"):
            raise RuntimeError("Transposed convolution with stride>1 results in len(opt) > len(ipt), so `padding == 'same'` is not permitted.")
        ## Support only drop_last=True
        if output_padding > 0:
            raise RuntimeError("Currently ConvT1dEx support only `output_padding=0` mode.")
        if (shape == "causal") and (padding not in ("same", "scale")):
            raise RuntimeError("ConvT1dEx with `causal=True` requires `padding=='same'|'scale'`.")
            # If len(opt)!=len(ipt) by not-same/scale padding, 'causal or not' is determined by opt usage.
            # For example:
            #     padding=0 & opt[0] as t=-1 -> normal conv with dropped head and tail
            #     padding=0 & opt[0] as t=0  -> causal conv with dropped tail
            # `causal` argument explicitly specify the mode, so should avoid this vague interpretation systematically.
        if (shape == "causal") and (padding_mode != "zeros"):
            raise RuntimeError("Currently ConvT1dEx support only `padding_mode='zeros'` for causal mode.")

        # input_padding: Padding^-1 during ConvT1dEx forward explicitly
        # conv_padding:  Padding^-1 in nn.ConvTranspose1d internally
        effective_kernel = 1 + (kernel_size - 1) * dilation

        # PyTorch native padding
        if isinstance(padding, int) or isinstance(padding, tuple):
            self._input_padding = (None, None)
            conv_padding = padding
        # extorch extended padding
        elif padding == "valid":
            self._input_padding = (None, None)
            conv_padding = 0
        else:
            _shape = "inv_causal" if shape == "causal" else shape
            padding_r, padding_l = padding_lr(effective_kernel, _shape, stride, _align, True)
            padding_r = padding_r if padding_r > 0 else None
            padding_l = padding_l if padding_l > 0 else None
            self._input_padding = (padding_r, padding_l)
            conv_padding = 0

        super().__init__(in_channels, out_channels, kernel_size, stride, conv_padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)

    def forward(self, x: Tensor):
        """Forward ConvT1dEx with non-uniform padding^-1"""
        opt_full = super().forward(x)
        ipad_l = None if self._input_padding[1] is None else -1 * self._input_padding[1]
        return opt_full[..., self._input_padding[0] : ipad_l]
