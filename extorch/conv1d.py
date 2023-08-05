"Extended Conv1d"

from typing import Literal, Any
import warnings

from torch import Tensor, nn
import torch.nn.functional as F

from .padding import padding_lr


class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    Additional features:
        - Causal convolution: ◢ shape kernel
        - Automatic padding of strided conv
            - modes
                - 'valid':      Use only valid values (no padding)
                - 'scale_drop': Scale kernel-fulfilled strides (drop non-fulfilled last stride), basically L_out = floor(L_in/stride)
                - 'scale_ceil': Scale all strides, including non-fulfilled last one,             basically L_out =  ceil(L_in/stride)
            - alignment
                - Normal conv: Kernel axis is aligned to the stride center
                - Causal conv: Kernel axis is aligned to the stride tail
    """
    def __init__(self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        *args:        Any,
        causal:       bool = False,
        stride:       int  = 1,
        padding:      Literal["same", "valid", "scale", "scale_drop", "scale_ceil"] | int | tuple[int] = 0,
        dilation:     int  = 1,
        groups:       int  = 1,
        bias:         bool = True,
        padding_mode: str  = "zeros",
        device             = None,
        dtype              = None,
    ):
        """All arguments of `nn.Conv1d`, and new `causal` option:

        Args:
            causal - Whether to use causal ConvT (all Right ◣)
            padding - Padding size or automatic padding mode (c.f. Class description)
        """

        # Backward compatibility
        padding = padding if padding != "scale" else "scale_ceil"

        #                                               normal                      causal
        shape:     Literal["delta", "causal"]        = "delta"  if not causal else "causal"
        align:     Literal["head", "center", "tail"] = "center" if not causal else "tail"
        drop_last: bool = padding == "scale_drop"

        # Validation
        if len(args) > 0:
            raise RuntimeError("Conv1dEx needs named arguments for stride and subsequents.")
        if (stride > 1) and (padding == "same"):
            raise RuntimeError("Convolution with stride>1 results in len(ipt) > len(opt), so `padding == 'same'` is not permitted.")
        if causal and (padding not in ("same", "scale_drop", "scale_ceil")):
            raise RuntimeError("Conv1dEx with `causal=True` requires `padding=='same'|'scale_drop'|'scale_ceil'`.")
            # If len(opt)<len(ipt) by not-same/scale padding, 'causal or not' is determined by opt usage.
            # For example:
            #     padding=0 & opt[0] as t=k-1     -> causal conv with dropped opt_t0 ~ opt_t{k-2}
            #     padding=0 & opt[0] as t=(k-1)/2 -> normal conv with dropped head and tail
            # `causal` argument explicitly specify the mode, so should avoid this vague interpretation systematically.
        if causal and padding_mode != "zeros":
            raise RuntimeError("Currently Conv1dEx support only `padding_mode='zeros'` for causal mode.")

        # Parameter conversion
        if padding in ("scale_drop", "scale_ceil") and stride == 1:
            padding = "same"

        # input_padding: Padding during Conv1dEx forward explicitly
        # conv_padding:  Padding in nn.Conv1d internally
        effective_kernel = 1 + (kernel_size - 1) * dilation

        # PyTorch native padding
        if shape == "delta" and ((padding == "same") or (padding == "valid") or (isinstance(padding, (int, tuple)))):
            self._input_padding = (0, 0)
            conv_padding = padding
            # Kernel centering warning: 'nn.Conv1d's built-in warning' if 'dilation*(kernel_size-1)+1 is even' else pass
            # In this case, 'padding_l + 1 == padding_r'
        # extorch extended padding
        else:
            self._input_padding = padding_lr(effective_kernel, shape, stride, align, drop_last)
            conv_padding = 0

        super().__init__(in_channels, out_channels, kernel_size, stride, conv_padding, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x: Tensor):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self._input_padding))
