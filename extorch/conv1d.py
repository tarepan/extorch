"Extended Conv1d"

from typing import Literal, Any
import warnings

from torch import Tensor, nn
import torch.nn.functional as F

from .padding import padding_lr


# Design Notes:
#    'drop_last when strided'-like behavior is not defined in PyTorch (PyTorch do not support auto-padding for strided Conv).
#    We defined the 'scale' mode for this purpose.

class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    Causal + Stride + Dilation is supported.
    """
    def __init__(self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        *args:        Any,
        stride:       int  = 1,
        padding:      Literal["same", "valid", "scale"] | int | tuple[int] = 0,
        dilation:     int  = 1,
        groups:       int  = 1,
        bias:         bool = True,
        padding_mode: str  = "zeros",
        device             = None,
        dtype              = None,
        causal:       None | bool = None,
        shape:        None | Literal["delta", "causal"] = None,
        align:        None | Literal["head", "center", "tail"] = None,
        drop_last:    bool = False,
    ):
        """All arguments of `nn.Conv1d`, and new `causal` option:

        Args:
            padding   - Padding mode ('scale' basically results in 'floor(L/s)' when `drop_last` or 'ceil(L/s)' when `not drop_last`, but there are some exceptions)
            shape     - Kernel shape, centerd 'delta' or right triangle 'causal'
            align     - Kernel axis alignment position in a frame
            drop_last - Whether to drop the last frame if kernel is not fulfilled in the frame. If False, pad for the frame.
        """

        # Deprecation warning
        if causal is not None:
            warnings.warn("Conv1dEx's `causal` argument is deprecated. Instead, use `shape='causal'`.")

        # Default value
        # NOTE: None is used for backward-compatibility
        ## causal & shape
        ### value/value
        if (causal is not None) and (shape is not None):
            _causal = causal
            _shape  = shape
        ### None/value
        elif (causal is None) and (shape is not None):
            _causal = True if shape == "causal" else False
            _shape  = shape
        ### value/None
        elif (causal is not None) and (shape is None):
            _causal = causal
            _shape  = "causal" if causal else "delta"
        ### None/None
        else:
            _causal = False
            _shape  = "delta"
        ## align
        if align is not None:
            _align = align
        else:
            # TODO: non-causal's design note (why not 'center'?)
            _align = "tail" if _shape == "causal" else "head"

        # Validation
        if len(args) > 0:
            raise RuntimeError("Conv1dEx needs named arguments for stride and subsequents.")
        if (stride > 1) and (padding == "same"):
            raise RuntimeError("Convolution with stride>1 results in len(ipt) > len(opt), so `padding == 'same'` is not permitted.")
        if _causal and (padding not in ("same", "scale")):
            raise RuntimeError("Conv1dEx with `causal=True` requires `padding=='same'|'scale'`.")
            # If len(opt)<len(ipt) by not-same/scale padding, 'causal or not' is determined by opt usage.
            # For example:
            #     padding=0 & opt[0] as t=k-1     -> causal conv with dropped opt_t0 ~ opt_t{k-2}
            #     padding=0 & opt[0] as t=(k-1)/2 -> normal conv with dropped head and tail
            # `causal` argument explicitly specify the mode, so should avoid this vague interpretation systematically.
        if _causal and padding_mode != "zeros":
            raise RuntimeError("Currently Conv1dEx support only `padding_mode='zeros'` for causal mode.")
        if _causal and _shape != "causal":
            raise RuntimeError(f"Same parameters are conflicted: causal {_causal} != kernel_shape {_shape}")

        # Warning
        if (_shape == "causal") and (stride > 1) and (_align != "tail"):
            warnings.warn("You run strided causal conv with `non-tail` alignment.")

        # Parameter conversion
        if padding == "scale" and stride == 1:
            padding = "same"

        # input_padding: Padding during Conv1dEx forward explicitly
        # conv_padding:  Padding in nn.Conv1d internally
        effective_kernel = 1 + (kernel_size - 1) * dilation

        # PyTorch native padding
        if _shape == "delta" and ((padding == "same") or (padding == "valid") or (isinstance(padding, int)) or (isinstance(padding, tuple))):
            self._input_padding = (0, 0)
            conv_padding = padding
            # Kernel centering warning: 'nn.Conv1d's built-in warning' if 'dilation*(kernel_size-1)+1 is even' else pass
            # In this case, 'padding_l + 1 == padding_r'
        # extorch extended padding
        else:
            self._input_padding = padding_lr(effective_kernel, _shape, stride, _align, drop_last)
            conv_padding = 0

        super().__init__(in_channels, out_channels, kernel_size, stride, conv_padding, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x: Tensor):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self._input_padding))
