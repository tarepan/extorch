"Extended Conv1d"

from warnings import warn
from typing import Union, Tuple

from torch import nn
import torch.nn.functional as F


class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    CausalConv + Stride is (naively) supported.
    CausalConv + Dilation is NOT yet supported.
    """
    def __init__(self,
        *args,
        padding: Union[str, int, Tuple] = 0,
        padding_mode: str = "zeros",
        causal: bool = False,
        **kwargs
    ):
        """All arguments of `nn.Conv1d`, and new `causal` option"""

        # Conv1d `kernel_size` could be positional or named.
        #   `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, ...)`
        kernel_size = args[2] if len(args) >= 3 else kwargs["kernel_size"]
        if len(args) >= 4:
            raise Exception("Currently Conv1dEx needs named argument usage for stride and subsequents.")

        # input_padding: Padding during Conv1dEx forward explicitly
        # conv_padding:  Padding in nn.Conv1d internally
        if causal:
            self._input_padding = (kernel_size-1, 0)
            conv_padding = 0
            if padding is not 0:
                if padding == "same":
                    # Automatic 'same' padding match explicit argument
                    pass
                else:
                    warn(f"Conv1dEx: `padding={padding}` is ignored, now using `causal` mode.")
            if padding_mode is not "zeros":
                raise Exception("Currently Conv1dEx support only padding_mode=zeros for causal mode.")
        else:
            self._input_padding = (0, 0)
            conv_padding = padding

        super().__init__(*args, padding=conv_padding, padding_mode=padding_mode, **kwargs)

    def forward(self, x):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self._input_padding))
