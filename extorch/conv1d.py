"Extended Conv1d"

from warnings import warn

from torch import Tensor, nn
import torch.nn.functional as F


class Conv1dEx(nn.Conv1d):
    """Extended Conv1d which support cansal convolution.

    Causal + Stride              is supported (naively).
    Causal + Dilation            is supported.
    Causal + Stride   + Dilation is supported.
    """
    def __init__(self,
        *args,
        padding: str | int | tuple[int, int] = 0,
        padding_mode: str = "zeros",
        causal: bool = False,
        **kwargs
    ):
        """All arguments of `nn.Conv1d`, and new `causal` option"""

        # Conv1d `kernel_size` could be positional or named.
        #   `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, ...)`
        kernel_size: int = args[2] if len(args) >= 3 else kwargs["kernel_size"]
        if len(args) >= 4:
            raise RuntimeError("Currently Conv1dEx needs named argument usage for stride and subsequents.")

        # input_padding: Padding during Conv1dEx forward explicitly
        # conv_padding:  Padding in nn.Conv1d internally
        if causal:
            # Causal + Dilation
            if ("dilation" in kwargs) and (kwargs["dilation"] > 1):
                dilation: int = kwargs["dilation"]
                self._input_padding = ((kernel_size - 1) * dilation, 0)

                # Causal + Dilation + Stride
                if ("stride" in kwargs) and (kwargs["stride"] > 1):
                    raise RuntimeError("Causal + Dilation + Stride is not supported yet.")
            # Causal
            else:
                self._input_padding = (kernel_size - 1, 0)

            # Resolve (conflicted) manual padding arguments
            conv_padding = 0
            if padding != 0:
                if padding == "same":
                    # Automatic 'same' padding match explicit argument
                    pass
                else:
                    warn(f"Conv1dEx: `padding={padding}` is ignored, now using `causal` mode.")

            # Validate unsupported arugments
            if padding_mode != "zeros":
                raise RuntimeError("Currently Conv1dEx support only padding_mode=zeros for causal mode.")

        else:
            self._input_padding = (0, 0)
            conv_padding = padding

        super().__init__(*args, padding=conv_padding, padding_mode=padding_mode, **kwargs)

    def forward(self, x: Tensor):
        """Forward Conv1d with non-uniform padding"""
        return super().forward(F.pad(x, self._input_padding))
