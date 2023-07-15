"""Convolution kernel/stride handling."""

from typing import Literal


def left_axis_right(length: int, axis: Literal["head", "center", "tail"]) -> tuple[int, int]:
    """Calculate Left/Right line division based on length and axis position.
    
    Args:
        size - Stride size
        align - Kernel alignment position in a stride (frame)
    """

    # [head]       axis
    #     L10 -> L0 ^●●●●●●●●● R9
    if axis ==  "head":
        left  = 0
        right = (length - 1) - left # == length-1
        return (left, right)

    # [center]         axis
    #     L10 -> L4 ●●●●^●●●●● R5
    if axis == "center":
        left  = (length - 1) // 2
        right = (length - 1) - left
        return (left, right)

    # [tail]                axis
    #     L10 -> L9 ●●●●●●●●●^ R0
    if axis ==  "tail":
        left  = length - 1
        right = (length - 1) - left # == 0
        return (left, right)

    raise RuntimeError(f"Not-supported axis position: {axis}")


def kernel_lr(size: int, shape: Literal["delta", "causal"]) -> tuple[int, int]:
    """Calculate kernel_left and kernel_right size from kernel parameters.
    
    Args:
        size  - Kernel size
        shape - Kernel shape
    """
    if shape == "delta":
        return left_axis_right(size, "center")
    if shape == "causal":
        return left_axis_right(size, "tail")

    raise RuntimeError(f"Not-supported kernel shape: {shape}")


def stride_lr(size: int, align: Literal["head", "center", "tail"]) -> tuple[int, int]:
    """Calculate stride_left and stride_right from stride parameters.
    
    Args:
        size - Stride size
        align - Kernel alignment position in a stride (frame)
    """
    return left_axis_right(size, align)


def padding_lr(
        kernel_size:  int,
        kernel_shape: Literal["delta", "causal"],
        stride_size:  int,
        stride_align: Literal["head", "center", "tail"],
        drop_last:    bool,
    ) -> tuple[int, int]:
    """Calculate padding_left and padding_right."""

    kernel_l, kernel_r = kernel_lr(kernel_size, kernel_shape)
    stride_l, stride_r = stride_lr(stride_size, stride_align)

    padding_l = max(0, kernel_l - stride_l)

    if drop_last:
        padding_r = max(0, kernel_r - stride_r)
    else:
        #       sl-1   kr
        #      |___|^|____|
        #   ●|●------------
        #
        padding_r = max(0, stride_l - 1 + 1 + kernel_r)

    return (padding_l, padding_r)
