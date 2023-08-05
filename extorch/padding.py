"""Convolution kernel/stride handling."""

from typing import Literal


def left_axis_right(length: int, axis: Literal["head", "center", "tail"]) -> tuple[int, int]:
    """Calculate Left/Right length when target is devided into Left/Axis/Right based on length and axis position.
    
    Args:
        length - Target length
        axis   - Kernel alignment position in a stride (frame)
    Returns:
               - (Left, Right), Left <= Right for center alignment
    """

    # [head]            axis
    #     length10 -> L0 ^●●●●●●●●● R9
    if axis == "head":
        left  = 0
        right = (length - 1) - left # == length-1
        return (left, right)

    # [center]              axis
    #     length10 -> L4 ●●●●^●●●●● R5
    if axis == "center":
        left  = (length - 1) // 2
        right = (length - 1) - left
        return (left, right)

    # [tail]                     axis
    #     length10 -> L9 ●●●●●●●●●^ R0
    if axis ==  "tail":
        left  = length - 1
        right = (length - 1) - left # == 0
        return (left, right)

    raise RuntimeError(f"Not-supported axis position: {axis}")


def kernel_lr(size: int, shape: Literal["delta", "causal", "inv_causal"]) -> tuple[int, int]:
    """Calculate kernel_left and kernel_right size from kernel parameters.
    
    Args:
        size  - Kernel size
        shape - Kernel shape, centered 'delta' | right-aligned 'causal' | left-aligned 'inv_causal'
    """
    if shape == "delta":
        return left_axis_right(size, "center")
    if shape == "causal":
        return left_axis_right(size, "tail")
    if shape == "inv_causal":
        return left_axis_right(size, "head")

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
        kernel_shape: Literal["delta", "causal", "inv_causal"],
        stride_size:  int,
        stride_align: Literal["head", "center", "tail"],
        drop_last:    bool,
    ) -> tuple[int, int]:
    """Calculate padding_left and padding_right.
    
    Args:
        drop_last - Whether to drop the last frame if kernel is not fulfilled in the frame. If False, pad for the frame.
    """

    kernel_l, kernel_r = kernel_lr(kernel_size, kernel_shape)
    stride_l, stride_r = stride_lr(stride_size, stride_align)

    # [Padding Left] In all alignment, pl = max(0, kl-sl)
    #
    #       center                  tail             head
    #        axis                   axis             axis
    #     kl  |                kl    |            kl  |
    #   |____||           |_________||          |____||
    #         |                      |                |
    #      sl |                sl    |           sl=0 |
    #     |__||             |_______||                |
    #   --●●●●●●●●●●●     --●●●●●●●●●●●         ------●●●●●●●●●●●
    #     |________|        |________|                |________|
    #       stride            stride                    stride
    padding_l = max(0, kernel_l - stride_l)

    if drop_last:
        # [Padding Right drop_last]
        #
        #       center               tail           head
        #        axis                axis           axis
        #         |  kr                |  kr          |   kr
        #         ||____|              ||____|        ||_______|
        #         |                    |              |
        #         | sr                 | sr=0         |  sr
        #         ||__|                |              ||_____|
        #     ●●●●●●●●●???     ●●●●●●●●●??????       ●●●●●●●●●????
        #      |______|         |______|              |______|
        #       stride           stride                stride
        #
        # Even if stride is not fulfilled under kr < sr config, kernel is fulfilled so there is no semantical needs of drop_last (so not dropped.)
        # [kr<sr] Zero padding and non-fulfilled stride, but correctly non drop_last
        #           center
        #            axis
        #             | kr
        #             ||__|
        #             |
        #             |   sr
        #             ||______|
        #     ●●●●●●●●●●●●●xxxx
        #      |______________|
        #           stride
        padding_r = max(0, kernel_r - stride_r)
    else:
        # [Padding Right no drop_last]
        #
        #          center                tail           head
        #           axis                 axis           axis
        #            |  kr                 |  kr          |   kr
        #            ||_____|              ||____|        ||_______|
        #            |                     |              |
        #       sl-1 |                sl-1 |              |
        #       |___||               |____||              |
        #     ●●-------------      ●●-------------       ●●---------
        #     _|__________|        _|______|             _|______|
        #         stride             stride                stride
        padding_r = max(0, stride_l - 1 + 1 + kernel_r)

    return (padding_l, padding_r)
