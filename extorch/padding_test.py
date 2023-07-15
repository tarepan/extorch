"""Test paddings."""

from .padding import kernel_lr, padding_lr, stride_lr


def test_kernel_lr():
    """Test `kernel_lr`.

    <delta>
    [k1] [k2] [k3] [k4] [k5]   [k6]

     ^    ^-  -^-  -^-- --^-- --^---
    0/0  0/1  1/1  1/2   2/2   2/3


    <causal>
    [k1] [k2] [k3] [k4]  [k5]   [k6]

     ^    -^  --^  ---^ ----^ -----^
    0/0  1/0  2/0   3/0   4/0    5/0
    
    """

    assert kernel_lr(1, "delta") == (0,0)
    assert kernel_lr(2, "delta") == (0,1)
    assert kernel_lr(3, "delta") == (1,1)
    assert kernel_lr(4, "delta") == (1,2)
    assert kernel_lr(5, "delta") == (2,2)
    assert kernel_lr(6, "delta") == (2,3)

    assert kernel_lr(1, "causal") == (0,0)
    assert kernel_lr(2, "causal") == (1,0)
    assert kernel_lr(3, "causal") == (2,0)
    assert kernel_lr(4, "causal") == (3,0)
    assert kernel_lr(5, "causal") == (4,0)
    assert kernel_lr(6, "causal") == (5,0)


def test_stride_lr():
    """Test `stride_lr`.

    <head>
    [k1] [k2] [k3] [k4] [k5]  [k6]

    ^    ^●   ^●●  ^●●● ^●●●● ^●●●●●
    0/0  0/1  0/2  0/3  0/4   0/5

    
    <center>
    [k1] [k2] [k3] [k4] [k5]   [k6]

     ^    ^●  ●^●  ●^●● ●●^●● ●●^●●●
    0/0  0/1  1/1  1/2   2/2   2/3


    <tail>
    [k1] [k2] [k3] [k4]  [k5]   [k6]

     ^    ●^  ●●^  ●●●^ ●●●●^ ●●●●●^
    0/0  1/0  2/0   3/0   4/0    5/0
    
    """

    assert stride_lr(1, "head")   == (0,0)
    assert stride_lr(2, "head")   == (0,1)
    assert stride_lr(3, "head")   == (0,2)
    assert stride_lr(4, "head")   == (0,3)
    assert stride_lr(5, "head")   == (0,4)
    assert stride_lr(6, "head")   == (0,5)

    assert stride_lr(1, "center") == (0,0)
    assert stride_lr(2, "center") == (0,1)
    assert stride_lr(3, "center") == (1,1)
    assert stride_lr(4, "center") == (1,2)
    assert stride_lr(5, "center") == (2,2)
    assert stride_lr(6, "center") == (2,3)

    assert stride_lr(1, "tail")   == (0,0)
    assert stride_lr(2, "tail")   == (1,0)
    assert stride_lr(3, "tail")   == (2,0)
    assert stride_lr(4, "tail")   == (3,0)
    assert stride_lr(5, "tail")   == (4,0)
    assert stride_lr(6, "tail")   == (5,0)


def test_padding_lr_delta_center_drop():
    """Test `padding_lr` with delta kernel, center-aligned frame, and drop-last."""

    # [k_Delta/s_Center]
    #             [k4s1] [k4s2] [k4s3] [k4s4] [k4s5]
    #     frame     ^      ^●    ●^●    ●^●●  ●●^●●
    #     kernel   _^__   _^__   _^__   _^__   _^__
    #     padding  1/2    1/ 1   0/ 1   0/0    0/0

    #              [k5s1]  [k5s2]  [k5s3]  [k5s4]  [k5s5]
    #     frame      ^       ^●     ●^●     ●^●●   ●●^●●
    #     kernel   __^__   __^__   __^__   __^__   __^__
    #     padding   2/2     2/ 1   1 / 1   1 /0     0/0

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "center", True) == (1, 2)
    assert padding_lr(4, "delta", 2, "center", True) == (1, 1)
    assert padding_lr(4, "delta", 3, "center", True) == (0, 1)
    assert padding_lr(4, "delta", 4, "center", True) == (0, 0)
    assert padding_lr(4, "delta", 5, "center", True) == (0, 0)
    assert padding_lr(5, "delta", 1, "center", True) == (2, 2)
    assert padding_lr(5, "delta", 2, "center", True) == (2, 1)
    assert padding_lr(5, "delta", 3, "center", True) == (1, 1)
    assert padding_lr(5, "delta", 4, "center", True) == (1, 0)
    assert padding_lr(5, "delta", 5, "center", True) == (0, 0)


def test_padding_lr_causal_center_drop():
    """Test `padding_lr` with causal kernel, center-aligned frame, and drop-last."""

    # [k_Causal/s_Center]
    #             [k4s1] [k4s2]  [k4s3]   [k4s4]   [k4s5]
    #     frame       ^      ^●     ●^●     ●^●●    ●●^●●
    #     kernel   ___^   ___^    ___^    ___^     ___^
    #     padding    3/0    3/0    2 /0    2 /0    1  /0

    #              [k5s1]  [k5s2]  [k5s3]   [k5s4]    [k5s5]
    #     frame        ^       ^●     ●^●      ●^●●    ●●^●●
    #     kernel   ____^   ____^   ____^    ____^    ____^
    #     padding     4/0     4/0    3 /0     3 /0    2  /0

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "center", True) == (3, 0)
    assert padding_lr(4, "causal", 2, "center", True) == (3, 0)
    assert padding_lr(4, "causal", 3, "center", True) == (2, 0)
    assert padding_lr(4, "causal", 4, "center", True) == (2, 0)
    assert padding_lr(4, "causal", 5, "center", True) == (1, 0)
    assert padding_lr(5, "causal", 1, "center", True) == (4, 0)
    assert padding_lr(5, "causal", 2, "center", True) == (4, 0)
    assert padding_lr(5, "causal", 3, "center", True) == (3, 0)
    assert padding_lr(5, "causal", 4, "center", True) == (3, 0)
    assert padding_lr(5, "causal", 5, "center", True) == (2, 0)


def test_padding_lr_delta_head_drop():
    """Test `padding_lr` with delta kernel, head-aligned frame, and drop-last."""

    # [k_Delta/s_Head]
    #             [k4s1] [k4s2] [k4s3] [k4s4] [k4s5]
    #     frame     ^      ^●    ^●●    ^●●●   ^●●●●
    #     kernel   _^__   _^__  _^__   _^__   _^__
    #     padding  1/2    1/ 1  1/0    1/0    1/0

    #              [k5s1]  [k5s2]  [k5s3]  [k5s4]  [k5s5]
    #     frame      ^       ^●      ^●●     ^●●●    ^●●●●
    #     kernel   __^__   __^__   __^__   __^__   __^__
    #     padding   2/2     2/ 1    2/0     2/0     2/0

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "head", True) == (1, 2)
    assert padding_lr(4, "delta", 2, "head", True) == (1, 1)
    assert padding_lr(4, "delta", 3, "head", True) == (1, 0)
    assert padding_lr(4, "delta", 4, "head", True) == (1, 0)
    assert padding_lr(4, "delta", 5, "head", True) == (1, 0)
    assert padding_lr(5, "delta", 1, "head", True) == (2, 2)
    assert padding_lr(5, "delta", 2, "head", True) == (2, 1)
    assert padding_lr(5, "delta", 3, "head", True) == (2, 0)
    assert padding_lr(5, "delta", 4, "head", True) == (2, 0)
    assert padding_lr(5, "delta", 5, "head", True) == (2, 0)


def test_padding_lr_causal_head_drop():
    """Test `padding_lr` with causal kernel, head-aligned frame, and drop-last."""

    # [k_Causal/s_Head]
    #             [k4s1]  [k4s2]  [k4s3]   [k4s4]   [k4s5]
    #     frame       ^       ^●      ^●●     ^●●●     ^●●●●
    #     kernel   ___^    ___^    ___^    ___^     ___^
    #     padding    3/0     3/0     3/0     3/0      3/0

    #              [k5s1]  [k5s2]  [k5s3]   [k5s4]     [k5s5]
    #     frame        ^       ^●      ^●●      ^●●●      ^●●●●
    #     kernel   ____^   ____^   ____^    ____^     ____^
    #     padding     4/0     4/0     4/0      4/0       4/0

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "head", True) == (3, 0)
    assert padding_lr(4, "causal", 2, "head", True) == (3, 0)
    assert padding_lr(4, "causal", 3, "head", True) == (3, 0)
    assert padding_lr(4, "causal", 4, "head", True) == (3, 0)
    assert padding_lr(4, "causal", 5, "head", True) == (3, 0)
    assert padding_lr(5, "causal", 1, "head", True) == (4, 0)
    assert padding_lr(5, "causal", 2, "head", True) == (4, 0)
    assert padding_lr(5, "causal", 3, "head", True) == (4, 0)
    assert padding_lr(5, "causal", 4, "head", True) == (4, 0)
    assert padding_lr(5, "causal", 5, "head", True) == (4, 0)


def test_padding_lr_delta_tail_drop():
    """Test `padding_lr` with delta kernel, tail-aligned frame, and drop-last."""

    # [k_Delta/s_Tail]
    #             [k4s1] [k4s2] [k4s3]  [k4s4]   [k4s5]
    #     frame     ^     ●^    ●●^     ●●●^    ●●●●^
    #     kernel   _^__   _^__   _^__     _^__     _^__
    #     padding  1/2    0/2    0/2      0/2      0/2

    #             [k5s1]  [k5s2]  [k5s3]  [k5s4]   [k5s5]
    #     frame      ^      ●^     ●●^    ●●●^    ●●●●^
    #     kernel   __^__   __^__   __^__   __^__    __^__
    #     padding   2/2     1/2     0/2     0/2      0/2

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "tail", True) == (1, 2)
    assert padding_lr(4, "delta", 2, "tail", True) == (0, 2)
    assert padding_lr(4, "delta", 3, "tail", True) == (0, 2)
    assert padding_lr(4, "delta", 4, "tail", True) == (0, 2)
    assert padding_lr(4, "delta", 5, "tail", True) == (0, 2)
    assert padding_lr(5, "delta", 1, "tail", True) == (2, 2)
    assert padding_lr(5, "delta", 2, "tail", True) == (1, 2)
    assert padding_lr(5, "delta", 3, "tail", True) == (0, 2)
    assert padding_lr(5, "delta", 4, "tail", True) == (0, 2)
    assert padding_lr(5, "delta", 5, "tail", True) == (0, 2)


def test_padding_lr_causal_tail_drop():
    """Test `padding_lr` with causal kernel, tail-aligned frame, and drop-last."""

    # [k_Causal/s_Tail]
    #             [k4s1] [k4s2]  [k4s3]  [k4s4]  [k4s5]
    #     frame       ^     ●^     ●●^    ●●●^   ●●●●^
    #     kernel   ___^   ___^    ___^    ___^    ___^
    #     padding    3/0   2 /0   1 /0      0/0     0/0

    #              [k5s1]  [k5s2]  [k5s3]   [k5s4]  [k5s5]
    #     frame        ^      ●^     ●●^     ●●●^   ●●●●^
    #     kernel   ____^   ____^   ____^    ____^   ____^
    #     padding     4/0    3 /0   2  /0   1   /0     0/0

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "tail", True) == (3, 0)
    assert padding_lr(4, "causal", 2, "tail", True) == (2, 0)
    assert padding_lr(4, "causal", 3, "tail", True) == (1, 0)
    assert padding_lr(4, "causal", 4, "tail", True) == (0, 0)
    assert padding_lr(4, "causal", 5, "tail", True) == (0, 0)
    assert padding_lr(5, "causal", 1, "tail", True) == (4, 0)
    assert padding_lr(5, "causal", 2, "tail", True) == (3, 0)
    assert padding_lr(5, "causal", 3, "tail", True) == (2, 0)
    assert padding_lr(5, "causal", 4, "tail", True) == (1, 0)
    assert padding_lr(5, "causal", 5, "tail", True) == (0, 0)


def test_padding_lr_delta_center_nodrop():
    """Test `padding_lr` with delta kernel, center-aligned frame, and no drop-last."""

    # [k_Delta/s_Center]
    #                [k4s1]       [k4s2]          [k4s3]           [k4s4]            [k4s5]
    #     frame     ^  ...^^''    ^● ...●^-'   ●^● ...●^●●'--  ●^●●  ...●^●●●'--   ●●^●●...●●^●●●-'--
    #     kernel   _^__..._^__   _^__..._^__   _^__...   _^__    _^__...    _^__   __^__...      _^__
    #     padding  1/  ... /2    1/  ... /2    0/  ...   /3      0/  ...    /3      0/  ...     /4

    #                 [k5s1]         [k5s2]        [k5s3]         [k5s4]            [k5s5]
    #     frame      ^  ...^^''    ^● ...●^-'    ●^● ...●●'--    ●^●●...●●'--   ●●^●●...●●-'--
    #     kernel   __^__..._^__  __^__..._^__   __^__...__^__   __^__...__^__   __^__... __^__
    #     padding   2/  ... /2    2/  ... /2    1 /~ ... /3     1 /~ ... /3     0/~  ... /4

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "center", False) == (1, 2)
    assert padding_lr(4, "delta", 2, "center", False) == (1, 2)
    assert padding_lr(4, "delta", 3, "center", False) == (0, 3)
    assert padding_lr(4, "delta", 4, "center", False) == (0, 3)
    assert padding_lr(4, "delta", 5, "center", False) == (0, 4)
    assert padding_lr(5, "delta", 1, "center", False) == (2, 2)
    assert padding_lr(5, "delta", 2, "center", False) == (2, 2)
    assert padding_lr(5, "delta", 3, "center", False) == (1, 3)
    assert padding_lr(5, "delta", 4, "center", False) == (1, 3)
    assert padding_lr(5, "delta", 5, "center", False) == (0, 4)


def test_padding_lr_causal_center_nodrop():
    """Test `padding_lr` with causal kernel, center-aligned frame, and no drop-last."""

    # [k_Causal/s_Center]
    #             [k4s1]           [k4s2]        [k4s3]          [k4s4]         [k4s5]
    #     frame       ^... ^       ^●...●^      ●^●... ●●'     ●^●●...●●'     ●●^●●...●●-'
    #     kernel   ___^..._^    ___^ ..._^    ___^ ...___^   ___^  ...__^    ___^  ...___^
    #     padding    3/... /0     3/ ... /0    2 / ...  /1    2 /  ... /1    1  /  ... /2

    #                 [k5s1]            [k5s2]     [k5s3]           [k5s4]        [k5s5]
    #     frame        ^...^^        ^●...●^      ●^●...●●'      ●^●●...●●'      ●●^●●...●●-'
    #     kernel   ____^..._^    ____^ ..._^    ____^...__^   ____^  ...__^    ____^  ...___^
    #     padding     4/... /0      4/ ... /0     3 /... /1     3 /0 ... /1     2  /  ... /2

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "center", False) == (3, 0)
    assert padding_lr(4, "causal", 2, "center", False) == (3, 0)
    assert padding_lr(4, "causal", 3, "center", False) == (2, 1)
    assert padding_lr(4, "causal", 4, "center", False) == (2, 1)
    assert padding_lr(4, "causal", 5, "center", False) == (1, 2)
    assert padding_lr(5, "causal", 1, "center", False) == (4, 0)
    assert padding_lr(5, "causal", 2, "center", False) == (4, 0)
    assert padding_lr(5, "causal", 3, "center", False) == (3, 1)
    assert padding_lr(5, "causal", 4, "center", False) == (3, 1)
    assert padding_lr(5, "causal", 5, "center", False) == (2, 2)


def test_padding_lr_delta_head_nodrop():
    """Test `padding_lr` with delta kernel, head-aligned frame, and no drop-last."""

    # [k_Delta/s_Head]
    #                 [k4s1]          [k4s2]       [k4s3]       [k4s4]         [k4s5]
    #     frame     ^  ... ^^--    ^● ... ●^--    ^●●...●^--    ^●●●... ●^--   ^●●●●...●^--
    #     kernel   _^__... _^__   _^__... _^__   _^__..._^__   _^__ ... _^__   _^__ ..._^__
    #     padding  1/  ...  /2    1/  ...  /2    1/  ... /2    1/   ...  /2    1/   ... /2

    #                 [k5s1]         [k5s2]         [k5s3]          [k5s4]         [k5s5]
    #     frame      ^  ...^^--     ^● ...●^--     ^●●...●^--     ^●●●...●^--     ^●●●●...●^--
    #     kernel   __^__..._^__   __^__..._^__   __^__..._^__   __^__ ..._^__   __^__  ..._^__
    #     padding   2/  ... /2     2/  ... /2     2/  ... /2     2/   ... /2     2/    ... /2

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "head", False) == (1, 2)
    assert padding_lr(4, "delta", 2, "head", False) == (1, 2)
    assert padding_lr(4, "delta", 3, "head", False) == (1, 2)
    assert padding_lr(4, "delta", 4, "head", False) == (1, 2)
    assert padding_lr(4, "delta", 5, "head", False) == (1, 2)
    assert padding_lr(5, "delta", 1, "head", False) == (2, 2)
    assert padding_lr(5, "delta", 2, "head", False) == (2, 2)
    assert padding_lr(5, "delta", 3, "head", False) == (2, 2)
    assert padding_lr(5, "delta", 4, "head", False) == (2, 2)
    assert padding_lr(5, "delta", 5, "head", False) == (2, 2)


def test_padding_lr_causal_head_nodrop():
    """Test `padding_lr` with causal kernel, head-aligned frame, and no drop-last."""

    # [k_Causal/s_Head]
    #                 [k4s1]        [k4s2]       [k4s3]         [k4s4]           [k4s5]
    #     frame       ^...^^       ^●...●^       ^●●...●^        ^●●●...●^       ^●●●●...●^
    #     kernel   ___^..._^    ___^ ..._^    ___^  ...__^    ___^   ..._^    ___^    ..._^
    #     padding    3/... /0     3/ ... /0     3/0 ...  /0     3/   ... /0     3/    ... /0

    #                [k5s1]         [k5s2]         [k5s3]           [k5s4]            [k5s5]
    #     frame        ^...^^        ^●...●^        ^●●...●^        ^●●●...●^         ^●●●●...●^
    #     kernel   ____^..._^    ____^ ..._^    ____^  ..._^    ____^   ..._^     ____^    ..._^
    #     padding     4/... /0      4/ ... /0      4/  ... /0      4/   ... /0       4/    ... /0

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "head", False) == (3, 0)
    assert padding_lr(4, "causal", 2, "head", False) == (3, 0)
    assert padding_lr(4, "causal", 3, "head", False) == (3, 0)
    assert padding_lr(4, "causal", 4, "head", False) == (3, 0)
    assert padding_lr(4, "causal", 5, "head", False) == (3, 0)
    assert padding_lr(5, "causal", 1, "head", False) == (4, 0)
    assert padding_lr(5, "causal", 2, "head", False) == (4, 0)
    assert padding_lr(5, "causal", 3, "head", False) == (4, 0)
    assert padding_lr(5, "causal", 4, "head", False) == (4, 0)
    assert padding_lr(5, "causal", 5, "head", False) == (4, 0)


def test_padding_lr_delta_tail_nodrop():
    """Test `padding_lr` with delta kernel, tail-aligned frame, and no drop-last."""

    # [k_Delta/s_Tail]
    #                 [k4s1]       [k4s2]           [k4s3]            [k4s4]            [k4s5]
    #     frame     ^  ...^^--   ●^  ...^●'--   ●●^  ...^●-'--   ●●●^  ...^●--'--   ●●●●^  ...^●---'--
    #     kernel   _^__..._^__   _^__... _^__    _^__...  _^__     _^__...   _^__      _^__...    _^__
    #     padding  1/  ... /2    0/  ... /3      0/  ... /4        0/  ... /5          0/  ... /6

    #                 [k5s1]           [k5s2]       [k5s3]            [k5s4]               [k5s5]
    #     frame      ^  ...^^--    ●^  ... ^●'--   ●●^  ...^●-'--   ●●●^  ...^●--'--   ●●●●^  ...^●---'--
    #     kernel   __^__..._^__   __^__... __^__   __^__... __^__    __^__...  __^__     __^__...   __^__
    #     padding   2/  ... /2     1/  ...  /3      0/  ... /4        0/  ... /5          0/  ... /6

    #                 k           s          drop_last
    assert padding_lr(4, "delta", 1, "tail", False) == (1, 2)
    assert padding_lr(4, "delta", 2, "tail", False) == (0, 3)
    assert padding_lr(4, "delta", 3, "tail", False) == (0, 4)
    assert padding_lr(4, "delta", 4, "tail", False) == (0, 5)
    assert padding_lr(4, "delta", 5, "tail", False) == (0, 6)
    assert padding_lr(5, "delta", 1, "tail", False) == (2, 2)
    assert padding_lr(5, "delta", 2, "tail", False) == (1, 3)
    assert padding_lr(5, "delta", 3, "tail", False) == (0, 4)
    assert padding_lr(5, "delta", 4, "tail", False) == (0, 5)
    assert padding_lr(5, "delta", 5, "tail", False) == (0, 6)


def test_padding_lr_causal_tail_nodrop():
    """Test `padding_lr` with causal kernel, tail-aligned frame, and no drop-last."""

    # [k_Causal/s_Tail]
    #                 [k4s1]     [k4s2]        [k4s3]         [k4s4]      [k4s5]
    #     frame       ^...^^      ●^...^●'    ●●^...^●-'   ●●●^...^●--'   ●●●●^...^●---'
    #     kernel   ___^..._^    ___^...__^   ___^...___^   ___^... ___^    ___^...  ___^
    #     padding    3/... /0    2 /... /1   1  /... /2      0/... /3        0/... /4

    #                 [k5s1]         [k5s2]       [k5s3]         [k5s4]        [k5s5]
    #     frame        ^...^^       ●^...^●'     ●●^...^●-'    ●●●^...^●--'   ●●●●^...^●---'
    #     kernel   ____^..._^    ____^...__^   ____^...___^   ____^...____^   ____^... ____^
    #     padding     4/... /0     3 /... /1    2  /... /2    1   /... /3        0/... /4

    #                 k           s          drop_last
    assert padding_lr(4, "causal", 1, "tail", False) == (3, 0)
    assert padding_lr(4, "causal", 2, "tail", False) == (2, 1)
    assert padding_lr(4, "causal", 3, "tail", False) == (1, 2)
    assert padding_lr(4, "causal", 4, "tail", False) == (0, 3)
    assert padding_lr(4, "causal", 5, "tail", False) == (0, 4)
    assert padding_lr(5, "causal", 1, "tail", False) == (4, 0)
    assert padding_lr(5, "causal", 2, "tail", False) == (3, 1)
    assert padding_lr(5, "causal", 3, "tail", False) == (2, 2)
    assert padding_lr(5, "causal", 4, "tail", False) == (1, 3)
    assert padding_lr(5, "causal", 5, "tail", False) == (0, 4)
