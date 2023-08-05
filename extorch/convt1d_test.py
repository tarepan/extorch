"""Test of ConvT1dEx"""

import torch
from torch import nn, tensor, allclose # pylint: disable=no-name-in-module
from .convt1d import ConvT1dEx


def test_convt1dex_no_stride():
    """ConvT1dEx should support causal convolution w/o stride.

    [normal] kernel (2,3,5)
    ipt                            1         2         3
     -----------------------------------------------------------------
                         2         3         5
                                   4         6        10
                                             6         9        15
     -----------------------------------------------------------------
    opt                  -         7        17        19         -

    [causal] kernel (2,3,5)
    ipt                            1         2         3
     ---------------------------------------------------------------------------
                                   2         3         5
                                             4         6        10
                                                       6         9        15
     ---------------------------------------------------------------------------
    opt                            2         7        17         -         -
    """

    with torch.no_grad():
        ipt = tensor([[[1., 2., 3.]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = ConvT1dEx(1, 1, 3,                 stride=1, padding="same", bias=False)
        conv_causal = ConvT1dEx(1, 1, 3, shape="causal", stride=1, padding="same", bias=False)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        opt_normal = conv_normal(ipt)
        opt_causal = conv_causal(ipt)

        assert allclose(opt_normal, tensor([[[ 7., 17., 19.]]]))
        assert allclose(opt_causal, tensor([[[ 2.,  7., 17.]]]))


def test_convt1dex_with_stride():
    """ConvT1dEx should support causal convolution w/ stride.

    [normal] kernel (2,3,5)
    ipt                       .    1    .    2    .    3    .
     -----------------------------------------------------------------
                              2    3    5
                                        4    6   10
                                                  6    9   15
     -----------------------------------------------------------------
    opt                       -    3    9    6   16    9   15

    [causal] kernel (2,3,5)
    ipt                            1    .    2    .    3    .
     -----------------------------------------------------------------
                                   2    3    5
                                             4    6   10
                                                       6    9   15
     -----------------------------------------------------------------
    opt                            2    3    9    6   16    9    -
    """

    with torch.no_grad():
        ipt = tensor([[[1., 2., 3.]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = ConvT1dEx(1, 1, 3,                 stride=2, padding="scale", bias=False)
        conv_causal = ConvT1dEx(1, 1, 3, shape="causal", stride=2, padding="scale", bias=False)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel
        o_normal = conv_normal(ipt)
        o_causal = conv_causal(ipt)

        print(o_normal)
        print(o_causal)
        assert allclose(o_normal, tensor([[[ 3.,  9.,  6., 16.,  9., 15.]]]))
        assert allclose(o_causal, tensor([[[ 2.,  3.,  9.,  6., 16.,  9.]]]))


def test_convt1dex_with_stride_dilation():
    """ConvT1dEx should support causal convolution w/ stride and dilation.

    [normal] effective kernel (2,0,3,0,5)
    ipt                  .    .    1    .    2    .    3    .    .
     -----------------------------------------------------------------
                         2    0    3    0    5
                                   4    0    6    0   10
                                             6    0    9    0   15
     -----------------------------------------------------------------
    opt                  -    -    7    0   17    0   19    0    -

    [causal] effective kernel (2,0,3,0,5)
    ipt                            1    .    2    .    3    .    .    .    .
     --------------------------------------------------------------------------
                                   2    0    3    0    5
                                             4    0    6    0   10
                                                       6    0    9    0   15
     --------------------------------------------------------------------------
    opt                            2    0    7    0   17    0    -    -    -
    """

    with torch.no_grad():
        ipt = tensor([[[1., 2., 3.]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = ConvT1dEx(1, 1, 3, stride=2,                 dilation=2, padding="scale", bias=False)
        conv_causal = ConvT1dEx(1, 1, 3, stride=2, shape="causal", dilation=2, padding="scale", bias=False)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel
        o_normal = conv_normal(ipt)
        o_causal = conv_causal(ipt)

        print(o_normal)
        print(o_causal)
        assert allclose(o_normal, tensor([[[ 7.,  0., 17.,  0., 19., 0.]]]))
        assert allclose(o_causal, tensor([[[ 2.,  0.,  7.,  0., 17., 0.]]]))
