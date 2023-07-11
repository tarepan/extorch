"""Test of Conv1dEx"""

import torch
from torch import nn, tensor, equal, no_grad # pylint: disable=no-name-in-module
from .conv1d import Conv1dEx


def test_conv1dex_no_stride():
    """Conv1dEx should support causal convolution w/o stride.
    
    [normal] kernel (2,3,5)
    ipt                  -         1         2         3         -
     -----------------------------------------------------------------
                                 0+3+10    2+6+15    4+9+0
     -----------------------------------------------------------------
    opt                           13        23        13

    [causal] kernel (2,3,5)
    ipt        -         -         1         2         3
     ---------------------------------------------------
                               0+0+5    0+3+10    2+6+15
     ---------------------------------------------------
    opt                            5        13        23
    """

    with no_grad():
        i = tensor([[[1., 2., 3.]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        # conv_normal = Conv1dEx(1, 1, 3, stride=1, padding="same", bias=False, causal=False)
        conv_normal = Conv1dEx(1, 1, 3, stride=1, padding="same", bias=False, causal=False)
        conv_causal = Conv1dEx(1, 1, 3, stride=1, padding="same", bias=False, causal=True)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)

        assert equal(o_normal, tensor([[[13., 23., 13.]]]))
        assert equal(o_causal, tensor([[[ 5., 13., 23.]]]))


def test_conv1dex_with_stride_input_even():
    """Conv1dEx should support causal convolution w/ stride, w/ even length input.
    
    [normal] kernel (2,3,5)
    ipt                  -         1         2         3         4         -
     ---------------------------------------------------------------------------
                                 0+3+10      -       4+9+20      -
     ---------------------------------------------------------------------------
    opt                           13                  33

    [causal] kernel (2,3,5)
    ipt        -         -         1         2         3         4
     -------------------------------------------------------------
                               0+0+5         -    2+6+15         -
     -------------------------------------------------------------
    opt                            5                  23    
    """

    with no_grad():
        i = tensor([[[1., 2., 3., 4.]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=2, padding="scale", bias=False, causal=False)
        conv_causal = Conv1dEx(1, 1, 3, stride=2, padding="scale", bias=False, causal=True)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)

        assert equal(o_normal, tensor([[[13., 33.]]]))
        assert equal(o_causal, tensor([[[ 5., 23.]]]))


def test_conv1dex_with_stride_input_odd():
    """Conv1dEx should support causal convolution w/ stride, w/ odd length input.
    
    [normal] kernel (2,3,5)
    ipt                  -         1         2         3         -
     -----------------------------------------------------------------
                                 0+3+10      -       4+9+0
     -----------------------------------------------------------------
    opt                           13                  13

    [causal] kernel (2,3,5)
    ipt        -         -         1         2         3
     ---------------------------------------------------
                               0+0+5         -    2+6+15
     ---------------------------------------------------
    opt                            5                  23
    """

    with torch.no_grad():
        i = tensor([[[1., 2., 3.,]]])
        kernel = nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=2, padding="scale", bias=False, causal=False)
        conv_causal = Conv1dEx(1, 1, 3, stride=2, padding="scale", bias=False, causal=True)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)

        assert equal(o_normal, tensor([[[13., 13.,]]]))
        assert equal(o_causal, tensor([[[ 5., 23.,]]]))


def test_conv1dex_dilated():
    """Conv1dEx should support causal convolution with dilation.
    
    [normal] effective kernel (2,0,3,0,5)
    ipt        -         -         1         2         3         4         5         -         -
     -----------------------------------------------------------------------------------------------
                                 0+3+15    0+6+20    2+9+25   4+12+0    6+15+0
     -----------------------------------------------------------------------------------------------
    opt                           18        26        36        16        21

    [causal] effective kernel (2,0,3,0,5)
    ipt        -         -         -         -         1         2         3         4         5
     -------------------------------------------------------------------------------------------
                                                   0+0+5    0+0+10    0+3+15    0+6+20    2+9+25
     -------------------------------------------------------------------------------------------
    opt                                                5        10        18        26        36

    """

    with torch.no_grad():
        i = tensor([[[1., 2., 3., 4., 5.,]]])
        kernel = torch.nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, dilation=2, padding="same", bias=False, causal=False)
        conv_causal = Conv1dEx(1, 1, 3, dilation=2, padding="same", bias=False, causal=True)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)

        assert equal(o_normal, tensor([[[18., 26., 36., 16., 21.,]]]))
        assert equal(o_causal, tensor([[[ 5., 10., 18., 26., 36.,]]]))


def test_conv1dex_dilated_stride():
    """Conv1dEx should support causal/strided/dilated convolution.
    
    [normal] effective kernel (2,0,3,0,5)
    ipt        -         -         1         2         3         4         5         -         -
     -----------------------------------------------------------------------------------------------
                                 0+3+15      -       2+9+25             6+15+0
     -----------------------------------------------------------------------------------------------
    opt                           18                  36                  21

    [causal] effective kernel (2,0,3,0,5)
    ipt        -         -         -         -         1         2         3         4         5
     -------------------------------------------------------------------------------------------
                                                   0+0+5              0+3+15              2+9+25
     -------------------------------------------------------------------------------------------
    opt                                                5                  18                  36

    """

    with torch.no_grad():
        i = tensor([[[1., 2., 3., 4., 5.,]]])
        kernel = torch.nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=2, dilation=2, padding="scale", bias=False, causal=False)
        conv_causal = Conv1dEx(1, 1, 3, stride=2, dilation=2, padding="scale", bias=False, causal=True)
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)

        assert equal(o_normal, tensor([[[18., 36., 21.,]]]))
        assert equal(o_causal, tensor([[[ 5., 18., 36.,]]]))
