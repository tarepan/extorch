"""Test of Conv1dEx"""

import numpy as np
import torch
from torch import tensor, equal
from extorch import Conv1dEx


def test_conv1dex_no_stride():
    """Conv1dEx should support causal convolution w/o stride"""

    with torch.no_grad():
        i = torch.tensor([[[1., 2., 3.]]])
        kernel = torch.nn.Parameter(torch.tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=1, padding=1,   bias=False, padding_mode='zeros')
        conv_causal = Conv1dEx(1, 1, 3, stride=1, causal=True, bias=False, padding_mode='zeros')
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)
        # Test
        assert all((o_normal.numpy() == np.array([[[13., 23., 13.]]]))[0][0])
        assert all((o_causal.numpy() == np.array([[[ 5., 13., 23.]]]))[0][0])


def test_conv1dex_with_stride_input_even():
    """Conv1dEx should support causal convolution w/ stride, w/ even length input"""

    with torch.no_grad():
        i = torch.tensor([[[1., 2., 3., 4.]]])
        kernel = torch.nn.Parameter(torch.tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=2, padding=1,   bias=False, padding_mode='zeros')
        conv_causal = Conv1dEx(1, 1, 3, stride=2, causal=True, bias=False, padding_mode='zeros')
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel
        o_normal = conv_normal(i)
        o_causal = conv_causal(i)
        ## Test
        assert all((o_normal.numpy() == np.array([[[13., 33.]]]))[0][0])
        assert all((o_causal.numpy() == np.array([[[ 5., 23.]]]))[0][0])


def test_conv1dex_with_stride_input_odd():
    """Conv1dEx should support causal convolution w/ stride, w/ odd length input"""

    with torch.no_grad():
        i = torch.tensor([[[1., 2., 3., 4., 5.]]])
        kernel = torch.nn.Parameter(torch.tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, stride=2, padding=1,   bias=False, padding_mode='zeros')
        conv_causal = Conv1dEx(1, 1, 3, stride=2, causal=True, bias=False, padding_mode='zeros')
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel
        o_normal = conv_normal(i)
        o_causal = conv_causal(i)
        ## Test
        assert all((o_normal.numpy() == np.array([[[13., 33., 23.]]]))[0][0])
        assert all((o_causal.numpy() == np.array([[[ 5., 23., 43.]]]))[0][0])


def test_conv1dex_dilated():
    """Conv1dEx should support causal convolution with dilation."""

    with torch.no_grad():
        i = tensor([[[1., 2., 3., 4., 5.,]]])
        kernel = torch.nn.Parameter(tensor([2., 3., 5.]))
        conv_normal = Conv1dEx(1, 1, 3, dilation=2,              padding="same", bias=False, padding_mode='zeros')
        conv_causal = Conv1dEx(1, 1, 3, dilation=2, causal=True, padding="same", bias=False, padding_mode='zeros')
        conv_normal.weight[0][0] = kernel
        conv_causal.weight[0][0] = kernel

        o_normal = conv_normal(i)
        o_causal = conv_causal(i)
        # Test
        assert equal(o_normal, tensor([[[18., 26., 36., 16., 21.,]]]))
        assert equal(o_causal, tensor([[[ 5., 10., 18., 26., 36.,]]]))
