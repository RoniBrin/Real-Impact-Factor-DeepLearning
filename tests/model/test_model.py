"""
test_model.py - Tests for GraphSAGE model architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
import pytest
from model import GraphSAGE


def test_model_output_shape():
    model      = GraphSAGE(in_channels=4, hidden_channels=128, out_channels=64)
    x          = torch.randn(10, 4)
    edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    out        = model(x, edge_index)
    assert out.shape == (10, 64)


def test_model_has_three_layers():
    model = GraphSAGE(in_channels=4, hidden_channels=128, out_channels=64)
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")
    assert hasattr(model, "conv3")


def test_model_forward_no_nan():
    model      = GraphSAGE(in_channels=4, hidden_channels=128, out_channels=64)
    x          = torch.randn(10, 4)
    edge_index = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    out        = model(x, edge_index)
    assert not torch.isnan(out).any()


def test_model_different_input_sizes():
    for in_ch in [1, 4, 8]:
        model = GraphSAGE(in_channels=in_ch, hidden_channels=64, out_channels=32)
        x     = torch.randn(5, in_ch)
        edge_index = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        out   = model(x, edge_index)
        assert out.shape == (5, 32)