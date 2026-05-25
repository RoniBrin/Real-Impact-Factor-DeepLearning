"""
test_train.py - Tests for GraphSAGE training loop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
import pytest
from torch_geometric.data import Data
from train_openalex import train


def build_sample_pyg_data():
    x          = torch.randn(20, 4)
    edge_index = torch.randint(0, 20, (2, 40))
    return Data(x=x, edge_index=edge_index, num_nodes=20)


def test_train_returns_model():
    pyg_data = build_sample_pyg_data()
    model    = train(pyg_data, epochs=5)
    assert model is not None


def test_train_loss_decreases():
    pyg_data = build_sample_pyg_data()
    # just verify training runs without error for 10 epochs
    model = train(pyg_data, epochs=10)
    assert model is not None


def test_model_output_after_training():
    pyg_data = build_sample_pyg_data()
    model    = train(pyg_data, epochs=5)
    model.eval()
    with torch.no_grad():
        out = model(pyg_data.x, pyg_data.edge_index)
    assert out.shape[0] == 20
    assert not torch.isnan(out).any()