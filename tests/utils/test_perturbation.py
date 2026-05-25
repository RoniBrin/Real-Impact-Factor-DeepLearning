"""
test_perturbation.py - Tests for edge perturbation functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import torch
import pytest
from perturbation import perturb_edges, compute_reconstruction_scores


def build_edge_index(num_nodes=20, num_edges=40):
    return torch.randint(0, num_nodes, (2, num_edges))


def test_perturb_edges_removes_correct_fraction():
    edge_index = build_edge_index(num_edges=100)
    perturbed, removed = perturb_edges(edge_index, fraction=0.3)

    assert removed.shape[1]   == 30
    assert perturbed.shape[1] == 70


def test_perturb_edges_no_overlap():
    edge_index = build_edge_index(num_edges=100)
    perturbed, removed = perturb_edges(edge_index, fraction=0.3)

    total = perturbed.shape[1] + removed.shape[1]
    assert total == edge_index.shape[1]


def test_reconstruction_scores_shape():
    from model import GraphSAGE
    model      = GraphSAGE(in_channels=4, hidden_channels=64, out_channels=32)
    x          = torch.randn(20, 4)
    edge_index = build_edge_index(num_nodes=20, num_edges=40)
    model.eval()
    with torch.no_grad():
        z      = model(x, edge_index)
        _, removed = perturb_edges(edge_index, fraction=0.3)
        scores = compute_reconstruction_scores(z, removed)

    assert scores.shape[0] == removed.shape[1]


def test_reconstruction_scores_range():
    from model import GraphSAGE
    model      = GraphSAGE(in_channels=4, hidden_channels=64, out_channels=32)
    x          = torch.randn(20, 4)
    edge_index = build_edge_index(num_nodes=20, num_edges=40)
    model.eval()
    with torch.no_grad():
        z      = model(x, edge_index)
        _, removed = perturb_edges(edge_index, fraction=0.3)
        scores = compute_reconstruction_scores(z, removed)

    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()