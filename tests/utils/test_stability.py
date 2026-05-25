"""
test_stability.py - Tests for stability score computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest
from stability import compute_stability_scores, summarize_stability


def test_stability_score_range():
    reconstruction_counts = {(0,1): 80, (1,2): 50, (2,3): 10}
    removal_counts        = {(0,1): 100, (1,2): 100, (2,3): 100}
    scores = compute_stability_scores(reconstruction_counts, removal_counts)

    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_stability_score_values():
    reconstruction_counts = {(0,1): 100, (1,2): 0}
    removal_counts        = {(0,1): 100, (1,2): 100}
    scores = compute_stability_scores(reconstruction_counts, removal_counts)

    assert scores[(0,1)] == 1.0
    assert scores[(1,2)] == 0.0


def test_stability_zero_removals():
    reconstruction_counts = {}
    removal_counts        = {(0,1): 0}
    scores = compute_stability_scores(reconstruction_counts, removal_counts)
    assert scores[(0,1)] == 0.0


def test_stability_all_reconstructed():
    n     = 100
    edges = [(i, i+1) for i in range(n)]
    reconstruction_counts = {e: 100 for e in edges}
    removal_counts        = {e: 100 for e in edges}
    scores = compute_stability_scores(reconstruction_counts, removal_counts)

    assert all(s == 1.0 for s in scores.values())