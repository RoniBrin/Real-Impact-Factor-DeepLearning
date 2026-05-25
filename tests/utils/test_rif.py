"""
test_rif.py - Tests for RIF computation functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import networkx as nx
import pytest
from rif import compute_filtered_rif, compute_weighted_rif


def build_sample_graph():
    G = nx.DiGraph()
    for i in range(25):
        G.add_node(f"p{i}", year=2016, journal="Journal A")
    for i in range(25, 45):
        G.add_node(f"p{i}", year=2017, journal="Journal B")
    # Journal B papers cite Journal A papers
    for i in range(25, 45):
        G.add_edge(f"p{i}", f"p{i-25}")
    return G


def build_stability_scores(G, score=1.0):
    return {
        (min(u, v), max(u, v)): score
        for u, v in G.edges()
    }


def test_filtered_rif_all_stable():
    G                = build_sample_graph()
    stability_scores = build_stability_scores(G, score=1.0)
    filtered_rif     = compute_filtered_rif(G, 2018, stability_scores, threshold=0.7)
    baseline_citations = 20
    baseline_papers    = 25
    assert filtered_rif["Journal A"] == round(baseline_citations / baseline_papers, 4)


def test_filtered_rif_all_unstable():
    G                = build_sample_graph()
    stability_scores = build_stability_scores(G, score=0.0)
    filtered_rif     = compute_filtered_rif(G, 2018, stability_scores, threshold=0.7)
    assert filtered_rif.get("Journal A", 0) == 0.0


def test_weighted_rif_between_zero_and_baseline():
    from graph_builder import compute_baseline_if
    G                = build_sample_graph()
    baseline_if      = compute_baseline_if(G, 2018)
    stability_scores = build_stability_scores(G, score=0.8)
    weighted_rif     = compute_weighted_rif(G, 2018, stability_scores)

    for journal in baseline_if:
        assert 0 <= weighted_rif.get(journal, 0) <= baseline_if[journal]


def test_filtered_rif_leq_baseline_if():
    from graph_builder import compute_baseline_if
    G                = build_sample_graph()
    baseline_if      = compute_baseline_if(G, 2018)
    stability_scores = build_stability_scores(G, score=0.8)
    filtered_rif     = compute_filtered_rif(G, 2018, stability_scores, threshold=0.7)

    for journal in baseline_if:
        assert filtered_rif.get(journal, 0) <= baseline_if[journal]