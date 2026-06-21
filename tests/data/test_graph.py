"""
test_graph.py - Tests for citation graph construction and IF computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import networkx as nx
import pytest
from graph_builder import compute_baseline_if
from rif import compute_filtered_rif, compute_weighted_rif


def build_sample_graph():
    """Builds a sample citation graph with enough papers to pass MIN_PAPERS filter."""
    G = nx.DiGraph()

    # Journal A: 25 papers in 2016-2017
    for i in range(25):
        G.add_node(f"a{i}", year=2016 if i < 13 else 2017, journal="Journal A")

    # Journal B: 25 papers in 2016-2017
    for i in range(25):
        G.add_node(f"b{i}", year=2016 if i < 13 else 2017, journal="Journal B")

    # Journal B papers cite Journal A papers
    for i in range(20):
        G.add_edge(f"b{i}", f"a{i}")

    return G


def build_stability_scores(G, score=1.0):
    return {
        (min(u, v), max(u, v)): score
        for u, v in G.edges()
    }


def test_graph_nodes_have_required_attributes():
    G = build_sample_graph()
    for node in G.nodes():
        assert "year"    in G.nodes[node]
        assert "journal" in G.nodes[node]


def test_graph_edges_within_dataset():
    G = build_sample_graph()
    node_ids = set(G.nodes())
    for u, v in G.edges():
        assert u in node_ids
        assert v in node_ids


def test_graph_is_directed():
    G = build_sample_graph()
    assert isinstance(G, nx.DiGraph)


def test_baseline_if_computation():
    G = build_sample_graph()
    baseline_if = compute_baseline_if(G, 2018)

    # Journal A: 25 papers, 20 citations received -> IF = 0.8
    assert "Journal A" in baseline_if
    assert baseline_if["Journal A"] == 0.8


def test_baseline_if_only_includes_journals_with_citations():
    G = build_sample_graph()
    baseline_if = compute_baseline_if(G, 2018)

    # Journal B cites others but receives no citations -> not in baseline_if
    assert "Journal B" not in baseline_if


def test_filtered_rif_leq_baseline_if():
    G                = build_sample_graph()
    baseline_if      = compute_baseline_if(G, 2018)
    stability_scores = build_stability_scores(G, score=1.0)
    filtered_rif     = compute_filtered_rif(G, 2018, stability_scores, threshold=0.7)

    for journal in baseline_if:
        assert filtered_rif.get(journal, 0) <= baseline_if[journal], \
            f"{journal}: filtered_rif > baseline_if"


def test_filtered_rif_zero_when_all_unstable():
    G                = build_sample_graph()
    stability_scores = build_stability_scores(G, score=0.0)
    filtered_rif     = compute_filtered_rif(G, 2018, stability_scores, threshold=0.7)

    assert filtered_rif.get("Journal A", 0) == 0.0


def test_weighted_rif_leq_baseline_if():
    G                = build_sample_graph()
    baseline_if      = compute_baseline_if(G, 2018)
    stability_scores = build_stability_scores(G, score=0.8)
    weighted_rif     = compute_weighted_rif(G, 2018, stability_scores)

    for journal in baseline_if:
        assert weighted_rif.get(journal, 0) <= baseline_if[journal], \
            f"{journal}: weighted_rif > baseline_if"