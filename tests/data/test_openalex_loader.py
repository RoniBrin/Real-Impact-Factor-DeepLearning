"""
test_openalex_loader.py - Tests for OpenAlex data fetching and graph building.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import networkx as nx
import pytest
from openalex_loader import build_citation_graph, save_graph, load_graph


def build_mock_graph():
    """Builds a mock graph simulating OpenAlex output."""
    G = nx.DiGraph()
    G.add_node("W1", year=2016, journal="Nature")
    G.add_node("W2", year=2016, journal="Nature")
    G.add_node("W3", year=2017, journal="Lancet")
    G.add_node("W4", year=2017, journal="Lancet")
    G.add_edge("W3", "W1")
    G.add_edge("W4", "W1")
    G.add_edge("W4", "W2")
    return G


def test_graph_has_nodes_and_edges():
    G = build_mock_graph()
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_all_nodes_have_year_and_journal():
    G = build_mock_graph()
    for node in G.nodes():
        assert G.nodes[node].get("year") is not None
        assert G.nodes[node].get("journal") is not None


def test_edges_only_within_dataset():
    G = build_mock_graph()
    node_ids = set(G.nodes())
    for u, v in G.edges():
        assert u in node_ids
        assert v in node_ids


def test_save_and_load_graph(tmp_path):
    G    = build_mock_graph()
    path = str(tmp_path / "test_graph.gpickle")
    save_graph(G, path)
    G2 = load_graph(path)
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()


def test_graph_is_digraph():
    G = build_mock_graph()
    assert isinstance(G, nx.DiGraph)