"""
data_loader.py - Loading the PubMed dataset from Planetoid.
"""

import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.utils import to_networkx


def load_pubmed():
    """
    Loads the PubMed dataset from Planetoid.
    Returns the PyG data object and a NetworkX graph.
    """
    print("Loading PubMed dataset...")
    dataset = Planetoid(root='data/PubMed', name='PubMed')
    data = dataset[0]

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Node features: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")

    # Convert to NetworkX for graph analysis
    G = to_networkx(data, to_undirected=True)
    print(f"NetworkX graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return data, G


if __name__ == "__main__":
    data, G = load_pubmed()