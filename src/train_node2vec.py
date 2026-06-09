"""
train_node2vec.py - Training Node2Vec using the node2vec package (not PyG).
Works with NetworkX graphs directly.
"""

import torch
import numpy as np
import networkx as nx
from node2vec import Node2Vec as N2V


def train_node2vec(G, embedding_dim=64, walk_length=20, num_walks=10,
                   workers=1, epochs=50, window=10):
    """
    Trains Node2Vec on a NetworkX graph.
    Returns node embeddings as a torch tensor indexed by integer node id.
    """
    print("\n========== NODE2VEC TRAINING ==========")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # Node2Vec requires undirected graph
    G_undirected = G.to_undirected()

    # train Node2Vec
    node2vec = N2V(
        G_undirected,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        quiet=True,
    )

    model = node2vec.fit(
        window=window,
        min_count=1,
        batch_words=4,
        epochs=epochs,
    )

    print("Node2Vec training complete!")

    # build embedding tensor indexed by integer node id
    nodes      = list(G.nodes())
    embeddings = np.stack([model.wv[str(node)] for node in nodes])
    z          = torch.tensor(embeddings, dtype=torch.float)

    print(f"Embeddings shape: {z.shape}")
    return z, nodes