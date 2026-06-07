"""
node2vec_model.py
Node2Vec embedding model.
"""

import torch
from torch_geometric.nn import Node2Vec


class Node2VecModel:
    """
    Wrapper class for PyTorch Geometric Node2Vec.
    """

    def __init__(
        self,
        edge_index,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        device="cpu"
    ):
        self.device = device

        self.model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=True
        ).to(device)

    def get_embeddings(self):
        """
        Returns the learned node embeddings.
        """
        return self.model.embedding.weight.detach()
