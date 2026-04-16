"""
model.py - GraphSAGE model for node embedding and link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE encoder.
    Produces node embeddings from graph structure and node features.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # First aggregation layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # Second aggregation layer
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        return x


def decode(z, edge_index):
    """
    Decodes node embeddings into edge probabilities using dot product.
    Returns probability for each edge in edge_index.
    """
    src = z[edge_index[0]]
    dst = z[edge_index[1]]
    return (src * dst).sum(dim=1)


def build_model(num_features, hidden_channels=64, out_channels=32):
    """
    Builds and returns a GraphSAGE model.
    """
    model = GraphSAGE(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )
    print(f"Model built: input={num_features}, hidden={hidden_channels}, output={out_channels}")
    return model


if __name__ == "__main__":
    from data_loader import load_pubmed

    data, G = load_pubmed()

    # Build model with PubMed feature size (500)
    model = build_model(num_features=data.num_node_features)
    print(model)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        print(f"Embedding shape: {z.shape}")