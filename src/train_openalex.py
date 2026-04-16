"""
train_openalex.py - Training GraphSAGE on the OpenAlex citation graph.
"""

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from openalex_loader import load_graph
from model import build_model


def build_pyg_data(G):
    """
    Converts OpenAlex NetworkX graph to PyTorch Geometric Data object.
    Uses node degree as the single feature.
    """
    # Remap nodes to consecutive integers
    G = nx.convert_node_labels_to_integers(G)
    num_nodes = G.number_of_nodes()

    # Use degree as node feature
    degrees = torch.tensor(
        [[G.degree(n)] for n in range(num_nodes)],
        dtype=torch.float
    )

    # Build edge_index
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=degrees, edge_index=edge_index, num_nodes=num_nodes), G


def train_epoch(model, optimizer, data):
    """
    Runs one training epoch using positive and negative edge sampling.
    Returns the loss value for this epoch.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    z = model(data.x, data.edge_index)

    # Positive edges
    pos_edge_index = data.edge_index

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.shape[1]
    )

    # Compute scores
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # Loss
    pos_labels = torch.ones(pos_scores.shape[0])
    neg_labels = torch.zeros(neg_scores.shape[0])
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train(data, epochs=50, lr=0.01, hidden=64, out=32):
    """
    Full training loop for GraphSAGE on OpenAlex graph.
    Returns the trained model.
    """
    model = build_model(
        num_features=1,
        hidden_channels=hidden,
        out_channels=out
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting training on OpenAlex graph for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, data)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss:.4f}")

    print("Training complete!")
    return model


if __name__ == "__main__":
    # Load OpenAlex graph
    G = load_graph()
    data, G = build_pyg_data(G)
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

    # Train model
    model = train(data, epochs=50)

    # Save model
    torch.save(model.state_dict(), "results/graphsage_openalex.pt")
    print("Model saved to results/graphsage_openalex.pt")