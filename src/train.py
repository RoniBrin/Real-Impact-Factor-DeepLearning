"""
train.py - Training the GraphSAGE model using link prediction.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from data_loader import load_pubmed
from model import build_model


def train_epoch(model, optimizer, data, neg_ratio=1):
    """
    Runs one training epoch using positive and negative edge sampling.
    Returns the loss value for this epoch.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass - compute embeddings
    z = model(data.x, data.edge_index)

    # Positive edges - edges that exist in the graph
    pos_edge_index = data.edge_index

    # Negative edges - random pairs that are NOT connected
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.shape[1] * neg_ratio
    )

    # Compute scores for positive and negative edges
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # Binary cross entropy loss
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
    Full training loop for GraphSAGE.
    Returns the trained model.
    """
    model = build_model(
        num_features=data.num_node_features,
        hidden_channels=hidden,
        out_channels=out
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, data)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3} | Loss: {loss:.4f}")

    print("Training complete!")
    return model


def save_model(model, path="results/graphsage_trained.pt"):
    """
    Saves the trained model to disk.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_trained_model(num_features, path="results/graphsage_trained.pt", hidden=64, out=32):
    """
    Loads a trained model from disk.
    """
    model = build_model(num_features, hidden, out)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    data, G = load_pubmed()
    model = train(data, epochs=50)
    save_model(model)