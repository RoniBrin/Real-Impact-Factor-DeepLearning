"""
train_openalex.py - Trains GraphSAGE on an OpenAlex citation graph.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from model import GraphSAGE


def train(pyg_data, epochs=100, lr=0.01, hidden=128, out=64):
    """
    Trains GraphSAGE with link prediction (dot-product decoder).
    Uses a learning rate scheduler that halves LR every 30 epochs.
    Returns trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = pyg_data.x.shape[1]
    model = GraphSAGE(in_channels, hidden, out).to(device)

    print(f"Model built: input={in_channels}, hidden={hidden}, output={out}, device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    x          = pyg_data.x.to(device)
    edge_index = pyg_data.edge_index.to(device)
    num_nodes  = pyg_data.num_nodes
    num_edges  = edge_index.shape[1]

    print(f"\nStarting training for {epochs} epochs...")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        z = model(x, edge_index)

        # positive pairs: real edges
        src, dst = edge_index[0], edge_index[1]
        pos_score = (z[src] * z[dst]).sum(dim=1)

        # negative pairs: random non-edges
        neg_src = torch.randint(0, num_nodes, (num_edges,), device=device)
        neg_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
        neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)

        # binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

    print("Training complete!")
    return model