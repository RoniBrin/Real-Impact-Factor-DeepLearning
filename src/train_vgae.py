"""
train_vgae.py - Training utilities for the VGAE model.

This module:
1. Loads a graph dataset
2. Builds the VGAE model
3. Trains the model
4. Evaluates link prediction performance
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges

from vgae_model import build_vgae_model


def load_dataset():
    print("\n========== LOADING DATASET ==========")

    dataset = Planetoid(
        root="data/Planetoid",
        name="PubMed",
        transform=NormalizeFeatures()
    )

    data = dataset[0]

    print("Dataset loaded successfully.")
    print("Number of nodes:", data.num_nodes)
    print("Number of features:", dataset.num_features)
    print("Number of edges:", data.num_edges)

    return dataset, data


def prepare_edges(data):
    print("\n========== PREPARING EDGE SPLITS ==========")

    data = train_test_split_edges(data)

    print("Edge split completed.")

    return data


def train(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    z = model.encode(
        data.x.to(device),
        data.train_pos_edge_index.to(device)
    )

    recon_loss = model.recon_loss(
        z,
        data.train_pos_edge_index.to(device)
    )

    kl_loss = (1 / data.num_nodes) * model.kl_loss()

    loss = recon_loss + kl_loss

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    print("\n========== EVALUATING MODEL ==========")

    model.eval()

    z = model.encode(
        data.x.to(device),
        data.train_pos_edge_index.to(device)
    )

    auc, ap = model.test(
        z,
        data.test_pos_edge_index.to(device),
        data.test_neg_edge_index.to(device)
    )

    print(f"AUC: {auc:.4f}")
    print(f"AP : {ap:.4f}")

    return auc, ap


def run_vgae_training():
    print("\n========== VGAE TRAINING PIPELINE ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    dataset, data = load_dataset()
    data = prepare_edges(data)

    model = build_vgae_model(
        num_features=dataset.num_features,
        hidden_channels=64,
        out_channels=32
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    data = data.to(device)

    epochs = 50

    print("\n========== START TRAINING ==========")

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, device)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

    print("\n========== TRAINING FINISHED ==========")

    evaluate(model, data, device)

    return model


if __name__ == "__main__":
    print("\n========== RUNNING train_vgae.py ==========")

    run_vgae_training()

    print("\n========== VGAE TRAINING COMPLETED ==========")
