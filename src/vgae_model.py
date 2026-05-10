"""
vgae_model.py - Variational Graph Autoencoder model for link prediction.

This file defines the VGAE architecture only.
The training logic should be implemented separately in train_vgae.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE


class VGAEEncoder(nn.Module):
    """
    Encoder for a Variational Graph Autoencoder.

    The encoder receives:
    - x: node features
    - edge_index: graph connectivity

    It returns:
    - mu: mean vector of the latent representation
    - logstd: log standard deviation vector of the latent representation
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super(VGAEEncoder, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        print("[VGAEEncoder] Forward started")

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)

        print("[VGAEEncoder] Forward completed")
        print("[VGAEEncoder] mu shape:", mu.shape)
        print("[VGAEEncoder] logstd shape:", logstd.shape)

        return mu, logstd


def build_vgae_model(num_features, hidden_channels=64, out_channels=32):
    """
    Builds and returns a PyTorch Geometric VGAE model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = VGAEEncoder(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )

    model = VGAE(encoder).to(device)

    print(
        f"VGAE model built: input={num_features}, "
        f"hidden={hidden_channels}, output={out_channels}, device={device}"
    )

    return model


if __name__ == "__main__":
    print("Running VGAE model smoke test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = 5
    num_features = 3

    x = torch.randn((num_nodes, num_features), dtype=torch.float).to(device)

    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=torch.long).to(device)

    model = build_vgae_model(
        num_features=num_features,
        hidden_channels=8,
        out_channels=4
    )

    model.eval()

    with torch.no_grad():
        z = model.encode(x, edge_index)

    print("Latent embedding shape:", z.shape)
    print("VGAE model smoke test completed successfully.")
