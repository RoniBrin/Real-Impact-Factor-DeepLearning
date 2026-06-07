"""
train_node2vec.py - Training utilities for Node2Vec.
"""

import torch

from node2vec_model import Node2VecModel


def train_node2vec(
    edge_index,
    num_nodes,
    embedding_dim=32,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    epochs=50,
    lr=0.01,
    batch_size=128,
):
    """
    Trains a Node2Vec model and returns node embeddings.
    """

    print("\n========== NODE2VEC TRAINING ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("Number of nodes:", num_nodes)
    print("Number of edges:", edge_index.shape[1])

    model_wrapper = Node2VecModel(
        edge_index=edge_index.to(device),
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        device=device,
    )

    model = model_wrapper.model

    loader = model.loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.SparseAdam(
        list(model.parameters()),
        lr=lr,
    )

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            loss = model.loss(
                pos_rw.to(device),
                neg_rw.to(device),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)

        if epoch % 10 == 0:
            print(f"  Node2Vec Epoch {epoch:>3} | Loss: {avg_loss:.4f}")

    print("Node2Vec training complete!")

    embeddings = model.embedding.weight.detach()

    print("Embeddings shape:", embeddings.shape)

    return embeddings


if __name__ == "__main__":
    print("\n========== RUNNING train_node2vec.py SMOKE TEST ==========")

    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4],
            [1, 2, 2, 3, 4, 0],
        ],
        dtype=torch.long,
    )

    embeddings = train_node2vec(
        edge_index=edge_index,
        num_nodes=5,
        embedding_dim=16,
        epochs=5,
    )

    print("Smoke test embeddings shape:", embeddings.shape)
    print("train_node2vec.py smoke test passed.")
