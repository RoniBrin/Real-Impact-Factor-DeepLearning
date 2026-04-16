"""
perturbation.py - Random edge removal and reconstruction tracking.
"""

import torch
import random


def perturb_edges(edge_index, fraction=0.3):
    """
    Randomly removes a fraction of edges from the graph.
    Returns the perturbed edge_index and the removed edges.
    """
    num_edges = edge_index.shape[1]
    num_remove = int(num_edges * fraction)

    # Randomly select edges to remove
    all_indices = list(range(num_edges))
    removed_indices = random.sample(all_indices, num_remove)
    kept_indices = [i for i in all_indices if i not in removed_indices]

    # Split into kept and removed
    perturbed_edge_index = edge_index[:, kept_indices]
    removed_edges = edge_index[:, removed_indices]

    print(f"Total edges: {num_edges}")
    print(f"Removed: {num_remove} ({fraction*100:.0f}%)")
    print(f"Remaining: {perturbed_edge_index.shape[1]}")

    return perturbed_edge_index, removed_edges


def compute_reconstruction_scores(z, removed_edges):
    """
    Computes similarity scores for removed edges using dot product.
    Returns a score tensor for each removed edge.
    """
    src = z[removed_edges[0]]
    dst = z[removed_edges[1]]
    scores = (src * dst).sum(dim=1)
    return torch.sigmoid(scores)


def track_reconstruction(reconstruction_counts, removal_counts, removed_edges, scores, threshold=0.5):
    """
    Updates reconstruction counters for each edge.
    An edge is considered reconstructed if its score exceeds the threshold.
    """
    for i in range(removed_edges.shape[1]):
        u = removed_edges[0][i].item()
        v = removed_edges[1][i].item()
        edge = (min(u, v), max(u, v))

        # Update removal count
        removal_counts[edge] = removal_counts.get(edge, 0) + 1

        # Update reconstruction count if score exceeds threshold
        if scores[i].item() > threshold:
            reconstruction_counts[edge] = reconstruction_counts.get(edge, 0) + 1

    return reconstruction_counts, removal_counts


if __name__ == "__main__":
    from data_loader import load_pubmed
    from model import build_model

    data, G = load_pubmed()
    model = build_model(num_features=data.num_node_features)

    # Test one perturbation iteration
    model.eval()
    with torch.no_grad():
        perturbed_edge_index, removed_edges = perturb_edges(data.edge_index, fraction=0.3)
        z = model(data.x, perturbed_edge_index)
        scores = compute_reconstruction_scores(z, removed_edges)

    print(f"\nReconstruction scores sample (first 5):")
    print(scores[:5])

    # Test tracking
    reconstruction_counts = {}
    removal_counts = {}
    reconstruction_counts, removal_counts = track_reconstruction(
        reconstruction_counts, removal_counts, removed_edges, scores
    )
    print(f"\nTracked {len(removal_counts)} unique edges")