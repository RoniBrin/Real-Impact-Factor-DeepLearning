"""
perturbation.py - Random edge removal and reconstruction tracking.
"""

import torch


def perturb_edges(edge_index, fraction=0.3):
    """
    Randomly removes a fraction of edges from the graph.
    Returns the perturbed edge_index and the removed edges.
    """
    num_edges = edge_index.shape[1]
    num_remove = int(num_edges * fraction)

    # Randomly select edges to remove using torch
    perm = torch.randperm(num_edges, device=edge_index.device)
    removed_indices = perm[:num_remove]
    kept_indices = perm[num_remove:]

    perturbed_edge_index = edge_index[:, kept_indices]
    removed_edges = edge_index[:, removed_indices]

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
    Updates reconstruction counters for each edge using vectorized operations.
    """
    # Move to CPU for dictionary operations
    u = removed_edges[0].cpu().numpy()
    v = removed_edges[1].cpu().numpy()
    s = scores.cpu().numpy()

    for i in range(len(u)):
        edge = (int(min(u[i], v[i])), int(max(u[i], v[i])))
        removal_counts[edge] = removal_counts.get(edge, 0) + 1
        if s[i] > threshold:
            reconstruction_counts[edge] = reconstruction_counts.get(edge, 0) + 1

    return reconstruction_counts, removal_counts