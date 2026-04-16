"""
stability.py - Computing edge stability scores from reconstruction counters.
"""


def compute_stability_scores(reconstruction_counts, removal_counts):
    """
    Computes stability score for each edge.
    Score = successful reconstructions / total removals.
    """
    stability_scores = {}

    for edge in removal_counts:
        removals = removal_counts[edge]
        reconstructions = reconstruction_counts.get(edge, 0)
        stability_scores[edge] = round(reconstructions / removals, 4) if removals > 0 else 0.0

    return stability_scores


def summarize_stability(stability_scores):
    """
    Prints a summary of the stability score distribution.
    """
    scores = list(stability_scores.values())
    total = len(scores)

    low    = sum(1 for s in scores if s < 0.25)
    medium = sum(1 for s in scores if 0.25 <= s < 0.75)
    high   = sum(1 for s in scores if s >= 0.75)

    print(f"\nStability Score Summary ({total} edges):")
    print(f"  Low    (< 0.25) : {low}  ({100*low/total:.1f}%)")
    print(f"  Medium (0.25-0.75): {medium} ({100*medium/total:.1f}%)")
    print(f"  High   (>= 0.75): {high}  ({100*high/total:.1f}%)")


if __name__ == "__main__":
    import torch
    from data_loader import load_pubmed
    from model import build_model
    from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction

    data, G = load_pubmed()
    model = build_model(num_features=data.num_node_features)

    reconstruction_counts = {}
    removal_counts = {}

    # Run 5 perturbation iterations
    print("\nRunning 5 perturbation iterations...")
    model.eval()
    for i in range(5):
        with torch.no_grad():
            perturbed_edge_index, removed_edges = perturb_edges(data.edge_index, fraction=0.3)
            z = model(data.x, perturbed_edge_index)
            scores = compute_reconstruction_scores(z, removed_edges)
        reconstruction_counts, removal_counts = track_reconstruction(
            reconstruction_counts, removal_counts, removed_edges, scores
        )
        print(f"  Iteration {i+1} done")

    # Compute and summarize stability scores
    stability_scores = compute_stability_scores(reconstruction_counts, removal_counts)
    summarize_stability(stability_scores)