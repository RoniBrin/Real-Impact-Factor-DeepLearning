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