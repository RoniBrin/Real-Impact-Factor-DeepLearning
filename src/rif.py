"""
rif.py - Computing the Real Impact Factor (RIF) using edge stability scores.
"""


def compute_filtered_rif(G, target_year, stability_scores, threshold=0.5):
    """
    Computes Filtered RIF for each journal in target year Y.
    Excludes citations with stability score below the threshold.
    """
    relevant_nodes = set(
        node for node in G.nodes()
        if G.nodes[node].get('year') in (target_year - 1, target_year - 2)
    )

    journal_citations = {}
    journal_papers = {}

    for node in relevant_nodes:
        journal = G.nodes[node].get('journal', 'Unknown')
        journal_papers[journal] = journal_papers.get(journal, 0) + 1

        for neighbor in G.neighbors(node):
            if neighbor not in relevant_nodes:
                edge = (min(node, neighbor), max(node, neighbor))
                score = stability_scores.get(edge, 0.0)
                # Only count citation if stability score exceeds threshold
                if score >= threshold:
                    journal_citations[journal] = journal_citations.get(journal, 0) + 1

    filtered_rif = {}
    for journal in journal_papers:
        papers = journal_papers[journal]
        citations = journal_citations.get(journal, 0)
        filtered_rif[journal] = round(citations / papers, 4) if papers > 0 else 0.0

    return filtered_rif


def compute_weighted_rif(G, target_year, stability_scores):
    """
    Computes Weighted RIF for each journal in target year Y.
    Each citation is weighted by its stability score.
    """
    relevant_nodes = set(
        node for node in G.nodes()
        if G.nodes[node].get('year') in (target_year - 1, target_year - 2)
    )

    journal_citations = {}
    journal_papers = {}

    for node in relevant_nodes:
        journal = G.nodes[node].get('journal', 'Unknown')
        journal_papers[journal] = journal_papers.get(journal, 0) + 1

        for neighbor in G.neighbors(node):
            if neighbor not in relevant_nodes:
                edge = (min(node, neighbor), max(node, neighbor))
                score = stability_scores.get(edge, 0.0)
                # Weight citation by stability score
                journal_citations[journal] = journal_citations.get(journal, 0) + score

    weighted_rif = {}
    for journal in journal_papers:
        papers = journal_papers[journal]
        citations = journal_citations.get(journal, 0)
        weighted_rif[journal] = round(citations / papers, 4) if papers > 0 else 0.0

    return weighted_rif


def print_rif_comparison(baseline_if, filtered_rif, weighted_rif, target_year):
    """
    Prints a comparison table of Baseline IF, Filtered RIF, and Weighted RIF.
    """
    print(f"\nIF vs RIF Comparison for year {target_year}:")
    print(f"{'Journal':<20} {'Baseline IF':>12} {'Filtered RIF':>13} {'Weighted RIF':>13}")
    print("-" * 60)
    for journal in baseline_if:
        b_if = baseline_if.get(journal, 0)
        f_rif = filtered_rif.get(journal, 0)
        w_rif = weighted_rif.get(journal, 0)
        print(f"{journal:<20} {b_if:>12} {f_rif:>13} {w_rif:>13}")


if __name__ == "__main__":
    import torch
    from data_loader import load_pubmed
    from graph_builder import assign_synthetic_metadata, compute_baseline_if
    from model import build_model
    from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction
    from stability import compute_stability_scores

    data, G = load_pubmed()
    G = assign_synthetic_metadata(G)
    model = build_model(num_features=data.num_node_features)

    # Run perturbation iterations
    reconstruction_counts = {}
    removal_counts = {}

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

    # Compute stability scores
    stability_scores = compute_stability_scores(reconstruction_counts, removal_counts)

    # Compute all metrics
    target_year = 2010
    baseline_if = compute_baseline_if(G, target_year)
    filtered_rif = compute_filtered_rif(G, target_year, stability_scores)
    weighted_rif = compute_weighted_rif(G, target_year, stability_scores)

    print_rif_comparison(baseline_if, filtered_rif, weighted_rif, target_year)