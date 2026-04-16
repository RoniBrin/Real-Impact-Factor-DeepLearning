"""
main.py - Main pipeline for RIF evaluation across multiple years.
"""

import torch
import pandas as pd
from data_loader import load_pubmed
from graph_builder import assign_synthetic_metadata, compute_baseline_if
from model import build_model
from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction
from stability import compute_stability_scores
from rif import compute_filtered_rif, compute_weighted_rif, print_rif_comparison

# Configuration
N_ITERATIONS = 5      # Number of perturbation iterations per year
FRACTION = 0.3        # Fraction of edges to remove per iteration
THRESHOLD = 0.5       # Stability threshold for Filtered RIF
YEAR_START = 2004     # First target year
YEAR_END = 2010       # Last target year


def run_pipeline(data, G, target_year, model):
    """
    Runs the full perturbation and RIF pipeline for a single target year.
    Returns baseline IF, filtered RIF, and weighted RIF.
    """
    reconstruction_counts = {}
    removal_counts = {}

    # Inner loop - perturbation iterations
    model.eval()
    for i in range(N_ITERATIONS):
        with torch.no_grad():
            perturbed_edge_index, removed_edges = perturb_edges(
                data.edge_index, fraction=FRACTION
            )
            z = model(data.x, perturbed_edge_index)
            scores = compute_reconstruction_scores(z, removed_edges)
        reconstruction_counts, removal_counts = track_reconstruction(
            reconstruction_counts, removal_counts, removed_edges, scores, THRESHOLD
        )

    # Compute stability scores
    stability_scores = compute_stability_scores(reconstruction_counts, removal_counts)

    # Compute metrics
    baseline_if = compute_baseline_if(G, target_year)
    filtered_rif = compute_filtered_rif(G, target_year, stability_scores, THRESHOLD)
    weighted_rif = compute_weighted_rif(G, target_year, stability_scores)

    return baseline_if, filtered_rif, weighted_rif


def save_results(results, path="results/rif_results.csv"):
    """
    Saves yearly results to a CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    # Load data
    data, G = load_pubmed()
    G = assign_synthetic_metadata(G)
    from train import load_trained_model
    model = load_trained_model(num_features=data.num_node_features)

    all_results = []

    # Outer loop - iterate over years
    for year in range(YEAR_START, YEAR_END + 1):
        print(f"\n{'='*50}")
        print(f"Processing year {year}...")
        print(f"{'='*50}")

        baseline_if, filtered_rif, weighted_rif = run_pipeline(data, G, year, model)
        print_rif_comparison(baseline_if, filtered_rif, weighted_rif, year)

        # Store results
        for journal in baseline_if:
            all_results.append({
                "year": year,
                "journal": journal,
                "baseline_if": baseline_if.get(journal, 0),
                "filtered_rif": filtered_rif.get(journal, 0),
                "weighted_rif": weighted_rif.get(journal, 0)
            })

    # Save all results
    save_results(all_results)
    print("\nPipeline complete!")