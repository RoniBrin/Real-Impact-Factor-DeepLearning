"""
main.py - Main pipeline for RIF evaluation across multiple years.
"""

import os
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

from openalex_loader import build_citation_graph, save_graph, load_graph
from graph_builder import compute_baseline_if
from train_openalex import train as train_model
from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction
from stability import compute_stability_scores, summarize_stability
from rif import compute_filtered_rif, compute_weighted_rif, print_rif_comparison

# Configuration
YEAR_START   = 2012
YEAR_END     = 2022
MAX_PAPERS   = 10000     # Papers fetched per year from OpenAlex
N_ITERATIONS = 200       # Perturbation iterations per year
FRACTION     = 0.3       # Fraction of edges removed per iteration
THRESHOLD    = 0.5       # Stability threshold for Filtered RIF
GRAPH_DIR    = "data/graphs"
RESULTS_CSV  = "results/rif_results.csv"
RESULTS_XLSX = "results/rif_results.xlsx"


def build_pyg_data(G):
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object.
    Uses node degree as the single node feature.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = nx.convert_node_labels_to_integers(G)
    num_nodes = G.number_of_nodes()

    degrees = torch.tensor(
        [[G.degree(n)] for n in range(num_nodes)],
        dtype=torch.float
    ).to(device)

    edges = list(G.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    return Data(x=degrees, edge_index=edge_index, num_nodes=num_nodes), G


def get_or_build_graph(target_year):
    """
    Loads graph from disk if cached, otherwise fetches from OpenAlex and saves.
    """
    path = os.path.join(GRAPH_DIR, f"graph_{target_year}.gpickle")

    if os.path.exists(path):
        print(f"Loading cached graph for {target_year}...")
        return load_graph(path)

    print(f"Building graph for {target_year} from OpenAlex...")
    G = build_citation_graph(target_year, max_papers=MAX_PAPERS)
    save_graph(G, path)
    return G


def run_perturbation(pyg_data, model):
    """
    Runs the perturbation loop and returns stability scores for all edges.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_data.x          = pyg_data.x.to(device)
    pyg_data.edge_index = pyg_data.edge_index.to(device)

    reconstruction_counts = {}
    removal_counts        = {}

    model.eval()
    for i in range(N_ITERATIONS):
        with torch.no_grad():
            perturbed_edge_index, removed_edges = perturb_edges(
                pyg_data.edge_index, fraction=FRACTION
            )
            z      = model(pyg_data.x, perturbed_edge_index)
            scores = compute_reconstruction_scores(z, removed_edges)

        reconstruction_counts, removal_counts = track_reconstruction(
            reconstruction_counts, removal_counts, removed_edges, scores, THRESHOLD
        )

        if (i + 1) % 50 == 0:
            print(f"  Perturbation {i+1}/{N_ITERATIONS}")

    stability_scores = compute_stability_scores(reconstruction_counts, removal_counts)
    summarize_stability(stability_scores)
    return stability_scores


def save_results(results):
    """Saves results list to CSV and Excel."""
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)

    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved CSV -> {RESULTS_CSV}")

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RIF Results")
        ws = writer.sheets["RIF Results"]
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes    = "A2"
    print(f"Saved Excel -> {RESULTS_XLSX}")


if __name__ == "__main__":
    all_results = []

    for target_year in range(YEAR_START, YEAR_END + 1):
        print(f"\n{'='*60}")
        print(f"  TARGET YEAR: {target_year}")
        print(f"{'='*60}")

        # Step 1 - load or build graph
        G = get_or_build_graph(target_year)

        if G.number_of_edges() == 0:
            print(f"  No edges for {target_year}, skipping.")
            continue

        # Step 2 - convert to PyG and train model
        pyg_data, G_int = build_pyg_data(G)
        model = train_model(pyg_data, epochs=50)

        # Step 3 - perturbation and stability scores
        stability_scores = run_perturbation(pyg_data, model)

        # Step 4 - compute IF and RIF on original graph
        baseline_if  = compute_baseline_if(G, target_year)
        filtered_rif = compute_filtered_rif(G, target_year, stability_scores, THRESHOLD)
        weighted_rif = compute_weighted_rif(G, target_year, stability_scores)

        print_rif_comparison(baseline_if, filtered_rif, weighted_rif, target_year)

        # Step 5 - store results
        for journal in baseline_if:
            all_results.append({
                "year":         target_year,
                "journal":      journal,
                "baseline_if":  baseline_if.get(journal, 0),
                "filtered_rif": filtered_rif.get(journal, 0),
                "weighted_rif": weighted_rif.get(journal, 0),
            })

    save_results(all_results)
    print("\nPipeline complete!")