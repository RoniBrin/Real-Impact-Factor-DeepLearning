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
YEAR_START   = 2018
YEAR_END     = 2022
MAX_PAPERS   = 3000
N_ITERATIONS = 100
FRACTION     = 0.3
THRESHOLD    = 0.5
GRAPH_DIR    = "/content/drive/MyDrive/RIF/graphs"
RESULTS_CSV  = "/content/drive/MyDrive/RIF/rif_results.csv"
RESULTS_XLSX = "/content/drive/MyDrive/RIF/rif_results.xlsx"


def build_pyg_data(G):
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object.
    Returns pyg_data, G_int, and mappings between integer and original node ids.
    Node attributes (year, journal) are preserved in G; G_int is used only for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build explicit mapping before conversion
    nodes = list(G.nodes())
    int_to_node = {i: node for i, node in enumerate(nodes)}

    G_int = nx.convert_node_labels_to_integers(G)
    num_nodes = G_int.number_of_nodes()

    degrees = torch.tensor(
        [[G_int.degree(n)] for n in range(num_nodes)],
        dtype=torch.float
    ).to(device)

    edges = list(G_int.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    pyg_data = Data(x=degrees, edge_index=edge_index, num_nodes=num_nodes)
    return pyg_data, G_int, int_to_node


def convert_stability_scores(stability_scores_int, int_to_node):
    """
    Converts stability scores keyed by integer node pairs
    back to original node id pairs, to match edges in G.
    """
    converted = {}
    for (u_int, v_int), score in stability_scores_int.items():
        u_orig = int_to_node.get(u_int, u_int)
        v_orig = int_to_node.get(v_int, v_int)
        edge = (min(u_orig, v_orig), max(u_orig, v_orig))
        converted[edge] = score
    return converted


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
    Runs the perturbation loop and returns integer-keyed stability scores
    and a dynamic threshold based on the median stability score.
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

    # Dynamic threshold: median stability score ensures ~50% of edges are filtered
    if stability_scores:
        scores_list = sorted(stability_scores.values())
        dynamic_threshold = scores_list[len(scores_list) // 2]
        print(f"  Dynamic threshold (median): {dynamic_threshold:.4f}")
    else:
        dynamic_threshold = THRESHOLD
        print(f"  No stability scores, using default threshold: {THRESHOLD}")

    return stability_scores, dynamic_threshold


def save_results(results):
    """Saves results list to CSV and Excel after every year."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
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
    os.makedirs(GRAPH_DIR, exist_ok=True)
    all_results = []

    for target_year in range(YEAR_START, YEAR_END + 1):
        print(f"\n{'='*60}")
        print(f"  TARGET YEAR: {target_year}")
        print(f"{'='*60}")

        # Step 1 - load or build graph with original node ids and attributes
        G = get_or_build_graph(target_year)

        if G.number_of_edges() == 0:
            print(f"  No edges for {target_year}, skipping.")
            continue

        # Step 2 - convert to PyG for training, keep int->original mapping
        pyg_data, G_int, int_to_node = build_pyg_data(G)
        model = train_model(pyg_data, epochs=50)

        # Step 3 - perturbation produces integer-keyed stability scores
        stability_scores_int, dynamic_threshold = run_perturbation(pyg_data, model)

        # Step 4 - convert stability scores back to original node id pairs
        stability_scores = convert_stability_scores(stability_scores_int, int_to_node)

        # Step 5 - compute IF and RIF on G (original) so year/journal attributes work
        baseline_if  = compute_baseline_if(G, target_year)
        filtered_rif = compute_filtered_rif(G, target_year, stability_scores, dynamic_threshold)
        weighted_rif = compute_weighted_rif(G, target_year, stability_scores)

        print_rif_comparison(baseline_if, filtered_rif, weighted_rif, target_year)

        # Step 6 - store results
        for journal in baseline_if:
            all_results.append({
                "year":               target_year,
                "journal":            journal,
                "baseline_if":        baseline_if.get(journal, 0),
                "filtered_rif":       filtered_rif.get(journal, 0),
                "weighted_rif":       weighted_rif.get(journal, 0),
                "stability_threshold": round(dynamic_threshold, 4),
            })

        # Step 7 - save after every year in case of disconnection
        save_results(all_results)
        print(f"Results saved after year {target_year}")

    print("\nPipeline complete!")