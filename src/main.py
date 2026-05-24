"""
main.py - Main RIF pipeline using journal-first architecture.
"""

import os
import pickle
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

from openalex_loader import (
    build_citation_graph, save_graph, load_graph,
    save_journals, load_journals
)
from graph_builder import compute_baseline_if
from train_openalex import train as train_model
from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction
from stability import compute_stability_scores, summarize_stability
from rif import compute_filtered_rif, compute_weighted_rif, print_rif_comparison

# Configuration
YEAR_START   = 2018
YEAR_END     = 2022
N_ITERATIONS = 100
FRACTION     = 0.3
THRESHOLD    = 0.5
TOP_JOURNALS = 100
GRAPH_DIR    = "/content/drive/MyDrive/RIF/graphs"
RESULTS_CSV  = "/content/drive/MyDrive/RIF/rif_results.csv"
RESULTS_XLSX = "/content/drive/MyDrive/RIF/rif_results.xlsx"


def build_pyg_data(G):
    """Converts DiGraph to PyG Data with 4 normalized node features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes       = list(G.nodes())
    int_to_node = {i: node for i, node in enumerate(nodes)}
    G_int       = nx.convert_node_labels_to_integers(G)
    num_nodes   = G_int.number_of_nodes()

    features = []
    for n in range(num_nodes):
        orig    = int_to_node[n]
        year    = G.nodes[orig].get("year", 0) or 0
        deg     = G_int.degree(n)
        in_deg  = G_int.in_degree(n)
        out_deg = G_int.out_degree(n)
        features.append([float(year), float(deg),
                         float(in_deg), float(out_deg)])

    x = torch.tensor(features, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    x = x.to(device)

    edges = list(G_int.edges())
    if edges:
        edge_index = torch.tensor(
            edges, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    pyg_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return pyg_data, G_int, int_to_node


def convert_stability_scores(stability_scores_int, int_to_node):
    """Converts integer-keyed scores back to original node id pairs."""
    converted = {}
    for (u_int, v_int), score in stability_scores_int.items():
        u_orig = int_to_node.get(u_int, u_int)
        v_orig = int_to_node.get(v_int, v_int)
        edge   = (min(u_orig, v_orig), max(u_orig, v_orig))
        converted[edge] = score
    return converted


def get_or_build_graph(target_year):
    """Loads graph and journals from cache, or builds fresh."""
    graph_path   = os.path.join(GRAPH_DIR, f"graph_{target_year}.gpickle")
    journal_path = os.path.join(GRAPH_DIR, f"journals_{target_year}.pkl")

    if os.path.exists(graph_path) and os.path.exists(journal_path):
        print(f"Loading cached graph for {target_year}...")
        G        = load_graph(graph_path)
        journals = load_journals(journal_path)
        return G, journals

    print(f"Building graph for {target_year} from OpenAlex...")
    G, journals = build_citation_graph(target_year, top_n=TOP_JOURNALS)
    save_graph(G, graph_path)
    save_journals(journals, journal_path)
    return G, journals


def run_perturbation(pyg_data, model):
    """Runs perturbation loop. Threshold = 80th percentile."""
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
            reconstruction_counts, removal_counts,
            removed_edges, scores, THRESHOLD
        )

        if (i + 1) % 25 == 0:
            print(f"  Perturbation {i+1}/{N_ITERATIONS}")

    stability_scores = compute_stability_scores(
        reconstruction_counts, removal_counts)
    summarize_stability(stability_scores)

    if stability_scores:
        scores_list       = sorted(stability_scores.values())
        idx               = int(len(scores_list) * 0.8)
        dynamic_threshold = scores_list[min(idx, len(scores_list) - 1)]
        print(f"  Dynamic threshold (80th pct): {dynamic_threshold:.4f}")
    else:
        dynamic_threshold = THRESHOLD

    return stability_scores, dynamic_threshold


def save_results(results):
    """Saves results to CSV and Excel."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RIF Results")
        ws = writer.sheets["RIF Results"]
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes    = "A2"

    print(f"Saved -> {RESULTS_CSV}")
    print(f"Saved -> {RESULTS_XLSX}")


if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)
    all_results = []

    for target_year in range(YEAR_START, YEAR_END + 1):
        print(f"\n{'='*60}")
        print(f"  TARGET YEAR: {target_year}")
        print(f"{'='*60}")

        # Step 1+2+3+4: get graph
        G, journals = get_or_build_graph(target_year)
        if G.number_of_edges() == 0:
            print(f"  No edges for {target_year}, skipping.")
            continue

        # Step 5: build PyG + train
        pyg_data, G_int, int_to_node = build_pyg_data(G)
        model = train_model(pyg_data, epochs=100)

        # Step 6: perturbation + stability
        stability_scores_int, dynamic_threshold = run_perturbation(
            pyg_data, model)
        stability_scores = convert_stability_scores(
            stability_scores_int, int_to_node)

        # Step 7: compute IF and RIF
        baseline_if  = compute_baseline_if(G, target_year)
        filtered_rif = compute_filtered_rif(
            G, target_year, stability_scores, dynamic_threshold)
        weighted_rif = compute_weighted_rif(
            G, target_year, stability_scores)

        print_rif_comparison(baseline_if, filtered_rif, weighted_rif,
                             target_year)

        # collect results
        for journal in baseline_if:
            all_results.append({
                "year":                target_year,
                "journal":             journal,
                "baseline_if":         baseline_if.get(journal, 0),
                "filtered_rif":        filtered_rif.get(journal, 0),
                "weighted_rif":        weighted_rif.get(journal, 0),
                "stability_threshold": round(dynamic_threshold, 4),
            })

        save_results(all_results)
        print(f"Results saved after year {target_year}")

    print("\nPipeline complete!")