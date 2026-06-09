"""
main_node2vec.py - RIF pipeline using Node2Vec embeddings.
"""

import os
import json
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

from openalex_loader import build_citation_graph, save_graph, load_graph
from graph_builder import compute_baseline_if
from train_node2vec import train_node2vec
from perturbation import perturb_edges, compute_reconstruction_scores, track_reconstruction
from stability import compute_stability_scores, summarize_stability
from rif import compute_filtered_rif, compute_weighted_rif, print_rif_comparison

# Configuration
YEAR_START   = 2018
YEAR_END     = 2022
N_ITERATIONS = 100
FRACTION     = 0.3
THRESHOLD    = 0.7
GRAPH_DIR    = "/content/drive/MyDrive/RIF/graphs"
RESULTS_CSV  = "/content/drive/MyDrive/RIF/rif_results_node2vec.csv"
RESULTS_XLSX = "/content/drive/MyDrive/RIF/rif_results_node2vec.xlsx"


def build_pyg_data(G):
    """
    Converts DiGraph to PyG Data.
    Node2Vec only needs edge_index — no node features required.
    Returns pyg_data, int_to_node mapping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes       = list(G.nodes())
    int_to_node = {i: node for i, node in enumerate(nodes)}
    G_int       = nx.convert_node_labels_to_integers(G)
    num_nodes   = G_int.number_of_nodes()

    edges = list(G_int.edges())
    if edges:
        edge_index = torch.tensor(
            edges, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    pyg_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return pyg_data, int_to_node


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
    """Loads graph from cache if available, otherwise builds from OpenAlex."""
    graph_path = os.path.join(GRAPH_DIR, f"graph_{target_year}.gpickle")

    if os.path.exists(graph_path):
        print(f"Loading cached graph for {target_year}...")
        return load_graph(graph_path)

    print(f"Building graph for {target_year} from OpenAlex...")
    G = build_citation_graph(target_year)
    save_graph(G, graph_path)
    return G


def run_perturbation_node2vec(pyg_data, embeddings, target_year):
    """
    Runs perturbation loop using Node2Vec embeddings.
    Uses fixed threshold = THRESHOLD.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = pyg_data.edge_index.to(device)
    z          = embeddings.to(device)

    reconstruction_counts = {}
    removal_counts        = {}

    for i in range(N_ITERATIONS):
        with torch.no_grad():
            perturbed_edge_index, removed_edges = perturb_edges(
                edge_index, fraction=FRACTION
            )
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

    scores_path = os.path.join(
        GRAPH_DIR, f"stability_scores_node2vec_{target_year}.json")
    with open(scores_path, "w") as f:
        json.dump(list(stability_scores.values()), f)

    print(f"  Threshold: {THRESHOLD}")
    return stability_scores, THRESHOLD


def save_results(results):
    """Saves results to CSV and Excel."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RIF Results Node2Vec")
        ws = writer.sheets["RIF Results Node2Vec"]
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes    = "A2"

    print(f"Saved -> {RESULTS_CSV}")
    print(f"Saved -> {RESULTS_XLSX}")


if __name__ == "__main__":
    os.makedirs(GRAPH_DIR, exist_ok=True)
    all_results = []

    for target_year in range(YEAR_START, YEAR_END + 1):
        print(f"\n{'='*60}")
        print(f"  TARGET YEAR: {target_year} — Node2Vec")
        print(f"{'='*60}")

        # Step 1: load or build graph
        G = get_or_build_graph(target_year)
        if G.number_of_edges() == 0:
            print(f"  No edges for {target_year}, skipping.")
            continue

        # Step 2: build PyG data
        pyg_data, int_to_node = build_pyg_data(G)

        # Step 3: train Node2Vec
        embeddings = train_node2vec(
            edge_index=pyg_data.edge_index,
            num_nodes=pyg_data.num_nodes,
            embedding_dim=64,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            epochs=50,
            lr=0.01,
            batch_size=128,
        )

        # Step 4: perturbation + stability
        stability_scores_int, threshold = run_perturbation_node2vec(
            pyg_data, embeddings, target_year)
        stability_scores = convert_stability_scores(
            stability_scores_int, int_to_node)

        # Step 5: compute IF and RIF
        baseline_if  = compute_baseline_if(G, target_year)
        filtered_rif = compute_filtered_rif(
            G, target_year, stability_scores, threshold)
        weighted_rif = compute_weighted_rif(
            G, target_year, stability_scores)

        print_rif_comparison(baseline_if, filtered_rif, weighted_rif,
                             target_year)

        # Step 6: collect results
        for journal in baseline_if:
            all_results.append({
                "year":                target_year,
                "journal":             journal,
                "baseline_if":         baseline_if.get(journal, 0),
                "filtered_rif":        filtered_rif.get(journal, 0),
                "weighted_rif":        weighted_rif.get(journal, 0),
                "stability_threshold": round(threshold, 4),
                "model":               "node2vec",
            })

        # Step 7: save after every year
        save_results(all_results)
        print(f"Results saved after year {target_year}")

    print("\nNode2Vec pipeline complete!")