"""
graph_builder.py - Building temporal subgraphs and computing Baseline IF.
"""

import networkx as nx


def extract_time_window(G, target_year):
    """
    Extracts a subgraph for a given target year Y.
    Includes all nodes published in Y-1 and Y-2,
    and citation edges between them.
    """
    relevant_nodes = [
        node for node in G.nodes()
        if G.nodes[node].get('year') in (target_year - 1, target_year - 2)
    ]
    subgraph = G.subgraph(relevant_nodes).copy()
    print(f"Year {target_year}: {subgraph.number_of_nodes()} nodes, "
          f"{subgraph.number_of_edges()} edges")
    return subgraph


def compute_baseline_if(G, target_year):
    """
    Computes the Baseline Impact Factor for every journal in the graph.
    IF(Y) = citations received by journal papers from Y-1 and Y-2,
            where citing papers are also from Y-1 or Y-2.
    Only journals with at least 5 papers and at least 1 citation are included.
    """
    relevant_years = (target_year - 1, target_year - 2)

    journal_papers    = {}
    journal_citations = {}

    for node in G.nodes():
        year    = G.nodes[node].get("year")
        journal = G.nodes[node].get("journal", "Unknown")

        if year not in relevant_years:
            continue

        journal_papers[journal] = journal_papers.get(journal, 0) + 1

        # Count incoming citations from papers in Y-1 or Y-2
        for predecessor in G.predecessors(node):
            if G.nodes[predecessor].get("year") in relevant_years:
                journal_citations[journal] = journal_citations.get(journal, 0) + 1

    print(f"\nBaseline IF breakdown for year {target_year}:")
    print(f"{'Journal':<45} {'Papers':>8} {'Citations':>10} {'IF':>8}")
    print("-" * 75)

    baseline_if = {}
    for journal, papers in sorted(journal_papers.items()):
        citations = journal_citations.get(journal, 0)
        if papers >= 5 and citations > 0:
            if_score = round(citations / papers, 4)
            baseline_if[journal] = if_score
            print(f"{journal:<45} {papers:>8} {citations:>10} {if_score:>8}")

    return baseline_if


if __name__ == "__main__":
    from openalex_loader import build_citation_graph

    TARGET_YEAR = 2018
    G = build_citation_graph(TARGET_YEAR, max_papers=3000)
    baseline_if = compute_baseline_if(G, TARGET_YEAR)

    print(f"\nTop journals by IF for {TARGET_YEAR}:")
    for journal, score in sorted(baseline_if.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {journal}: {score}")