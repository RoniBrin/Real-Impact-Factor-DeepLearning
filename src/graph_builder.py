"""
graph_builder.py - Building temporal subgraphs and computing Baseline IF.
"""

import random
import networkx as nx


def assign_synthetic_metadata(G, journals=None, year_range=(2000, 2020)):
    """
    Assigns synthetic year and journal metadata to each node.
    Used for development and testing before OpenAlex integration.
    """
    if journals is None:
        journals = ["Nature", "Science", "PubMed Central", "NEJM", "Lancet"]

    for node in G.nodes():
        G.nodes[node]['year'] = random.randint(year_range[0], year_range[1])
        G.nodes[node]['journal'] = random.choice(journals)

    print(f"Assigned synthetic metadata to {G.number_of_nodes()} nodes")
    print(f"Year range: {year_range[0]} - {year_range[1]}")
    print(f"Journals: {journals}")
    return G


def extract_time_window(G, target_year):
    """
    Extracts a subgraph for a given target year Y.
    Includes papers published in Y-1 and Y-2,
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
    Computes the Baseline Impact Factor for each journal in target year Y.
    IF = total citations received / number of papers published.
    """
    # Papers published in Y-1 and Y-2
    relevant_nodes = set(
        node for node in G.nodes()
        if G.nodes[node].get('year') in (target_year - 1, target_year - 2)
    )

    # Count citations and papers per journal
    journal_citations = {}
    journal_papers = {}

    for node in relevant_nodes:
        journal = G.nodes[node].get('journal', 'Unknown')
        journal_papers[journal] = journal_papers.get(journal, 0) + 1
        citations = sum(1 for neighbor in G.neighbors(node)
                        if neighbor not in relevant_nodes)
        journal_citations[journal] = journal_citations.get(journal, 0) + citations

    # Print detailed breakdown
    print(f"\nDetailed IF breakdown for year {target_year}:")
    print(f"{'Journal':<20} {'Papers':>8} {'Citations':>10} {'IF':>8}")
    print("-" * 50)
    for journal in journal_papers:
        papers = journal_papers[journal]
        citations = journal_citations.get(journal, 0)
        if_score = round(citations / papers, 4) if papers > 0 else 0.0
        print(f"{journal:<20} {papers:>8} {citations:>10} {if_score:>8}")

    # Compute IF
    baseline_if = {}
    for journal in journal_papers:
        papers = journal_papers[journal]
        citations = journal_citations.get(journal, 0)
        baseline_if[journal] = round(citations / papers, 4) if papers > 0 else 0.0

    return baseline_if


if __name__ == "__main__":
    from data_loader import load_pubmed

    data, G = load_pubmed()
    G = assign_synthetic_metadata(G)

    target_year = 2010
    subgraph = extract_time_window(G, target_year)
    baseline_if = compute_baseline_if(G, target_year)

    print(f"\nBaseline IF for year {target_year}:")
    for journal, if_score in sorted(baseline_if.items(), key=lambda x: x[1], reverse=True):
        print(f"  {journal}: {if_score}")