"""
rif.py - Computing the Real Impact Factor (RIF) using edge stability scores.
"""


def compute_filtered_rif(G, target_year, stability_scores, threshold=0.5):
    """
    Computes Filtered RIF for each journal in target year Y.
    Excludes citations with stability score below the threshold.
    Counts incoming edges (predecessors) from Y-1 or Y-2 only.
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

        for predecessor in G.predecessors(node):
            if G.nodes[predecessor].get('year') in (target_year - 1, target_year - 2):
                edge = (min(node, predecessor), max(node, predecessor))
                score = stability_scores.get(edge, 0.0)
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
    Counts incoming edges (predecessors) from Y-1 or Y-2 only.
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

        for predecessor in G.predecessors(node):
            if G.nodes[predecessor].get('year') in (target_year - 1, target_year - 2):
                edge = (min(node, predecessor), max(node, predecessor))
                score = stability_scores.get(edge, 0.0)
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