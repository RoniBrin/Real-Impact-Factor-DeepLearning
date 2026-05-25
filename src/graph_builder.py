"""
graph_builder.py - Computes Baseline IF from the citation graph.
"""


def compute_baseline_if(G, target_year):
    """
    Computes Baseline IF for every journal in the graph.
    IF(Y) = citations received / papers published in Y-1, Y-2.
    Only journals with papers >= 20 and citations > 0 are included.
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

        for predecessor in G.predecessors(node):
            if G.nodes[predecessor].get("year") in relevant_years:
                journal_citations[journal] = \
                    journal_citations.get(journal, 0) + 1

    print(f"\nBaseline IF for year {target_year}:")
    print(f"{'Journal':<50} {'Papers':>8} {'Citations':>10} {'IF':>8}")
    print("-" * 80)

    baseline_if = {}
    for journal, papers in sorted(journal_papers.items()):
        citations = journal_citations.get(journal, 0)
        if papers >= 20 and citations > 0:
            if_score = round(citations / papers, 4)
            baseline_if[journal] = if_score
            print(f"{journal:<50} {papers:>8} {citations:>10} {if_score:>8}")

    return baseline_if