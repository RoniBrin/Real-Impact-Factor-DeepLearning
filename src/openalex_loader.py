"""
openalex_loader.py - Fetching real citation data from OpenAlex API.
"""

import email

import requests
import networkx as nx
import time


def fetch_papers_by_topic(topic="diabetes", max_papers=5000, email="your@email.com"):
    """
    Fetches papers from OpenAlex API by topic.
    Returns a list of papers with metadata.
    """
    papers = []
    cursor = "*"
    base_url = "https://api.openalex.org/works"

    headers = {}

    params = {
        "filter": "default.search:diabetes,has_references:true",
        "select": "id,title,publication_year,primary_location,referenced_works",
        "per_page": 200,
        "cursor": cursor,
        "mailto": email
    }

    print(f"Fetching papers about '{topic}' from OpenAlex...")

    while len(papers) < max_papers:
        params["cursor"] = cursor
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        papers.extend(results)
        print(f"  Fetched {len(papers)} papers so far...")

        # Get next cursor
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.1)  # Rate limiting

    print(f"Total papers fetched: {len(papers)}")
    return papers[:max_papers]


def build_citation_graph(papers):
    """
    Builds a directed citation graph from OpenAlex papers.
    Nodes = papers, edges = citation links.
    """
    G = nx.DiGraph()

    # Map OpenAlex IDs to node indices
    paper_ids = {paper["id"]: i for i, paper in enumerate(papers)}

    # Add nodes with metadata
    for paper in papers:
        node_id = paper_ids[paper["id"]]
        year = paper.get("publication_year")
        journal = "Unknown"

        location = paper.get("primary_location")
        if location and location.get("source"):
            journal = location["source"].get("display_name", "Unknown")

        G.add_node(node_id, year=year, journal=journal, openalex_id=paper["id"])

    # Add edges (citations)
    for paper in papers:
        src = paper_ids[paper["id"]]
        for ref in paper.get("referenced_works", []):
            if ref in paper_ids:
                dst = paper_ids[ref]
                G.add_edge(src, dst)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_graph_stats(G):
    """
    Prints basic statistics about the graph.
    """
    years = [G.nodes[n]["year"] for n in G.nodes() if G.nodes[n]["year"]]
    journals = [G.nodes[n]["journal"] for n in G.nodes()]

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Year range: {min(years)} - {max(years)}")
    print(f"  Unique journals: {len(set(journals))}")
    print(f"  Top 5 journals:")
    from collections import Counter
    journal_counts = Counter(journals).most_common(5)
    for journal, count in journal_counts:
        print(f"    {journal}: {count} papers")

def save_graph(G, path="data/openalex_graph.gpickle"):
    """Saves the graph to disk."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path="data/openalex_graph.gpickle"):
    """Loads the graph from disk."""
    import pickle
    with open(path, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G



if __name__ == "__main__":
    papers = fetch_papers_by_topic(
        topic="diabetes",
        max_papers=5000,
        email="roni.brinn@gmail.com"
    )
    G = build_citation_graph(papers)
    get_graph_stats(G)
    save_graph(G)