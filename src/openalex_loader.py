"""
openalex_loader.py - Fetching real citation data from OpenAlex API.
Two-step fetch: papers from Y-1/Y-2, then their citers from Y-1/Y-2.
"""

import requests
import networkx as nx
import time
import pickle
import os


EMAIL    = "roni.brinn@gmail.com"
BASE_URL = "https://api.openalex.org/works"
FIELD_ID = "fields/27"  # Medicine


def _get(params, retries=3):
    """
    Sends a GET request to OpenAlex with retry logic.
    Returns parsed JSON or None on failure.
    """
    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            print(f"  HTTP {response.status_code}, retry {attempt+1}/{retries}")
        except Exception as e:
            print(f"  Request error: {e}, retry {attempt+1}/{retries}")
        time.sleep(1)
    return None


def fetch_papers_by_years(year1, year2, max_papers=10000, email=EMAIL):
    """
    Fetches up to max_papers articles from the medical field
    published in year1 or year2.
    Returns a list of paper dicts: id, year, journal.
    """
    papers = []
    cursor = "*"

    params = {
        "filter":   f"topics.field.id:{FIELD_ID},"
                    f"publication_year:{year1}|{year2},"
                    f"type:article",
        "select":   "id,publication_year,primary_location",
        "per_page": 200,
        "cursor":   cursor,
        "mailto":   email,
    }

    print(f"Fetching papers for years {year1}-{year2} (max {max_papers})...")

    while len(papers) < max_papers:
        params["cursor"] = cursor
        data = _get(params)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for p in results:
            journal = "Unknown"
            loc = p.get("primary_location")
            if loc and loc.get("source"):
                journal = loc["source"].get("display_name", "Unknown")

            papers.append({
                "id":      p["id"],
                "year":    p.get("publication_year"),
                "journal": journal,
            })

        print(f"  Fetched {len(papers)} papers so far...")

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.1)

    papers = papers[:max_papers]
    print(f"Total papers fetched: {len(papers)}")
    return papers


def fetch_citers(paper_id, year1, year2, email=EMAIL):
    """
    Fetches all articles that cited paper_id, published in year1 or year2.
    Returns a list of citer dicts: id, year, journal.
    """
    citers = []
    cursor = "*"

    params = {
        "filter":   f"cites:{paper_id},"
                    f"publication_year:{year1}|{year2},"
                    f"type:article",
        "select":   "id,publication_year,primary_location",
        "per_page": 200,
        "cursor":   cursor,
        "mailto":   email,
    }

    while True:
        params["cursor"] = cursor
        data = _get(params)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for p in results:
            journal = "Unknown"
            loc = p.get("primary_location")
            if loc and loc.get("source"):
                journal = loc["source"].get("display_name", "Unknown")

            citers.append({
                "id":      p["id"],
                "year":    p.get("publication_year"),
                "journal": journal,
            })

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.05)

    return citers


def build_citation_graph(target_year, max_papers=10000, email=EMAIL):
    """
    Builds a directed citation graph for a given target year.

    Step 1: fetch up to max_papers from Y-1 and Y-2.
    Step 2: for each paper, fetch all citers from Y-1 and Y-2.
    Step 3: build graph — nodes are papers, edges are citation links.

    All nodes carry year and journal attributes.
    Returns a NetworkX DiGraph.
    """
    year1 = target_year - 2
    year2 = target_year - 1

    # Step 1 - fetch base papers
    papers = fetch_papers_by_years(year1, year2, max_papers, email)
    if not papers:
        print("No papers fetched, returning empty graph.")
        return nx.DiGraph()

    G = nx.DiGraph()
    paper_ids = set()

    for p in papers:
        G.add_node(p["id"], year=p["year"], journal=p["journal"])
        paper_ids.add(p["id"])

    # Step 2 - fetch citers for each paper
    print(f"\nFetching citers for {len(papers)} papers...")
    for i, paper in enumerate(papers):
        citers = fetch_citers(paper["id"], year1, year2, email)

        for citer in citers:
            if citer["id"] not in G:
                G.add_node(citer["id"], year=citer["year"], journal=citer["journal"])
            # Edge direction: citer -> cited paper
            G.add_edge(citer["id"], paper["id"])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(papers)} papers "
                  f"| Nodes: {G.number_of_nodes()} "
                  f"| Edges: {G.number_of_edges()}")

    print(f"\nGraph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_graph_stats(G):
    """Prints basic statistics about the graph."""
    from collections import Counter

    years    = [G.nodes[n]["year"]    for n in G.nodes() if G.nodes[n].get("year")]
    journals = [G.nodes[n]["journal"] for n in G.nodes()]

    print(f"\nGraph Statistics:")
    print(f"  Nodes:           {G.number_of_nodes()}")
    print(f"  Edges:           {G.number_of_edges()}")
    if years:
        print(f"  Year range:      {min(years)} - {max(years)}")
    print(f"  Unique journals: {len(set(journals))}")
    print(f"  Top 10 journals by paper count:")
    for journal, count in Counter(journals).most_common(10):
        print(f"    {journal}: {count}")


def save_graph(G, path):
    """Saves a NetworkX graph to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path):
    """Loads a NetworkX graph from disk."""
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    TARGET_YEAR = 2018
    G = build_citation_graph(TARGET_YEAR, max_papers=10000)
    get_graph_stats(G)
    save_graph(G, f"data/graphs/graph_{TARGET_YEAR}.gpickle")