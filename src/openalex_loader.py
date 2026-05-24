"""
openalex_loader.py - Fetches citation graphs from OpenAlex API with backoff.
"""

import time
import requests
import networkx as nx
import pickle
import os

BASE_URL = "https://api.openalex.org"
HEADERS  = {"User-Agent": "RIF-Research/1.0 (mailto:roni.brinn@gmail.com)"}

MEDICINE_FIELD = "fields/27"
PAPER_TYPE     = "article"


def _get(url, params, retries=5):
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"  HTTP 429, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif r.status_code >= 500:
                wait = 2 ** attempt * 2
                print(f"  HTTP {r.status_code}, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code}, skipping")
                return None
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt * 3
            print(f"  Request error: {e}, waiting {wait}s")
            time.sleep(wait)
    return None


def _fetch_papers(year_start, year_end, max_papers):
    """Fetches up to max_papers medical articles from year_start to year_end."""
    papers = []
    cursor = "*"
    per_page = 200

    filter_str = (
        f"primary_topic.field.id:{MEDICINE_FIELD},"
        f"type:{PAPER_TYPE},"
        f"publication_year:{year_start}-{year_end}"
    )

    print(f"Fetching papers for years {year_start}-{year_end} (max {max_papers})...")

    while len(papers) < max_papers:
        params = {
            "filter":   filter_str,
            "per-page": per_page,
            "cursor":   cursor,
            "select":   "id,publication_year,primary_location,cited_by_count",
        }
        data = _get(f"{BASE_URL}/works", params)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for p in results:
            if len(papers) >= max_papers:
                break
            pid     = p.get("id", "")
            year    = p.get("publication_year")
            source  = p.get("primary_location") or {}
            src     = source.get("source") or {}
            journal = src.get("display_name", "Unknown")
            if pid and year:
                papers.append({"id": pid, "year": year, "journal": journal})

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        print(f"  Fetched {len(papers)} papers so far...")
        time.sleep(0.15)  # polite delay between pages

    print(f"Total papers fetched: {len(papers)}")
    return papers


def _fetch_citers(paper_id, year_start, year_end):
    """Returns list of paper ids that cited paper_id, published in year_start-year_end."""
    filter_str = (
        f"cites:{paper_id},"
        f"primary_topic.field.id:{MEDICINE_FIELD},"
        f"type:{PAPER_TYPE},"
        f"publication_year:{year_start}-{year_end}"
    )
    params = {
        "filter":   filter_str,
        "per-page": 200,
        "select":   "id,publication_year,primary_location",
    }
    data = _get(f"{BASE_URL}/works", params)
    if not data:
        return []

    citers = []
    for p in data.get("results", []):
        pid  = p.get("id", "")
        year = p.get("publication_year")
        source  = p.get("primary_location") or {}
        src     = source.get("source") or {}
        journal = src.get("display_name", "Unknown")
        if pid and year:
            citers.append({"id": pid, "year": year, "journal": journal})
    return citers


def build_citation_graph(target_year, max_papers=3000):
    """
    Builds a directed citation graph for target_year.
    Nodes = papers from (target_year-2, target_year-1).
    Edges = citer -> cited, both from that window.
    """
    y1 = target_year - 1
    y2 = target_year - 2

    papers = _fetch_papers(y2, y1, max_papers)

    G = nx.DiGraph()
    paper_ids = set()

    for p in papers:
        G.add_node(p["id"], year=p["year"], journal=p["journal"])
        paper_ids.add(p["id"])

    print(f"\nFetching citers for {len(papers)} papers...")

    for i, p in enumerate(papers):
        citers = _fetch_citers(p["id"], y2, y1)
        for c in citers:
            # add citer node if not present
            if c["id"] not in paper_ids:
                G.add_node(c["id"], year=c["year"], journal=c["journal"])
                paper_ids.add(c["id"])
            G.add_edge(c["id"], p["id"])  # citer -> cited

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(papers)} papers | "
                  f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

        # polite delay — reduces 429s significantly
        time.sleep(0.2)

    print(f"\nGraph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def save_graph(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G