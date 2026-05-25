"""
openalex_loader.py - Builds citation graph from medical papers using referenced_works.
Fetches papers by medical topic, builds graph from internal references.
"""

import time
import requests
import networkx as nx
import pickle
import os

BASE_URL       = "https://api.openalex.org"
HEADERS        = {"User-Agent": "RIF-Research/1.0 (mailto:roni.brinn@gmail.com)"}
PAPER_TYPE     = "article"
MEDICINE_FIELD = "27"
MAX_PAPERS     = 50000
MIN_PAPERS     = 20


def _get(url, params, retries=5):
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  HTTP 429, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif r.status_code >= 500:
                wait = 2 ** attempt * 2
                print(f"  HTTP {r.status_code}, waiting {wait}s")
                time.sleep(wait)
            else:
                return None
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt * 3
            print(f"  Request error: {e}, waiting {wait}s")
            time.sleep(wait)
    return None


def fetch_medical_papers(year_start, year_end, max_papers=MAX_PAPERS):
    """
    Fetches medical articles published in year_start to year_end.
    Each paper includes: id, year, journal, referenced_works.
    Returns list of paper dicts.
    """
    papers  = []
    cursor  = "*"
    per_page = 200

    filter_str = (
        f"primary_topic.field.id:{MEDICINE_FIELD},"
        f"type:{PAPER_TYPE},"
        f"publication_year:{year_start}-{year_end}"
    )

    print(f"Fetching medical papers ({year_start}-{year_end}, max {max_papers})...")

    while len(papers) < max_papers:
        params = {
            "filter":   filter_str,
            "per-page": per_page,
            "cursor":   cursor,
            "select":   "id,publication_year,primary_location,referenced_works",
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
            source  = (p.get("primary_location") or {}).get("source") or {}
            journal = source.get("display_name", "Unknown")
            refs    = p.get("referenced_works", [])
            if pid and year:
                papers.append({
                    "id":               pid,
                    "year":             year,
                    "journal":          journal,
                    "referenced_works": refs,
                })

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        if len(papers) % 2000 == 0:
            print(f"  Fetched {len(papers)} papers so far...")

        time.sleep(0.15)

    print(f"Total papers fetched: {len(papers)}")
    return papers


def build_citation_graph(target_year, max_papers=MAX_PAPERS):
    """
    Full pipeline:
    1. Fetch medical papers from Y-2, Y-1
    2. Build DiGraph: Paper A -> Paper B if B is in dataset
    3. Print journal stats: papers published and citations received
    Returns G (DiGraph).
    """
    y1 = target_year - 1
    y2 = target_year - 2

    # Step 1: fetch papers
    papers    = fetch_medical_papers(y2, y1, max_papers)
    paper_ids = {p["id"] for p in papers}

    # Step 2: build graph
    print("\nBuilding citation graph...")
    G = nx.DiGraph()

    for p in papers:
        G.add_node(p["id"], year=p["year"], journal=p["journal"])

    for p in papers:
        for ref_id in p["referenced_works"]:
            if ref_id in paper_ids:
                G.add_edge(p["id"], ref_id)

    print(f"Graph complete: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Step 3: print journal stats
    _print_journal_stats(G, target_year)

    return G


def _print_journal_stats(G, target_year):
    """
    Prints per-journal stats:
    - papers published in Y-1, Y-2
    - citations received from papers in Y-1, Y-2
    Only journals with papers >= MIN_PAPERS and citations > 0.
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

        # citations received = in-degree from papers in same window
        for predecessor in G.predecessors(node):
            if G.nodes[predecessor].get("year") in relevant_years:
                journal_citations[journal] = \
                    journal_citations.get(journal, 0) + 1

    print(f"\nJournal stats for year {target_year}:")
    print(f"{'Journal':<50} {'Papers':>8} {'Citations Received':>18}")
    print("-" * 80)

    for journal, papers in sorted(journal_papers.items()):
        citations = journal_citations.get(journal, 0)
        if papers >= MIN_PAPERS and citations > 0:
            print(f"{journal:<50} {papers:>8} {citations:>18}")


def save_graph(G, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def save_journals(journals, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(journals, f)


def load_journals(path):
    with open(path, "rb") as f:
        return pickle.load(f)