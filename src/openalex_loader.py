"""
openalex_loader.py - Builds citation graphs from OpenAlex using referenced_works.
Architecture:
  1. Fetch top 100 health sciences journals by cited_by_count
  2. Quality filter: papers >= 20
  3. Fetch up to MAX_PAPERS_PER_JOURNAL papers with referenced_works
  4. Build citation graph using referenced_works within dataset only
"""

import time
import requests
import networkx as nx
import pickle
import os

BASE_URL              = "https://api.openalex.org"
HEADERS               = {"User-Agent": "RIF-Research/1.0 (mailto:roni.brinn@gmail.com)"}
PAPER_TYPE            = "article"
MIN_PAPERS            = 20
TOP_JOURNALS          = 100
MAX_PAPERS_PER_JOURNAL = 500
HEALTH_SCIENCES_DOMAIN = "4"  # OpenAlex domain 4 = Health Sciences


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


def fetch_top_journals(top_n=TOP_JOURNALS):
    """
    Fetches top N health sciences journals sorted by cited_by_count.
    Uses domain 4 (Health Sciences) to filter out non-medical journals.
    Returns list of dicts: {id, display_name, cited_by_count, works_count}
    """
    print(f"Fetching top {top_n} health sciences journals by cited_by_count...")

    journals = []
    per_page = 100
    page     = 1

    while len(journals) < top_n:
        params = {
            "filter":   f"type:journal,"
                        f"topics.domain.id:{HEALTH_SCIENCES_DOMAIN}",
            "sort":     "cited_by_count:desc",
            "per-page": per_page,
            "page":     page,
            "select":   "id,display_name,cited_by_count,works_count",
        }
        data = _get(f"{BASE_URL}/sources", params)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for j in results:
            if len(journals) >= top_n:
                break
            journals.append({
                "id":             j.get("id", ""),
                "display_name":   j.get("display_name", "Unknown"),
                "cited_by_count": j.get("cited_by_count", 0),
                "works_count":    j.get("works_count", 0),
            })

        page += 1
        time.sleep(0.2)

    print(f"  Fetched {len(journals)} journals")
    return journals


def get_papers_count(journal_id, year_start, year_end):
    """Returns paper count for a journal in the given time window."""
    params = {
        "filter":   f"primary_location.source.id:{journal_id},"
                    f"publication_year:{year_start}-{year_end},"
                    f"type:{PAPER_TYPE}",
        "per-page": 1,
        "select":   "id",
    }
    data = _get(f"{BASE_URL}/works", params)
    return data.get("meta", {}).get("count", 0) if data else 0


def filter_journals(journals, year_start, year_end):
    """
    Keeps only journals with papers >= MIN_PAPERS in the time window.
    Citations will be computed from the graph after construction.
    """
    print(f"\nApplying quality filter (papers>={MIN_PAPERS})...")

    filtered = []
    for i, j in enumerate(journals):
        papers = get_papers_count(j["id"], year_start, year_end)

        if papers >= MIN_PAPERS:
            filtered.append({**j, "papers_count": papers})
            print(f"  ✓ {j['display_name']:<50} papers={papers:5d}")
        else:
            print(f"  ✗ {j['display_name']:<50} papers={papers:5d} (filtered out)")

        if (i + 1) % 10 == 0:
            print(f"  Checked {i+1}/{len(journals)} journals...")

        time.sleep(0.15)

    print(f"\nJournals after filter: {len(filtered)}/{len(journals)}")
    return filtered


def fetch_papers_for_journal(journal_id, journal_name, year_start, year_end,
                              max_papers=MAX_PAPERS_PER_JOURNAL):
    """
    Fetches up to max_papers papers published by a journal in the time window.
    Each paper includes: id, year, journal, referenced_works.
    """
    papers   = []
    cursor   = "*"
    per_page = 200

    while True:
        if len(papers) >= max_papers:
            break

        params = {
            "filter":   f"primary_location.source.id:{journal_id},"
                        f"publication_year:{year_start}-{year_end},"
                        f"type:{PAPER_TYPE}",
            "per-page": min(per_page, max_papers - len(papers)),
            "cursor":   cursor,
            "select":   "id,publication_year,referenced_works",
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
            pid  = p.get("id", "")
            year = p.get("publication_year")
            refs = p.get("referenced_works", [])
            if pid and year:
                papers.append({
                    "id":               pid,
                    "year":             year,
                    "journal":          journal_name,
                    "referenced_works": refs,
                })

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(0.15)

    return papers


def build_citation_graph(target_year, top_n=TOP_JOURNALS):
    """
    Full pipeline:
    1. Fetch top N health sciences journals
    2. Quality filter: papers >= MIN_PAPERS
    3. Fetch up to MAX_PAPERS_PER_JOURNAL papers per journal with referenced_works
    4. Build DiGraph: Paper A -> Paper B if B is in dataset
    Returns G (DiGraph) and filtered_journals list.
    """
    y1 = target_year - 1
    y2 = target_year - 2

    # Step 1
    journals = fetch_top_journals(top_n)

    # Step 2
    filtered_journals = filter_journals(journals, y2, y1)

    if not filtered_journals:
        print("No journals passed the filter.")
        return nx.DiGraph(), []

    # Step 3
    print(f"\nFetching papers for {len(filtered_journals)} journals "
          f"(years {y2}-{y1}, max {MAX_PAPERS_PER_JOURNAL} per journal)...")

    all_papers = []
    for i, j in enumerate(filtered_journals):
        papers = fetch_papers_for_journal(j["id"], j["display_name"], y2, y1)
        all_papers.extend(papers)
        print(f"  [{i+1}/{len(filtered_journals)}] "
              f"{j['display_name']}: {len(papers)} papers fetched")
        time.sleep(0.2)

    print(f"\nTotal papers fetched: {len(all_papers)}")

    # Step 4
    print("Building citation graph...")
    paper_ids = {p["id"] for p in all_papers}

    G = nx.DiGraph()

    for p in all_papers:
        G.add_node(p["id"], year=p["year"], journal=p["journal"])

    for p in all_papers:
        for ref_id in p["referenced_works"]:
            if ref_id in paper_ids:
                G.add_edge(p["id"], ref_id)

    print(f"Graph complete: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    return G, filtered_journals


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
    """Saves filtered journals list to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(journals, f)
    print(f"Journals saved to {path}")


def load_journals(path):
    with open(path, "rb") as f:
        return pickle.load(f)