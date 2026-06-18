#!/usr/bin/env python
"""
Enrich the Human_Factors abstract batches with two extra metadata fields,
fetched from Crossref (the same source the abstracts came from), keyed by DOI:

  - citation_count : int   -> Crossref "is-referenced-by-count"
  - authors_orcid  : list of {"name": str, "orcid": str|None}, one entry per
                     Crossref author, in order. orcid is the bare ORCID id
                     (e.g. "0000-0002-7973-0641") or null when unavailable.
  - enriched_date  : str   -> ISO date the citation count was captured (counts
                             drift over time, so we stamp provenance).

The existing fields (title, authors, date, doi, abstract) are left untouched,
so engine.py keeps working unchanged.

Resumable & non-destructive: records that already carry "citation_count" are
skipped unless --force is given. Progress is flushed to disk periodically, so an
interrupted run can simply be re-run.

Usage:
    python enrich_metadata.py                # enrich missing records
    python enrich_metadata.py --force        # re-fetch everything
    python enrich_metadata.py --limit 20     # smoke-test on first 20 missing
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ABSTRACTS_GLOB = os.path.join(HERE, "data", "Human_Factors_*abstracts*.json")
MAILTO = "jdleehome@gmail.com"            # Crossref "polite pool"
CROSSREF = "https://api.crossref.org/works/{doi}"
WORKERS = 8
FLUSH_EVERY = 50
ENRICHED_DATE = time.strftime("%Y-%m-%d")


def fetch(doi: str, session: requests.Session, retries: int = 4) -> dict:
    """Return {citation_count, authors_orcid} for a DOI, with backoff retries."""
    url = CROSSREF.format(doi=requests.utils.quote(doi, safe=""))
    for attempt in range(retries):
        try:
            r = session.get(url, params={"mailto": MAILTO}, timeout=30)
            if r.status_code == 404:
                return {"citation_count": None, "authors_orcid": [], "_error": "404 not in Crossref"}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            m = r.json()["message"]
            authors = []
            for a in m.get("author", []):
                name = (a.get("given", "") + " " + a.get("family", "")).strip()
                if not name:
                    name = a.get("name", "").strip()
                orcid = a.get("ORCID")
                if orcid:
                    orcid = orcid.rsplit("/", 1)[-1]  # bare id, drop URL prefix
                authors.append({"name": name, "orcid": orcid})
            return {
                "citation_count": m.get("is-referenced-by-count"),
                "authors_orcid": authors,
            }
        except requests.RequestException as e:
            if attempt == retries - 1:
                return {"citation_count": None, "authors_orcid": [], "_error": str(e)}
            time.sleep(2 ** attempt)
    return {"citation_count": None, "authors_orcid": [], "_error": "exhausted retries"}


def enrich_file(path: str, force: bool, limit: int | None) -> dict:
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)

    todo = [i for i, r in enumerate(records)
            if force or "citation_count" not in r]
    if limit is not None:
        todo = todo[:limit]

    stats = {"file": os.path.basename(path), "total": len(records),
             "fetched": 0, "errors": 0, "with_orcid": 0, "no_abstract_skipped": 0}
    if not todo:
        print(f"  {stats['file']}: nothing to do (all {len(records)} enriched)")
        return stats

    print(f"  {stats['file']}: enriching {len(todo)}/{len(records)} records...")
    done = 0
    with requests.Session() as session:
        session.headers["User-Agent"] = f"HF-Reviewer-Finder/1.0 (mailto:{MAILTO})"
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {pool.submit(fetch, records[i]["doi"], session): i for i in todo}
            for fut in as_completed(futs):
                i = futs[fut]
                res = fut.result()
                if res.get("_error"):
                    stats["errors"] += 1
                records[i]["citation_count"] = res["citation_count"]
                records[i]["authors_orcid"] = res["authors_orcid"]
                records[i]["enriched_date"] = ENRICHED_DATE
                if any(a.get("orcid") for a in res["authors_orcid"]):
                    stats["with_orcid"] += 1
                stats["fetched"] += 1
                done += 1
                if done % FLUSH_EVERY == 0:
                    _save(path, records)
                    print(f"    ...{done}/{len(todo)} (flushed)")
    _save(path, records)
    print(f"    done: {stats['fetched']} fetched, {stats['errors']} errors, "
          f"{stats['with_orcid']} papers with >=1 ORCID")
    return stats


def _save(path: str, records: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if already enriched")
    ap.add_argument("--limit", type=int, default=None, help="cap records per file (smoke test)")
    args = ap.parse_args()

    files = sorted(glob.glob(ABSTRACTS_GLOB))
    if not files:
        sys.exit(f"No abstract files found at {ABSTRACTS_GLOB}")
    print(f"Enriching {len(files)} file(s) from Crossref (citation_count + ORCID)...")
    t0 = time.time()
    agg = {"fetched": 0, "errors": 0, "with_orcid": 0}
    for path in files:
        s = enrich_file(path, args.force, args.limit)
        for k in agg:
            agg[k] += s.get(k, 0)
    print(f"\nTotal: {agg['fetched']} fetched, {agg['errors']} errors, "
          f"{agg['with_orcid']} papers with ORCID  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
