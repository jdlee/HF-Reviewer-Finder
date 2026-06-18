#!/usr/bin/env python
"""
Drop editorial boilerplate (acknowledgments, errata, corrigenda, publication
notices) from the Human_Factors_*abstracts*.json batches. These items have no
scientific content, so they only add noise to the semantic map and reviewer
matching.

Idempotent: matches by title, so re-running after re-fetching abstracts removes
them again. Run `python precompute.py` afterward to rebuild the embeddings /
projection cache for the smaller corpus.
"""
import glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
GLOB = os.path.join(HERE, "data", "Human_Factors_*abstracts*.json")

# Anchored at the title start; \w* catches plurals and the "Acknowlegments" typo.
BOILERPLATE = re.compile(
    r"^\s*(acknowle\w*|erratum|errata|corrigend\w*|publication notice)\b", re.I
)


def main() -> None:
    files = sorted(glob.glob(GLOB))
    if not files:
        raise SystemExit(f"No abstract files at {GLOB}")
    total_before = total_after = 0
    removed = []
    for path in files:
        recs = json.load(open(path, encoding="utf-8"))
        total_before += len(recs)
        kept, drop = [], []
        for r in recs:
            (drop if BOILERPLATE.match(r.get("title", "")) else kept).append(r)
        total_after += len(kept)
        removed += [r["title"] for r in drop]
        if drop:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(kept, fh, ensure_ascii=False, indent=2)
        print(f"{os.path.basename(path)}: {len(recs)} -> {len(kept)}  (removed {len(drop)})")

    print(f"\nRemoved {len(removed)} boilerplate records; corpus {total_before} -> {total_after}")
    for t in sorted(removed):
        print(f"  - {t[:80]}")


if __name__ == "__main__":
    main()
