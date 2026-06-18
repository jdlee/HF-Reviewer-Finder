#!/usr/bin/env python
"""
Merge the per-author Not_EB research profiles (data/not_eb_parts/*.json) into the
reviewer table (data/members_enriched.csv).

Idempotent: every run rebuilds ALL rank=="Not_EB" rows from the part files,
leaving the original editorial-board rows untouched. Safe to re-run as more part
files are added (e.g., after the session limit resets and the remaining authors
are researched).
"""
import csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "members_enriched.csv")
PARTS = os.path.join(HERE, "data", "not_eb_parts", "*.json")

COLS = ["name","rank","found","recent_1","recent_1_synopsis","recent_2",
        "recent_2_synopsis","recent_3","recent_3_synopsis","seminal_1",
        "seminal_1_synopsis","seminal_2","seminal_2_synopsis",
        "expertise_overview","notes"]


def main():
    # existing rows minus any prior Not_EB rows (so re-runs don't duplicate)
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("rank") != "Not_EB"]
    keep = len(rows)

    new = []
    for path in sorted(glob.glob(PARTS)):
        if path.endswith("AGENT_SPEC.md"):
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print("SKIP (unparseable)", os.path.basename(path), e)
            continue
        d["rank"] = "Not_EB"
        missing = [c for c in COLS if c not in d]
        if missing:
            print("WARN", os.path.basename(path), "missing fields:", missing)
        new.append({c: d.get(c, "") for c in COLS})

    # sort new rows by name for stable output
    new.sort(key=lambda r: r["name"].lower())
    out = rows + new

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in out:
            w.writerow({c: r.get(c, "") for c in COLS})

    print(f"Editorial-board rows kept: {keep}")
    print(f"Not_EB rows written:      {len(new)}")
    print(f"Total reviewer rows:      {len(out)}")


if __name__ == "__main__":
    main()
