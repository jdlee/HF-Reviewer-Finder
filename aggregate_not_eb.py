#!/usr/bin/env python
"""
Merge the per-author Not_EB research profiles (data/not_eb_parts/*.json) into the
reviewer table (data/members_enriched.csv).

Idempotent: every run rebuilds ALL rank=="Not_EB" rows from the part files,
leaving the original editorial-board rows untouched. Safe to re-run as more part
files are added (e.g., after the session limit resets and the remaining authors
are researched).
"""
import csv, glob, json, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "members_enriched.csv")
PARTS = os.path.join(HERE, "data", "not_eb_parts", "*.json")

COLS = ["name","rank","found","recent_1","recent_1_synopsis","recent_2",
        "recent_2_synopsis","recent_3","recent_3_synopsis","seminal_1",
        "seminal_1_synopsis","seminal_2","seminal_2_synopsis",
        "expertise_overview","notes"]

# Common given-name nicknames, so a Not_EB author is recognized as the same
# person as an editorial-board reviewer listed under a nickname (e.g. the PR
# "McDonald, Tony" == prolific author "McDonald, Anthony D.").
_NICK = {"tony":"anthony", "ben":"benjamin", "mike":"michael", "rob":"robert",
         "bob":"robert", "chris":"christopher", "dave":"david", "dan":"daniel",
         "jim":"james", "joe":"joseph", "greg":"gregory", "nate":"nathan",
         "kate":"katherine", "liz":"elizabeth", "rick":"richard", "rich":"richard",
         "matt":"matthew", "tom":"thomas", "andy":"andrew", "will":"william",
         "bill":"william", "nick":"nicholas", "sam":"samuel"}


def _person_key(name: str):
    """(surname-final-token, canonical-first-name) used to match the same person
    across name renderings. ASCII-folded, nickname-normalized."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    last, first = (n.split(",", 1) + [""])[:2] if "," in n else \
                  (n.rsplit(" ", 1)[-1], n.rsplit(" ", 1)[0]) if " " in n else (n, "")
    last_toks = re.findall(r"[a-z]+", last)
    first_toks = re.findall(r"[a-z]+", first)
    surname = last_toks[-1] if last_toks else ""
    ft = first_toks[0] if first_toks else ""
    return surname, _NICK.get(ft, ft)


def main():
    # existing rows minus any prior Not_EB rows (so re-runs don't duplicate)
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("rank") != "Not_EB"]
    keep = len(rows)

    # Identity of every real reviewer (AE/EB/PR/R) → drop any Not_EB who is the
    # same person, keeping the editorial-board entry.
    real = {}
    for r in rows:
        k = _person_key(r["name"])
        if k[0] and k[1]:
            real.setdefault(k, r)

    new, dropped = [], []
    for path in sorted(glob.glob(PARTS)):
        if path.endswith("AGENT_SPEC.md"):
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print("SKIP (unparseable)", os.path.basename(path), e)
            continue
        d["rank"] = "Not_EB"
        k = _person_key(d["name"])
        if k[0] and k[1] and k in real:
            m = real[k]
            dropped.append((d["name"], m["rank"], m["name"]))
            continue
        missing = [c for c in COLS if c not in d]
        if missing:
            print("WARN", os.path.basename(path), "missing fields:", missing)
        new.append({c: d.get(c, "") for c in COLS})

    for nm, mrank, mname in dropped:
        print(f"DEDUP: dropped Not_EB '{nm}' — same person as {mrank} '{mname}'")

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
    print(f"Not_EB dropped as dup:    {len(dropped)}")
    print(f"Total reviewer rows:      {len(out)}")


if __name__ == "__main__":
    main()
