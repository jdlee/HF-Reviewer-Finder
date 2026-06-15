# HF Reviewer Finder — Design (2026-06-15)

## Purpose
Given a *Human Factors* manuscript (title + abstract or topic description), help an
editor find the most topically relevant reviewers from the journal's reviewer pool,
shown both as a 2D semantic map and a ranked table.

## Decisions
- **Location:** `~/_Code/HF-Reviewer-Finder/`
- **Match engine:** local sentence-transformers embeddings (`all-MiniLM-L6-v2`),
  cosine similarity. No API key; offline after first model download.
- **Visualization:** Altair 2D semantic map (reviewers as points colored/sized by
  similarity; manuscript shown as a ★) + ranked table. PCA projection by default,
  UMAP optional with automatic PCA fallback.
- **Framework:** Streamlit (matches user preference; Altair over Plotly).

## Architecture
```
engine.py   pure-Python: load CSV → build per-reviewer profile text → embed
            (disk-cached) → cosine similarity → 2D projection (+ query transform)
app.py      Streamlit UI: text box, example loader, sidebar filters, Altair map,
            ranked table, per-reviewer detail expander
data/members_enriched.csv   75 reviewers (from HF_Review skill)
```

## Data flow
1. `load_members()` builds `profile_text` per reviewer = expertise_overview + the
   3 recent + 2 seminal citations and their synopses.
2. `embed_corpus()` embeds all profiles once, cached to `data/.cache/emb_<hash>.npy`
   (hash of texts + model name → re-embeds only when the CSV changes).
3. On query: `embed_query()` → `cosine_similarity()` → `project_2d()` places the
   query in the same PCA/UMAP space via `.transform()`.
4. UI renders the map (layered Altair: points + top-12 labels + query ★) and a
   similarity-ranked table with a top-N slider and role filter.

## Components
- **Sidebar:** projection method (PCA/UMAP), role filter (AE/EB/R/PR), top-N slider.
- **Map:** color = similarity (viridis), size = similarity, tooltip = name/role/score/
  expertise; manuscript ★ in crimson.
- **Table:** rank #, reviewer, role, similarity (progress bar), expertise, recent pub.
- **Detail expander:** full expertise overview + all publications + notes for one reviewer.

## Error handling / edge cases
- Empty query → info message, `st.stop()`.
- UMAP missing or transform failure → silent PCA fallback.
- Missing/blank cells handled via `fillna("")` and truthy checks.

## Testing
- `python engine.py "<query>"` CLI smoke test prints top-10 matches (no browser).
- Manual: launch Streamlit, verify map renders, ★ appears, table ranks sensibly
  (e.g. a trust/automation abstract surfaces Lee, Wickens, de Winter, etc.).

## YAGNI (explicitly out of scope)
- No live Altair brush→table linking (sidebar filters suffice).
- No ScholarOne API integration (separate concern; see HF_Review skill).
- No conflict-of-interest / co-author exclusion logic (future enhancement).
