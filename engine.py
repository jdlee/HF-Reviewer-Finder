"""
HF Reviewer Finder — matching engine.

Pure-Python (no Streamlit) so it can be unit-/smoke-tested from the CLI.

Builds one text "profile" per Human Factors reviewer from their expertise
overview + publication citations/synopses, embeds them locally with
sentence-transformers, and scores/positions reviewers relative to an
arbitrary query text (e.g. a manuscript abstract).
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "members_enriched.csv")
CACHE_DIR = os.path.join(HERE, "data", ".cache")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Columns whose text describes a reviewer's expertise / output.
_TEXT_COLS = [
    "expertise_overview",
    "recent_1", "recent_1_synopsis",
    "recent_2", "recent_2_synopsis",
    "recent_3", "recent_3_synopsis",
    "seminal_1", "seminal_1_synopsis",
    "seminal_2", "seminal_2_synopsis",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_members(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the enriched roster and build a `profile_text` column per reviewer."""
    df = pd.read_csv(path).fillna("")

    def profile(row) -> str:
        parts = [str(row.get(c, "")) for c in _TEXT_COLS]
        return " \n".join(p for p in parts if p.strip())

    df["profile_text"] = df.apply(profile, axis=1)
    # A short, human-readable expertise blurb for tables/tooltips.
    df["expertise_short"] = df["expertise_overview"].apply(_truncate)
    df["top_recent"] = df["recent_1"].apply(_truncate)
    return df


def _truncate(text: str, n: int = 220) -> str:
    text = str(text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_model(model_name: str = MODEL_NAME):
    """Load (and cache) the sentence-transformers model. Lazy import keeps the
    module importable even before the heavy dependency is installed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _hash_texts(texts: list[str], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_corpus(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed reviewer profiles, caching to disk so it only runs once per corpus."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _hash_texts(texts, model_name)
    cache_file = os.path.join(CACHE_DIR, f"emb_{key}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    model = get_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache_file, emb)
    return emb


def embed_query(text: str, model_name: str = MODEL_NAME) -> np.ndarray:
    model = get_model(model_name)
    v = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)[0]


# --------------------------------------------------------------------------- #
# Similarity + projection
# --------------------------------------------------------------------------- #
def cosine_similarity(corpus_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Both inputs are L2-normalized, so cosine == dot product."""
    return corpus_emb @ query_emb


def project_2d(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray,
    method: str = "pca",
    seed: int = 42,
):
    """Project reviewer embeddings to 2D and place the query in the same space.

    Returns (coords[N,2], query_coord[2]).
    PCA is deterministic and supports out-of-sample `.transform`. UMAP gives
    more topical clustering if `umap-learn` is installed.
    """
    method = (method or "pca").lower()
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15)
            coords = reducer.fit_transform(corpus_emb)
            q = reducer.transform(query_emb.reshape(1, -1))[0]
            return np.asarray(coords), np.asarray(q)
        except Exception:
            # Fall back to PCA if UMAP is unavailable or fails on transform.
            method = "pca"

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(corpus_emb)
    q = pca.transform(query_emb.reshape(1, -1))[0]
    return np.asarray(coords), np.asarray(q)


def rank_reviewers(df: pd.DataFrame, query: str, method: str = "pca") -> pd.DataFrame:
    """Convenience end-to-end: returns df copy with similarity, x, y, query cols."""
    texts = df["profile_text"].tolist()
    corpus = embed_corpus(texts)
    q = embed_query(query)
    sims = cosine_similarity(corpus, q)
    coords, qcoord = project_2d(corpus, q, method=method)

    out = df.copy()
    out["similarity"] = sims
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out.attrs["query_x"] = float(qcoord[0])
    out.attrs["query_y"] = float(qcoord[1])
    return out.sort_values("similarity", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # CLI smoke test.
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else (
        "Drivers' trust in and reliance on advanced driver-assistance automation, "
        "and how takeover requests affect attention during conditionally automated driving."
    )
    df = load_members()
    print(f"Loaded {len(df)} reviewers.")
    ranked = rank_reviewers(df, q)
    print(f"\nQuery: {q}\n")
    print("Top 10 matches:")
    for i, r in ranked.head(10).iterrows():
        print(f"  {i+1:2d}. {r['name']:<28} {r['rank']:<3} sim={r['similarity']:.3f}")
