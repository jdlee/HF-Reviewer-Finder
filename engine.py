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
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "members_enriched.csv")
ABSTRACTS_GLOB = os.path.join(HERE, "data", "Human_Factors_*abstracts*.json")
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


def load_abstracts(glob_pattern: str = ABSTRACTS_GLOB) -> pd.DataFrame:
    """Load all Human Factors article-abstract batches in the data folder
    (title + abstract per row), concatenated and de-duplicated by DOI/title."""
    import glob
    import json

    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No abstract files found matching {glob_pattern}. "
            "Add Human_Factors_*abstracts*.json files to the data/ folder."
        )
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    df = pd.DataFrame(rows).fillna("")
    if df.empty:
        raise ValueError(f"Abstract files matched {glob_pattern} but contained no records.")
    for col in ("doi", "title", "abstract"):
        if col not in df.columns:
            df[col] = ""
    key = df["doi"].where(df["doi"].astype(str).str.strip() != "", df["title"])
    df = df.loc[~key.duplicated()].reset_index(drop=True)
    df["abstract_text"] = (df["title"].astype(str) + ". " + df["abstract"].astype(str)).str.strip()
    df["title_short"] = df["title"].astype(str).apply(_truncate)
    return df


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


def _save_npy(path: str, arr: np.ndarray) -> None:
    """Persist an array, tolerating a read-only / unwritable cache dir."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, arr)
    except OSError:
        pass  # read-only FS → recompute in memory next time, don't crash


def embed_corpus(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed reviewer profiles, caching to disk so it only runs once per corpus."""
    key = _hash_texts(texts, model_name)
    cache_file = os.path.join(CACHE_DIR, f"emb_{key}.npy")
    if os.path.exists(cache_file):
        try:
            return np.load(cache_file)
        except Exception:
            pass  # corrupt cache → recompute
    model = get_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    _save_npy(cache_file, emb)
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


def _make_reducer(method: str, seed: int):
    """Construct a 2D reducer. Falls back to PCA if UMAP is unavailable."""
    method = (method or "pca").lower()
    if method == "umap":
        try:
            import umap  # type: ignore

            return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15)
        except Exception:
            pass
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed)


def fit_project(
    corpus_emb: np.ndarray,
    extra_emb: np.ndarray | None = None,
    method: str = "pca",
    seed: int = 42,
):
    """Co-fit the 2D reducer on reviewers AND papers together so both groups
    share one manifold, then read off each group's coordinates.

    Returns (reducer, corpus_coords[N,2], extra_coords[M,2] or None). The fitted
    reducer is returned so the (per-query) manuscript point can later be placed
    in the *same* space via `transform_query` without re-fitting.
    """
    n = len(corpus_emb)
    fit_data = corpus_emb if extra_emb is None else np.vstack([corpus_emb, extra_emb])
    reducer = _make_reducer(method, seed)
    try:
        embedded = np.asarray(reducer.fit_transform(fit_data))
    except Exception:
        # UMAP can occasionally fail; fall back to PCA on the same data.
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=seed)
        embedded = np.asarray(reducer.fit_transform(fit_data))
    corpus_coords = embedded[:n]
    extra_coords = None if extra_emb is None else embedded[n:]
    return reducer, corpus_coords, extra_coords


def transform_query(reducer, query_emb: np.ndarray) -> np.ndarray:
    """Place a single new point (the manuscript) into an already-fitted space."""
    return np.asarray(reducer.transform(np.asarray(query_emb).reshape(1, -1))[0])


def _projection_key(corpus_emb, extra_emb, method: str) -> str:
    """Cache key over the *content* of the embeddings + the projection library
    versions, so the artifact rebuilds when the data changes (even at the same
    row count) and is never shared across incompatible library versions."""
    import sklearn

    h = hashlib.sha256(method.encode("utf-8"))
    h.update(np.ascontiguousarray(corpus_emb, dtype=np.float32).tobytes())
    if extra_emb is not None:
        h.update(np.ascontiguousarray(extra_emb, dtype=np.float32).tobytes())
    try:
        import umap  # type: ignore

        uv = umap.__version__
    except Exception:
        uv = "none"
    h.update(f"|sklearn={sklearn.__version__}|umap={uv}".encode("utf-8"))
    return h.hexdigest()[:16]


def load_or_build_projection(corpus_emb, extra_emb=None, method="pca", seed=42):
    """Like `fit_project`, but the fitted reducer and coordinates are persisted
    to disk, so the (slow) UMAP co-fit runs only ONCE per dataset+method ever —
    subsequent app starts load it instantly.

    The cache key encodes the embedding content and the sklearn/umap versions, so
    it rebuilds when the data changes and never loads a reducer from an
    incompatible library version. On load it also verifies the reducer type still
    matches the requested method (e.g. won't serve a PCA reducer when UMAP is now
    available), rebuilding if not.

    Returns (reducer, corpus_coords, extra_coords).
    """
    key = _projection_key(corpus_emb, extra_emb, method)
    path = os.path.join(CACHE_DIR, f"proj_{method}_{key}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            if _reducer_matches(d["reducer"], method):
                return d["reducer"], d["corpus_coords"], d["extra_coords"]
        except Exception:
            pass  # corrupt/incompatible cache → rebuild
    reducer, c, e = fit_project(corpus_emb, extra_emb, method=method, seed=seed)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"reducer": reducer, "corpus_coords": c, "extra_coords": e}, fh)
    except OSError:
        pass  # read-only FS → recompute next time, don't crash
    return reducer, c, e


def _reducer_matches(reducer, method: str) -> bool:
    """True if `reducer` is the kind that `method` would build right now. When
    UMAP is unavailable, the PCA fallback is the correct match for 'umap' too."""
    cls = type(reducer).__name__.lower()
    if method.lower() == "umap":
        try:
            import umap  # noqa: F401  (only checking availability)

            return "umap" in cls
        except Exception:
            return "pca" in cls  # umap unavailable → PCA fallback is expected
    return "pca" in cls


# Default manuscript shown (and ranked) on first load — kept here so both the
# app and the precompute script reference the same text.
DEFAULT_QUERY = (
    "Research on trust in generative and agentic AI has focused on system "
    "trustworthiness and user adoption, neglecting the cognitive mechanisms that "
    "govern how people calibrate trust in systems that are probabilistic, opaque, "
    "and socially legible. The central question is not whether AI is trusted or "
    "trustworthy, but whether it is trustable: designed so people can form, "
    "maintain, and revise warranted trust. Calibration matches trust to capability; "
    "warrant grounds it in the actual causes of capability. A computational review "
    "of 2,342 papers and a narrative synthesis identify characteristics "
    "distinguishing generative and agentic AI from traditional automation. We "
    "identify 14 trust-relevant characteristics and organize them by three "
    "challenges: calibration (how much to trust), comprehension (what is being "
    "trusted), and boundary (where AI’s contribution ends and the person’s begins). "
    "These characteristics disrupt trust calibration through four cognitive "
    "mechanisms: attribution, by impoverishing covariation information; construal "
    "level, by widening psychological distance; sensemaking, by stabilizing initial "
    "frames against revision; and anthropomorphism, by activating human-oriented "
    "social cognition. Together, these mechanisms drive the person’s attributional "
    "hierarchy out of alignment with the AI's functional abstraction hierarchy; "
    "misalignment produces miscalibration. We propose cross-hierarchy alignment as "
    "the central design construct for trustable AI, which defines strategies for "
    "functional anthropomorphism in interface design, and for role and relationship "
    "engineering—operationalized through 11 design principles. Panarchy theory "
    "extends the analysis to sociotechnical systems, showing how attributional bleed "
    "may propagate miscalibration to institutional and societal scales. Trustable AI "
    "requires designing the human-AI ecology that shapes trust, not only the model, "
    "interface, or dyadic interaction."
)


def default_query_embedding() -> np.ndarray:
    """Embedding of DEFAULT_QUERY, persisted to disk so the initial ranked load
    needs no model in memory. Computed once (loading the model) by precompute.py."""
    key = hashlib.sha256(DEFAULT_QUERY.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"default_query_{key}.npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass  # corrupt cache → recompute
    v = embed_query(DEFAULT_QUERY)
    _save_npy(path, v)
    return v


def project_2d(
    corpus_emb: np.ndarray,
    query_emb: np.ndarray | None = None,
    extra_emb: np.ndarray | None = None,
    method: str = "pca",
    seed: int = 42,
):
    """Convenience wrapper around `fit_project` + `transform_query`.

    Returns (corpus_coords[N,2], query_coord[2] or None, extra_coords[M,2] or None).
    Reviewers and papers are co-fit on one shared manifold (for both PCA and
    UMAP); the query is transformed into that space.
    """
    reducer, coords, extra = fit_project(corpus_emb, extra_emb, method=method, seed=seed)
    q = None if query_emb is None else transform_query(reducer, query_emb)
    return coords, q, extra


def rank_reviewers(df: pd.DataFrame, query: str, method: str = "pca") -> pd.DataFrame:
    """Convenience end-to-end: returns df copy with similarity, x, y, query cols."""
    texts = df["profile_text"].tolist()
    corpus = embed_corpus(texts)
    q = embed_query(query)
    sims = cosine_similarity(corpus, q)
    coords, qcoord, _ = project_2d(corpus, q, method=method)

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
