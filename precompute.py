#!/usr/bin/env python
"""
Pre-calculate everything the app needs so it loads instantly:
  - reviewer + abstract embeddings   (data/.cache/emb_*.npy)
  - PCA and UMAP co-fit projections  (data/.cache/proj_*.pkl)   <- the slow UMAP fit
  - the default-manuscript embedding  (data/.cache/default_query_*.npy)

Run once after changing the reviewer CSV or adding abstract batches:

    python precompute.py

After this, the app starts with no model load and no UMAP fitting; it just
reads these artifacts from disk.
"""
import time

import engine


def main() -> None:
    t0 = time.time()
    rev = engine.load_members()
    remb = engine.embed_corpus(rev["profile_text"].tolist())
    print(f"reviewers embedded: {remb.shape}")

    ab = engine.load_abstracts()
    aemb = engine.embed_corpus(ab["abstract_text"].tolist())
    print(f"abstracts embedded: {aemb.shape}")

    for method in ("pca", "umap"):
        t = time.time()
        engine.load_or_build_projection_coords(remb, aemb, method=method)
        print(f"projection coords '{method}' built/cached in {time.time() - t:.1f}s")

    engine.default_query_embedding()
    print("default-query embedding cached")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
