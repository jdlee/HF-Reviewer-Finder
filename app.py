"""
HF Reviewer Finder — Streamlit app.

Paste manuscript text (title + abstract) into the box; the app maps the Human
Factors journal's reviewer pool in a 2D semantic space, highlights where your
text lands, and ranks reviewers by topical similarity to it.

Run:  streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import engine

st.set_page_config(page_title="HF Reviewer Finder", layout="wide")

EXAMPLES = {
    "— pick an example —": "",
    "Trust in driving automation": (
        "Drivers' trust in and reliance on advanced driver-assistance systems during "
        "conditionally automated driving. We examine how takeover request timing and "
        "modality affect situation awareness, attention allocation, and reliance "
        "calibration in a high-fidelity driving simulator."
    ),
    "Occupational ergonomics / exoskeletons": (
        "Effects of a passive back-support exoskeleton on lumbar muscle activity, "
        "spinal loading, and perceived exertion during repetitive manual material "
        "handling. Surface EMG and motion capture quantify musculoskeletal disorder "
        "risk reduction in a simulated warehouse task."
    ),
    "Human-AI teaming": (
        "Coordination and shared mental models in human-autonomy teams performing a "
        "collaborative search-and-rescue task. We assess how an AI teammate's "
        "communication transparency shapes team performance, trust, and workload."
    ),
    "Healthcare human factors": (
        "A usability and workload evaluation of an electronic health record alerting "
        "interface in the emergency department, examining alarm fatigue, interruption "
        "recovery, and clinician decision-making under time pressure."
    ),
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return engine.load_members()


@st.cache_resource(show_spinner="Loading embedding model…")
def _model():
    return engine.get_model()


@st.cache_data(show_spinner="Embedding reviewer profiles…")
def corpus_embeddings(texts: tuple[str, ...]):
    return engine.embed_corpus(list(texts))


@st.cache_data(show_spinner=False)
def query_vec(text: str):
    return engine.embed_query(text)


df = load_data()

st.title("🔍 HF Reviewer Finder")
st.caption(
    f"Map and rank the {len(df)} *Human Factors* journal reviewers by how closely their "
    "expertise matches a manuscript. Matching uses local sentence-transformer embeddings "
    "of each reviewer's expertise overview and key publications."
)

# ---- Sidebar controls ---------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    method = st.radio(
        "2D projection", ["pca", "umap"], index=0,
        help="PCA is fast & deterministic. UMAP clusters topics more tightly "
             "(requires umap-learn; falls back to PCA if unavailable).",
    )
    ranks = sorted(df["rank"].unique())
    rank_filter = st.multiselect(
        "Reviewer roles", ranks, default=ranks,
        help="AE=Associate Editor · EB=Editorial Board · R=Reviewer · PR=Past/Provisional Reviewer",
    )
    top_n = st.slider("Show top N in table", 5, len(df), 20)
    st.divider()
    st.caption("Citations are LLM-researched from the web — verify before formal use.")

# ---- Query input --------------------------------------------------------- #
ex = st.selectbox("Load an example", list(EXAMPLES.keys()))
default_text = EXAMPLES.get(ex, "")
query = st.text_area(
    "Manuscript text (title + abstract, or any topic description)",
    value=default_text,
    height=160,
    placeholder="Paste a manuscript title and abstract here…",
)

if not query.strip():
    st.info("Enter manuscript text above (or load an example) to find matching reviewers.")
    st.stop()

# ---- Compute ------------------------------------------------------------- #
_model()  # warm the model cache (shows spinner once)
corpus = corpus_embeddings(tuple(df["profile_text"].tolist()))
q = query_vec(query)

sims = engine.cosine_similarity(corpus, q)
coords, qcoord = engine.project_2d(corpus, q, method=method)

view = df.copy()
view["similarity"] = sims
view["x"] = coords[:, 0]
view["y"] = coords[:, 1]
view = view[view["rank"].isin(rank_filter)].copy()
view = view.sort_values("similarity", ascending=False).reset_index(drop=True)
view.insert(0, "match_rank", view.index + 1)

# ---- 2D semantic map ----------------------------------------------------- #
st.subheader("Semantic map")
st.caption("Each point is a reviewer; the ★ is your manuscript. Closer + brighter = better topical match.")

sim_domain = [float(view["similarity"].min()), float(view["similarity"].max())]
points = (
    alt.Chart(view)
    .mark_circle(opacity=0.85)
    .encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
        size=alt.Size("similarity:Q", scale=alt.Scale(range=[40, 600]), legend=None),
        color=alt.Color(
            "similarity:Q",
            scale=alt.Scale(scheme="viridis", domain=sim_domain),
            legend=alt.Legend(title="Similarity"),
        ),
        tooltip=[
            alt.Tooltip("name:N", title="Reviewer"),
            alt.Tooltip("rank:N", title="Role"),
            alt.Tooltip("similarity:Q", title="Similarity", format=".3f"),
            alt.Tooltip("expertise_short:N", title="Expertise"),
        ],
    )
)
labels = (
    alt.Chart(view.head(12))
    .mark_text(align="left", dx=8, dy=0, fontSize=10, color="#333")
    .encode(x="x:Q", y="y:Q", text="name:N")
)
query_df = pd.DataFrame({"x": [qcoord[0]], "y": [qcoord[1]], "label": ["YOUR TEXT"]})
star = (
    alt.Chart(query_df)
    .mark_point(shape="triangle-up", size=400, color="crimson", filled=True, stroke="black")
    .encode(x="x:Q", y="y:Q", tooltip=alt.value("Your manuscript text"))
)
star_label = (
    alt.Chart(query_df)
    .mark_text(align="center", dy=-16, fontSize=12, fontWeight="bold", color="crimson")
    .encode(x="x:Q", y="y:Q", text="label:N")
)
chart = (points + labels + star + star_label).properties(height=520).interactive()
st.altair_chart(chart, use_container_width=True)

# ---- Ranked table -------------------------------------------------------- #
st.subheader(f"Top {min(top_n, len(view))} matching reviewers")
table = view.head(top_n)[
    ["match_rank", "name", "rank", "similarity", "expertise_short", "top_recent"]
].rename(
    columns={
        "match_rank": "#",
        "name": "Reviewer",
        "rank": "Role",
        "similarity": "Similarity",
        "expertise_short": "Expertise",
        "top_recent": "Most recent publication",
    }
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Similarity": st.column_config.ProgressColumn(
            "Similarity", min_value=0.0, max_value=float(view["similarity"].max()),
            format="%.3f",
        ),
    },
)

# ---- Per-reviewer detail ------------------------------------------------- #
with st.expander("Reviewer detail"):
    who = st.selectbox("Reviewer", view["name"].tolist())
    row = view[view["name"] == who].iloc[0]
    st.markdown(f"### {row['name']}  ·  _{row['rank']}_  ·  similarity **{row['similarity']:.3f}**")
    st.markdown(f"**Expertise.** {row['expertise_overview']}")
    st.markdown("**Recent publications**")
    for i in (1, 2, 3):
        cit, syn = row.get(f"recent_{i}", ""), row.get(f"recent_{i}_synopsis", "")
        if str(cit).strip():
            st.markdown(f"- {cit}\n\n  _{syn}_")
    st.markdown("**Seminal publications**")
    for i in (1, 2):
        cit, syn = row.get(f"seminal_{i}", ""), row.get(f"seminal_{i}_synopsis", "")
        if str(cit).strip():
            st.markdown(f"- {cit}\n\n  _{syn}_")
    if str(row.get("notes", "")).strip():
        st.caption(f"Notes: {row['notes']}")
