"""
HF Reviewer Finder — Streamlit app.

Paste manuscript text (title + abstract) into the box; the app maps the Human
Factors journal's reviewer pool in a 2D semantic space, highlights where your
text lands, and ranks reviewers by topical similarity to it.

Run:  streamlit run app.py
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import engine

st.set_page_config(
    page_title="HF Reviewer Finder",
    page_icon="🔍",
    layout="wide",
)

# ---- Apple-HIG-inspired styling: clarity, deference, depth ---------------- #
st.markdown(
    """
    <style>
    :root { --hig-accent:#0071e3; --hig-radius:14px; }
    html, body, [class*="css"], .stMarkdown, textarea, input, select, button {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }
    h1 { font-weight: 700; letter-spacing: -0.022em; }
    h2, h3 { font-weight: 600; letter-spacing: -0.012em; }
    /* Primary submit button — Apple capsule style */
    .stFormSubmitButton button {
        background: var(--hig-accent); color:#fff; border:none;
        border-radius: 980px; padding: 0.6rem 1.2rem; font-weight: 600;
        transition: background .15s ease, transform .05s ease;
    }
    .stFormSubmitButton button:hover { background:#0077ed; color:#fff; }
    .stFormSubmitButton button:active { transform: scale(0.98); }
    /* Soft rounded surfaces */
    .stTextArea textarea { border-radius: var(--hig-radius); }
    [data-testid="stDataFrame"],
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--hig-radius); }
    [data-testid="stSidebar"] { border-right: 1px solid #e8e8ed; }
    /* Quieter captions */
    [data-testid="stCaptionContainer"] { color:#6e6e73; }
    /* Cleaner app chrome (deference) */
    [data-testid="stDecoration"] { display:none; }
    #MainMenu, footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    # The default manuscript's embedding is precomputed on disk → no model load.
    if text == engine.DEFAULT_QUERY:
        return engine.default_query_embedding()
    return engine.embed_query(text)


@st.cache_data(show_spinner=False)
def load_abstracts() -> pd.DataFrame:
    return engine.load_abstracts()


@st.cache_data(show_spinner="Embedding background article abstracts…")
def abstract_embeddings(texts: tuple[str, ...]):
    return engine.embed_corpus(list(texts))


@st.cache_data(show_spinner="Building 2D projection (co-fitting reviewers + papers)…")
def projection_coords(method: str, sig: tuple, _corpus, _extra):
    """Co-fit reviewers + papers coordinates once per method; cached across reruns
    and persisted to disk as portable .npz. `sig` (hashed) invalidates on data
    change; `_corpus`/`_extra` are large arrays passed unhashed. Returns only
    coordinates (no reducer), so the runtime never needs numba/UMAP at request
    time — the query is placed with engine.place_query."""
    return engine.load_or_build_projection_coords(_corpus, _extra, method=method)


try:
    df = load_data()
    abstracts = load_abstracts()
    abs_emb = abstract_embeddings(tuple(abstracts["abstract_text"].tolist()))
except Exception as exc:  # missing/malformed data files → clear message, not a stack trace
    st.error(
        "Could not load data from the `data/` folder "
        f"({type(exc).__name__}: {exc}). Ensure `members_enriched.csv` and the "
        "`Human_Factors_*abstracts*.json` files are present."
    )
    st.stop()

# Seed the default manuscript (defined in engine, shared with precompute.py).
for _k in ("query", "query_input"):
    if _k not in st.session_state:
        st.session_state[_k] = engine.DEFAULT_QUERY

# ---- Sidebar: manuscript description (the query input) ------------------- #
with st.sidebar:
    st.header("Manuscript description")
    with st.form("query_form", border=False):
        st.text_area(
            "Manuscript description",
            height=300,
            key="query_input",
            label_visibility="collapsed",
            placeholder="Enter the manuscript's title, abstract, or any description of its content…",
        )
        submitted = st.form_submit_button(
            "Find reviewers", type="primary", use_container_width=True
        )
    if submitted:
        st.session_state["query"] = st.session_state.get("query_input", "")
    st.divider()
    method = st.radio(
        "2D projection", ["pca", "umap"], index=0,
        format_func=lambda m: {
            "pca": "pca (preserve macro structure)",
            "umap": "umap (preserve local structure)",
        }[m],
        help="PCA is fast & deterministic. UMAP clusters topics more tightly "
             "(requires umap-learn; falls back to PCA if unavailable).",
    )
    st.caption("Citations are LLM-researched from the web — verify before formal use.")

query = st.session_state.get("query", "")
has_query = bool(query.strip())

st.title(f"🔍 Reviewer Finder: {len(df)} reviewers mapped")
st.caption(
    "Local sentence-transformer embeddings of each *Human Factors* reviewer's expertise and "
    "key publications, ranked against your manuscript."
)
st.caption(
    "Each point is a reviewer, positioned by expertise; open circles are prolific "
    "authors not on the editorial board. "
    + ("The red circle is your manuscript; closer + brighter = better topical match."
       if has_query else
       "Enter a manuscript description in the sidebar to rank reviewers by relevance.")
)

# ---- Compute (map always; ranking only when a query is present) ---------- #
corpus = corpus_embeddings(tuple(df["profile_text"].tolist()))
# Reviewers + papers are co-fit on one shared manifold (coords cached per method).
coords, abs_coords = projection_coords(method, (len(df), len(abstracts)), corpus, abs_emb)
if has_query:
    q = query_vec(query)  # default query is precomputed; others load the model lazily
    sims = engine.cosine_similarity(corpus, q)
    # Numba-free placement of the manuscript marker (no UMAP at request time).
    fit_emb = np.vstack([corpus, abs_emb])
    fit_coords = np.vstack([coords, abs_coords])
    qcoord = engine.place_query(q, fit_emb, fit_coords)
else:
    qcoord = None

abs_df = pd.DataFrame({
    "x": abs_coords[:, 0],
    "y": abs_coords[:, 1],
    "title": abstracts["title_short"].values,
})

view = df.copy()
view["x"] = coords[:, 0]
view["y"] = coords[:, 1]
if has_query:
    view["similarity"] = sims
    view = view.sort_values("similarity", ascending=False).reset_index(drop=True)
else:
    view = view.sort_values("name").reset_index(drop=True)
view.insert(0, "match_rank", view.index + 1)

# ---- 2D semantic map (visible on load) ----------------------------------- #
enc = dict(
    x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False)),
    y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
    tooltip=[
        alt.Tooltip("name:N", title="Reviewer"),
        alt.Tooltip("rank:N", title="Role"),
        *([alt.Tooltip("similarity:Q", title="Similarity", format=".2f")] if has_query else []),
        alt.Tooltip("expertise_short:N", title="Expertise"),
    ],
)
if has_query:
    sim_domain = [float(view["similarity"].min()), float(view["similarity"].max())]


def _size_enc(size_range: list[int], flat: int):
    """Size by similarity when ranking, else a flat size. Non-EB layers pass a
    smaller range/flat value so they read as secondary.

    The shared `domain` (= sim_domain) is essential: it ties both layers to the
    *same* similarity→size mapping, so a non-EB and an EB point with equal
    similarity differ only by the layer's `range` (the intended 15% size gap).
    The layered chart also resolves `size` independently (see below) so each
    layer's `range` is actually honoured rather than merged by Vega-Lite."""
    if has_query:
        return alt.Size(
            "similarity:Q",
            scale=alt.Scale(range=size_range, domain=sim_domain),
            legend=None,
        )
    return alt.value(flat)


def _color_enc(show_legend: bool):
    """Shared viridis-by-similarity scale; only one layer shows the legend."""
    if has_query:
        return alt.Color(
            "similarity:Q",
            scale=alt.Scale(scheme="viridis", domain=sim_domain, reverse=True),
            legend=alt.Legend(title="Similarity") if show_legend else None,
        )
    return alt.value("#4c78a8")


# Background "field" of recent Human Factors article abstracts.
background = (
    alt.Chart(abs_df)
    .mark_circle(size=20, color="#d4d4d8", opacity=0.45)
    .encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
        tooltip=alt.Tooltip("title:N", title="Recent HF article"),
    )
)

# Reviewers split by role: editorial-board members are solid circles; non-EB
# candidate authors (rank "Not_EB") read as secondary — smaller, fainter open
# circles. Drawn first so the editorial board sits on top where they overlap.
is_neb = view["rank"] == "Not_EB"
eb_points = (
    alt.Chart(view[~is_neb])
    .mark_circle(opacity=0.85)
    .encode(**enc, size=_size_enc([40, 600], 140), color=_color_enc(True))
)
neb_points = (
    alt.Chart(view[is_neb])
    .mark_point(filled=False, opacity=0.7, strokeWidth=2)
    # Non-EB circles 15% smaller (diameter) than the editorial board's, i.e.
    # the EB size/area range [40, 600]/140 scaled by 0.85**2 ≈ 0.7225.
    .encode(**enc, size=_size_enc([29, 434], 101), color=_color_enc(False))
)
# Non-EB drawn last so the open rings are never hidden under filled EB circles.
layers = [background, eb_points, neb_points]
if has_query:
    # Top-N reviewer names, dodged vertically so nearby labels don't overlap.
    lab = view.head(12).copy()
    lab["y_label"] = engine.dodge_label_y(
        lab["x"].to_numpy(), lab["y"].to_numpy(),
        view["x"].max() - view["x"].min(),
        view["y"].max() - view["y"].min(),
    )
    labels = (
        alt.Chart(lab)
        .mark_text(align="left", dx=8, dy=0, fontSize=10, color="#333")
        .encode(x="x:Q", y="y_label:Q", text="name:N")
    )
    query_df = pd.DataFrame({"x": [qcoord[0]], "y": [qcoord[1]], "label": ["Manuscript"]})
    submission = (
        alt.Chart(query_df)
        .mark_point(shape="circle", size=520, color="#ff3b30", filled=True, stroke="white", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q", tooltip=alt.value("Manuscript"))
    )
    submission_label = (
        alt.Chart(query_df)
        .mark_text(align="center", dy=-18, fontSize=12, fontWeight="bold", color="#ff3b30")
        .encode(x="x:Q", y="y:Q", text="label:N")
    )
    layers += [labels, submission, submission_label]

# No .interactive() — zoom/pan disabled.
chart = (
    alt.layer(*layers)
    .properties(height=520)
    # Size scales must be independent so the EB ([40,600]) and non-EB ([29,434])
    # ranges each apply; otherwise Vega-Lite merges them into one shared size
    # scale and the 15% size gap is lost. Color stays shared → one legend.
    .resolve_scale(size="independent")
    .configure_view(strokeWidth=0)
    .configure(font="-apple-system, BlinkMacSystemFont, 'SF Pro Text', Helvetica, sans-serif")
    .configure_legend(titleFontWeight="normal", labelColor="#6e6e73", titleColor="#6e6e73")
)
st.altair_chart(chart, use_container_width=True)

if not has_query:
    st.stop()

# ---- Ranked table (scrollable ~8 rows; filter + select a row for details) - #
TABLE_N = 25
st.subheader(f"Top {min(TABLE_N, len(view))} ranked reviewers")
name_filter = st.text_input(
    "Filter by reviewer name",
    placeholder="Filter by reviewer name…",
    label_visibility="collapsed",
)
table_src = view
if name_filter.strip():
    table_src = view[view["name"].str.contains(
        name_filter.strip(), case=False, na=False, regex=False)]
table_src = table_src.head(TABLE_N)
st.caption(
    (f"{len(table_src)} match “{name_filter.strip()}” — " if name_filter.strip() else "")
    + "select a row to see that reviewer's details below."
)
table = table_src[
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
# Pre-select the top-ranked reviewer the first time the table renders.
if "rev_table" not in st.session_state:
    st.session_state["rev_table"] = {"selection": {"rows": [0], "columns": []}}
event = st.dataframe(
    table,
    key="rev_table",
    hide_index=True,
    use_container_width=True,
    height=315,  # ~8 rows + header, then scrolls
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Similarity": st.column_config.ProgressColumn(
            "Similarity", min_value=0.0, max_value=float(view["similarity"].max()),
            format="%.2f",
        ),
    },
)

# ---- Detailed reviewer information (driven by table selection) ---------- #
sel = list(event.selection.rows) if getattr(event, "selection", None) else []
st.subheader("Reviewer details")
if len(table_src) == 0:
    st.info("No reviewers match that filter.")
    st.stop()
idx = sel[0] if (sel and sel[0] < len(table_src)) else 0
row = table_src.iloc[idx]
with st.container(border=True):
    st.markdown(f"**{row['name']}**  ·  _{row['rank']}_  ·  similarity **{row['similarity']:.2f}**")
    st.markdown(f"_{row['expertise_overview']}_")
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
