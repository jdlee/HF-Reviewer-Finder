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
    return engine.embed_query(text)


@st.cache_data(show_spinner=False)
def load_abstracts() -> pd.DataFrame:
    return engine.load_abstracts()


@st.cache_data(show_spinner="Embedding background article abstracts…")
def abstract_embeddings(texts: tuple[str, ...]):
    return engine.embed_corpus(list(texts))


df = load_data()
abstracts = load_abstracts()
abs_emb = abstract_embeddings(tuple(abstracts["abstract_text"].tolist()))

# Default manuscript shown (and ranked) on first load.
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
for _k in ("query", "query_input"):
    if _k not in st.session_state:
        st.session_state[_k] = DEFAULT_QUERY

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
        help="PCA is fast & deterministic. UMAP clusters topics more tightly "
             "(requires umap-learn; falls back to PCA if unavailable).",
    )
    st.caption("Citations are LLM-researched from the web — verify before formal use.")

query = st.session_state.get("query", "")
has_query = bool(query.strip())

st.title(f"🔍 HF Reviewer Finder: {len(df)} reviewers mapped and ranked")
st.caption(
    "Local sentence-transformer embeddings of each *Human Factors* reviewer's expertise and "
    "key publications, ranked against your manuscript."
)
st.caption(
    "Each point is a reviewer, positioned by expertise. "
    + ("The red circle is your manuscript; closer + brighter = better topical match."
       if has_query else
       "Enter a manuscript description in the sidebar to rank reviewers by relevance.")
)

# ---- Compute (map always; ranking only when a query is present) ---------- #
corpus = corpus_embeddings(tuple(df["profile_text"].tolist()))
if has_query:
    _model()  # warm the model cache (shows spinner once)
    q = query_vec(query)
    sims = engine.cosine_similarity(corpus, q)
    coords, qcoord, abs_coords = engine.project_2d(corpus, q, extra_emb=abs_emb, method=method)
else:
    coords, qcoord, abs_coords = engine.project_2d(corpus, None, extra_emb=abs_emb, method=method)

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
    enc["size"] = alt.Size("similarity:Q", scale=alt.Scale(range=[40, 600]), legend=None)
    enc["color"] = alt.Color(
        "similarity:Q",
        scale=alt.Scale(scheme="viridis", domain=sim_domain),
        legend=alt.Legend(title="Similarity"),
    )
else:
    enc["size"] = alt.value(140)
    enc["color"] = alt.value("#4c78a8")

# Background "field" of the last 500 Human Factors article abstracts.
background = (
    alt.Chart(abs_df)
    .mark_circle(size=20, color="#d4d4d8", opacity=0.45)
    .encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False)),
        tooltip=alt.Tooltip("title:N", title="Recent HF article"),
    )
)
points = alt.Chart(view).mark_circle(opacity=0.85).encode(**enc)
layers = [background, points]
if has_query:
    labels = (
        alt.Chart(view.head(12))
        .mark_text(align="left", dx=8, dy=0, fontSize=10, color="#333")
        .encode(x="x:Q", y="y:Q", text="name:N")
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
    .configure_view(strokeWidth=0)
    .configure(font="-apple-system, BlinkMacSystemFont, 'SF Pro Text', Helvetica, sans-serif")
    .configure_legend(titleFontWeight="normal", labelColor="#6e6e73", titleColor="#6e6e73")
)
st.altair_chart(chart, use_container_width=True)

if not has_query:
    st.stop()

# ---- Ranked table (top 20, scrollable ~8 rows; select a row for details) - #
TABLE_N = 20
st.subheader(f"Top {min(TABLE_N, len(view))} ranked reviewers")
st.caption("Select a row to see that reviewer's details below.")
table = view.head(TABLE_N)[
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
idx = sel[0] if sel else 0
row = view.iloc[idx]
st.subheader("Reviewer details")
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
