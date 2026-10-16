"""
Signal: find what your users are really saying.

Run with:  streamlit run app.py
"""

import html
import os

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import synthesizer as syn

load_dotenv()
st.set_page_config(page_title="Signal", page_icon="〰️", layout="centered")

SAMPLE_PATH = "sample_data/stride_reviews.csv"
EXAMPLE_PATH = "sample_data/example_report.json"   # pre-built report shown without a key
MODELS = {
    "Balanced (Claude Sonnet)": "claude-sonnet-5-5",
    "Fast and cheapest (Claude Haiku)": "claude-haiku-5-5",
}

# Pastel tones per kind of theme, from strongest to softest.
TONES = {
    "problem": ["#E9A3B6", "#F0B9C7", "#F5CCD7", "#F9DDE4", "#FBE8ED"],
    "request": ["#EED69A", "#F3E2B3", "#F7ECCB"],
    "praise":  ["#A9D8BE", "#C3E4D1", "#D9EEE2"],
    "other":   ["#DCDAE6"],
}
KIND_LABEL = {"problem": "Problem", "request": "Request", "praise": "Praise"}


# --------------------------------------------------------------------------- #
#  Look and feel
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Figtree:wght@400;500;600&display=swap');

:root {
  --ink: #2E2A3D; --muted: #7A7590; --line: #E4E1EE; --paper: #FFFFFF;
  --lav: #7B66C9; --lav-soft: #EEEBF7; --rose: #B4536E; --sage: #3F8A63;
}
html, body, .stApp, .stMarkdown, .stMarkdown *, button, input, textarea {
  font-family: 'Figtree', system-ui, sans-serif;
}
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.5rem; max-width: 780px; }

h1, h2, h3, .wordmark, .step, .headline, .theme-name, .rank, .quote, .stats strong { font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important; letter-spacing: -0.01em; color: var(--ink); }

/* Hero */
.hero { margin-bottom: 2.2rem; }
.wordmark { font-family: 'Instrument Serif', serif; font-size: 4.2rem; line-height: 1; color: var(--ink); margin: 0; }
.wave { display: flex; align-items: flex-end; gap: 4px; height: 34px; margin: .9rem 0 1.1rem; }
.wave span { width: 6px; border-radius: 3px; display: block; }
.tagline { font-size: 1.15rem; color: var(--muted); max-width: 34rem; line-height: 1.55; margin: 0; }

/* Section headings */
.step { font-family: 'Instrument Serif', serif; font-size: 1.7rem; color: var(--ink); margin: 1.6rem 0 .2rem; }
.hint { color: var(--muted); font-size: .95rem; margin: 0 0 .8rem; }

/* Inputs & buttons */
.stButton > button[kind="primary"], .stDownloadButton > button {
  border-radius: 999px; padding: .55rem 1.5rem; font-weight: 600; border: none;
}
.stButton > button[kind="primary"] { background: var(--ink); color: #fff; }
.stButton > button[kind="primary"]:hover { background: var(--lav); color: #fff; }
.stButton > button[kind="secondary"] { border-radius: 999px; padding: .55rem 1.4rem; font-weight: 500;
  background: transparent; border: 1.5px solid #CFC8E6; color: var(--ink); }
.stButton > button[kind="secondary"]:hover { border-color: var(--lav); color: var(--lav); }
.example-note { border: 1.5px dashed #CFC8E6; border-radius: 14px; padding: .8rem 1.1rem; color: var(--muted);
  font-size: .92rem; line-height: 1.5; margin-bottom: 1rem; }
.example-note b { color: var(--ink); font-weight: 600; }
.stDownloadButton > button { background: var(--lav-soft); color: var(--ink); }
.stDownloadButton > button:hover { background: #E2DCF5; color: var(--ink); }
[data-testid="stFileUploaderDropzone"] { background: var(--paper); border: 1.5px dashed #CFC8E6; border-radius: 14px; }
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 14px; background: var(--paper); }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #C9BCF0, #E9A3B6); }

/* Results */
.headline { font-family: 'Instrument Serif', serif; font-size: 2.5rem; line-height: 1.12; color: var(--ink); margin: .4rem 0 1rem; }
.summary { font-size: 1.08rem; line-height: 1.65; color: #4A4560; max-width: 40rem; }
.reco { background: var(--lav-soft); border-radius: 18px; padding: 1.1rem 1.3rem; margin: 1.3rem 0 1.6rem; color: var(--ink); line-height: 1.55; }
.reco b { display: block; font-weight: 600; margin-bottom: .2rem; color: var(--lav); }
.stats { display: flex; gap: 2.4rem; margin: .4rem 0 1.4rem; flex-wrap: wrap; }
.stats div { color: var(--muted); font-size: .92rem; }
.stats strong { display: block; font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 2rem; color: var(--ink); line-height: 1.1; }

/* Spectrum: every theme's share of all comments, in one bar */
.spectrum { display: flex; height: 22px; border-radius: 999px; overflow: hidden; gap: 3px; margin: .2rem 0 .8rem; }
.spectrum span { display: block; height: 100%; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; font-size: .86rem; color: var(--muted); margin-bottom: 1.8rem; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .4rem; vertical-align: -1px; }

/* Theme rows */
.theme { padding: 1.4rem 0 1.5rem; border-top: 1px solid var(--line); }
.theme-head { display: flex; align-items: baseline; gap: .9rem; }
.rank { font-family: 'Instrument Serif', serif; font-size: 1.6rem; color: #B8B2CC; min-width: 1.4rem; }
.theme-name { font-family: 'Instrument Serif', serif; font-size: 1.65rem; color: var(--ink); flex: 1; line-height: 1.15; }
.count { font-size: .9rem; color: var(--muted); white-space: nowrap; }
.desc { color: #4A4560; margin: .35rem 0 .8rem 2.3rem; line-height: 1.55; }
.bar { display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: #EFEDF5; margin: 0 0 .45rem 2.3rem; }
.bar .neg { background: #E9A3B6; } .bar .neu { background: #D6D2E3; } .bar .pos { background: #A9D8BE; }
.split { font-size: .82rem; color: var(--muted); margin: 0 0 .9rem 2.3rem; }
.quote { font-family: 'Instrument Serif', serif; font-style: italic; font-size: 1.12rem; color: #4A4560; line-height: 1.45;
         border-left: 2px solid #D8CFF3; padding: .1rem 0 .1rem .9rem; margin: .55rem 0 0 2.3rem; }

.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
.stTabs [data-baseweb="tab"] { font-size: 1rem; }
.empty { color: var(--muted); padding: 1.5rem 0; }

@media (max-width: 640px) {
  .wordmark { font-size: 3.2rem; } .headline { font-size: 2rem; }
  .desc, .bar, .split, .quote { margin-left: 0; }
}
</style>
""", unsafe_allow_html=True)


def e(text) -> str:
    """Escape user text before putting it inside HTML."""
    return html.escape(str(text))


# --------------------------------------------------------------------------- #
#  Hero
# --------------------------------------------------------------------------- #
wave_heights = [10, 18, 12, 26, 34, 22, 14, 30, 20, 11, 16, 8]
wave_colors = ["#C9BCF0", "#E9A3B6", "#A9D8BE", "#EED69A"]
wave = "".join(
    f'<span style="height:{h}px;background:{wave_colors[i % 4]}"></span>'
    for i, h in enumerate(wave_heights)
)
st.markdown(f"""
<div class="hero">
  <div class="wordmark">Signal</div>
  <div class="wave">{wave}</div>
  <div class="tagline">Drop in your customer feedback. Get back the problems that matter most,
  how many people mention them, and their own words as proof.</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Step 1: choose the feedback
# --------------------------------------------------------------------------- #
st.markdown('<div class="step">Your feedback</div>'
            '<div class="hint">A CSV with one comment per row: app reviews, support tickets, survey answers.</div>',
            unsafe_allow_html=True)

source = st.segmented_control(
    "Source", ["Try sample reviews", "Upload a CSV"], default="Try sample reviews",
    label_visibility="collapsed",
)

df, context = None, ""
if source == "Upload a CSV":
    upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
        except Exception:
            st.error("This file couldn't be read as a CSV. Save it as CSV (UTF-8) and upload it again.")
    context = st.text_input("What product is this feedback about? (optional)",
                            placeholder="e.g. a meal-kit delivery app for busy families")
else:
    df = pd.read_csv(SAMPLE_PATH)
    context = "Stride, a running and fitness app with watch sync, guided training plans and a paid subscription."
    st.caption("150 reviews of Stride, a made-up running app. Five issues are planted in the data, "
               "so you can check that Signal finds them.")

text_col = None
if df is not None:
    text_cols = [c for c in df.columns
                 if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if not text_cols:
        st.error("This file has no text columns. Signal needs a column of written comments.")
        df = None
    else:
        # Guess the comment column: the one with the longest text on average.
        guess = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
        text_col = st.selectbox("Which column holds the comments?", text_cols,
                                index=text_cols.index(guess))
        n = int(df[text_col].astype(str).str.strip().astype(bool).sum())
        note = f" Signal will analyze the first {syn.MAX_ROWS}." if n > syn.MAX_ROWS else ""
        with st.expander(f"Preview: {n} comments ready.{note}"):
            st.dataframe(df.head(20), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
#  Settings: API key and model
# --------------------------------------------------------------------------- #
env_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
with st.expander("Settings"):
    if env_key:
        st.caption("Using the Anthropic API key from your .env file.")
        typed_key = ""
    else:
        typed_key = st.text_input(
            "Anthropic API key", type="password", placeholder="sk-ant-...",
            help="Used only for this session and never saved. Get one at console.anthropic.com.",
        )
    model_label = st.radio("Model", list(MODELS), horizontal=True)
api_key = env_key or typed_key.strip()


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
run_key = (source, text_col, len(df) if df is not None else 0)
if st.session_state.get("run_key") != run_key:      # new data means old results no longer apply
    st.session_state.pop("report", None)

has_example = source == "Try sample reviews" and os.path.exists(EXAMPLE_PATH)
b1, b2, _ = st.columns([1.1, 1.5, 2])
analyze = b1.button("Analyze feedback", type="primary", disabled=df is None)
if has_example and b2.button("See an example report", type="secondary"):
    report, note = syn.load_report(EXAMPLE_PATH)
    st.session_state.update(report=report, example_note=note, run_key=run_key)

if analyze:
    if not api_key:
        tip = " Or click See an example report to view one without a key." if has_example else ""
        st.warning(f"Add your Anthropic API key under Settings to run the analysis.{tip}")
    else:
        bar = st.progress(0.0, text="Starting")
        try:
            report = syn.run(df, text_col, api_key, MODELS[model_label], context,
                             on_step=lambda label, p: bar.progress(min(p, 1.0), text=label))
            st.session_state.update(report=report, example_note=None, run_key=run_key)
        except anthropic.AuthenticationError:
            st.error("The API key was rejected. Check it under Settings and try again.")
        except anthropic.APIError as err:
            st.error(f"The AI service returned an error: {getattr(err, 'message', err)}")
        except (ValueError, RuntimeError) as err:
            st.error(str(err))
        finally:
            bar.empty()


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #
def theme_colors(themes):
    used = {k: 0 for k in TONES}
    colors = {}
    for t in themes:
        palette = TONES[t.kind]
        colors[t.name] = palette[min(used[t.kind], len(palette) - 1)]
        used[t.kind] += 1
    colors[syn.OTHER] = TONES["other"][0]
    return colors


def render_theme(t, rank, color):
    total = max(t.mentions, 1)
    neg, neu, pos = (100 * x / total for x in (t.negative, t.neutral, t.positive))
    quotes = "".join(f'<div class="quote">“{e(q)}”</div>' for q in t.quotes)
    mention_word = "mention" if t.mentions == 1 else "mentions"
    st.markdown(f"""
    <div class="theme">
      <div class="theme-head">
        <span class="rank">{rank}</span>
        <span class="theme-name"><i style="display:inline-block;width:10px;height:10px;border-radius:50%;
              background:{color};margin-right:.55rem;vertical-align:4px"></i>{e(t.name)}</span>
        <span class="count">{t.mentions} {mention_word}, {t.share:g}%</span>
      </div>
      <div class="desc">{e(t.description)}</div>
      <div class="bar"><span class="neg" style="width:{neg}%"></span><span class="neu" style="width:{neu}%"></span><span class="pos" style="width:{pos}%"></span></div>
      <div class="split">{t.negative} negative, {t.neutral} neutral, {t.positive} positive</div>
      {quotes}
    </div>""", unsafe_allow_html=True)


report = st.session_state.get("report")
if report:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    note = st.session_state.get("example_note")
    if note is not None:
        st.markdown(f'<div class="example-note"><b>Example report.</b> This was prepared ahead of time, '
                    f'not run just now. {e(note)} Add your API key under Settings to run a live '
                    f'analysis.</div>', unsafe_allow_html=True)
    negative_pct = round(100 * (report.tagged["sentiment"] == "negative").mean())
    st.markdown(f"""
    <div class="headline">{e(report.headline)}</div>
    <div class="summary">{e(report.summary)}</div>
    <div class="reco"><b>If you fix one thing</b>{e(report.recommendation)}</div>
    <div class="stats">
      <div><strong>{report.total}</strong>comments read</div>
      <div><strong>{len(report.themes)}</strong>themes found</div>
      <div><strong>{negative_pct}%</strong>negative overall</div>
    </div>""", unsafe_allow_html=True)
    if report.truncated:
        st.caption(f"Only the first {syn.MAX_ROWS} comments were analyzed.")

    colors = theme_colors(report.themes)
    shares = report.tagged["theme"].value_counts(normalize=True)
    order = [t.name for t in report.themes] + ([syn.OTHER] if syn.OTHER in shares else [])
    spectrum = "".join(
        f'<span title="{e(n)}: {shares[n]:.0%}" style="flex:{shares[n]};background:{colors[n]}"></span>'
        for n in order
    )
    legend = "".join(f'<span><i style="background:{colors[n]}"></i>{e(n)}</span>' for n in order)
    st.markdown(f'<div class="spectrum">{spectrum}</div><div class="legend">{legend}</div>',
                unsafe_allow_html=True)

    problems, requests, praise = (report.by_kind(k) for k in ("problem", "request", "praise"))
    tabs = st.tabs([f"Problems ({len(problems)})", f"Requests ({len(requests)})",
                    f"Praise ({len(praise)})", "All comments"])
    for tab, group, empty in [
        (tabs[0], problems, "No problems stood out in this feedback."),
        (tabs[1], requests, "No feature requests stood out in this feedback."),
        (tabs[2], praise, "No clear praise themes in this feedback."),
    ]:
        with tab:
            if not group:
                st.markdown(f'<div class="empty">{empty}</div>', unsafe_allow_html=True)
            for i, t in enumerate(group, 1):
                render_theme(t, i, colors[t.name])

    with tabs[3]:
        pick = st.selectbox("Show comments about", ["All themes"] + order)
        view = report.tagged if pick == "All themes" else report.tagged[report.tagged["theme"] == pick]
        st.dataframe(view, hide_index=True, width="stretch")

    st.markdown('<div class="step">Take it with you</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download report (.md)", syn.to_markdown(report, "Signal feedback report"),
                       file_name="signal_report.md", mime="text/markdown", width="stretch")
    c2.download_button("Download tagged comments (.csv)", report.tagged.to_csv(index=False),
                       file_name="signal_tagged_comments.csv", mime="text/csv", width="stretch")
