"""
Expert Analysis  ·  Scrape → Extract → RAG
===========================================
Three-tab workflow:
  1. Scrape     — fetch any match preview URL → clean Markdown
  2. Library    — browse, inspect and delete saved reports
  3. AI Analyst — RAG or direct LLM analysis over saved reports
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ── Allow importing from the project root ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_analysis.ingestion import ScrapingError, scrape_url
from expert_analysis.storage import (
    all_markdowns, delete_report, list_reports, load_markdown,
    load_report, save_report,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expert Analysis · WC 2026",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared CSS (Liquid Glass dark, matches main app) ──────────────────────────
st.markdown("""
<style>
/* ── global background ── */
.stApp { background: radial-gradient(ellipse at 20% 20%, #0d1b2a 0%, #0a0f1e 60%, #060b14 100%); }

/* ── glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.glass-card h4 { color: #e2e8f0; margin: 0 0 .5rem 0; font-size: 1rem; }

/* ── section headings ── */
h1, h2, h3 { color: #e2e8f0 !important; }
h2 { font-size: 1.35rem !important; }

/* ── metric pills ── */
.pill {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #90cdf4;
    margin: 2px 3px;
}
.pill-red   { background: rgba(252,129,74,0.15); border-color: rgba(252,129,74,0.3); color: #fc814a; }
.pill-green { background: rgba(72,187,120,0.15); border-color: rgba(72,187,120,0.3); color: #68d391; }
.pill-gold  { background: rgba(236,201,75,0.15); border-color: rgba(236,201,75,0.3); color: #f6e05e; }

/* ── quick-prompt chips ── */
.qchip button {
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
    border-radius: 16px !important;
}

/* ── tab strip ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 6px 18px;
    color: #94a3b8;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.18) !important;
    color: #90cdf4 !important;
}

/* ── text inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── buttons ── */
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.07) !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.18) !important;
    border-color: rgba(99,179,237,0.4) !important;
}

/* ── expanders ── */
.stExpander { background: rgba(255,255,255,0.03) !important; border-radius: 12px !important; }
/* Keep summary clickable — only hide the ligature text nodes, not the SVG arrow */
.stExpander details > summary { font-size: 0 !important; line-height: 1.6rem !important; min-height: 2rem; cursor: pointer; }
.stExpander details > summary svg { width: 1rem !important; height: 1rem !important; opacity: 0.5; }
.stExpander details > summary p  { font-size: 0.88rem !important; line-height: 1.4 !important; color: #94a3b8 !important; display: inline !important; }

/* ── code blocks ── */
pre { background: rgba(0,0,0,0.35) !important; border-radius: 10px !important; font-size: 0.8rem !important; }
/* Lock to the dark design — hide the theme switcher; keep chrome dark. */
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stHeader"], header[data-testid="stHeader"],
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDialog"] > div, div[role="dialog"] {
  background: #16161f !important; color: rgba(255,255,255,0.90) !important; }
/* Sidebar expand/collapse controls -> replace ghosting Material icon with a hamburger */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] { position: relative !important; }
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"] * { font-size: 0 !important; }
[data-testid="stSidebarCollapseButton"]::after,
[data-testid="stExpandSidebarButton"]::after {
  content: "\2630"; position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  font-size: 1.1rem !important; line-height: 1;
  color: rgba(255,255,255,0.6); pointer-events: none; }
</style>
""", unsafe_allow_html=True)

# ── Authentication gate ───────────────────────────────────────────────────────
from auth import require_login
require_login()

# ── Session-state helpers ─────────────────────────────────────────────────────

def _ss(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _llm_cfg():
    """Pull LLM config from session state (shared with main app if already set)."""
    base = st.session_state.get("ea_base_url") or st.session_state.get("llm_base_url", "http://127.0.0.1:11434/v1")
    model = st.session_state.get("ea_model") or st.session_state.get("llm_model", "deepseek-r1:1.5b")
    return base, model


# ── Initialise session state ──────────────────────────────────────────────────
_ss("ea_scrape_md", "")
_ss("ea_scrape_title", "")
_ss("ea_scrape_url", "")
_ss("ea_extracted", {})
_ss("ea_rag_engine", None)
_ss("ea_rag_built_slugs", [])
_ss("ea_chat_history", [])
_ss("ea_base_url", "http://127.0.0.1:11434/v1")
_ss("ea_model", "deepseek-r1:1.5b")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 Expert Analysis")
st.markdown(
    "<p style='color:#64748b;font-size:.9rem;margin-top:-.5rem;'>Pre-match scouting pipeline · "
    "Scrape → AI Extract → RAG Interrogation</p>",
    unsafe_allow_html=True,
)

tab_scrape, tab_library, tab_analyst = st.tabs(["⬇️ Scrape Preview", "📁 Report Library", "🧠 AI Analyst"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCRAPE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scrape:
    st.markdown("### Scrape Match Preview")

    # ─────────────────────────────────────────────────────────────────────────
    # MANUAL URL + FULL EXTRACTION WORKFLOW
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("#### ⬇️ Scrape a URL (with AI extraction)")
    st.markdown(
        "<p style='color:#64748b;font-size:.85rem;'>"
        "Paste a URL directly to scrape it, run AI structure extraction, then save.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Supported sources & tips"):
        st.markdown("""
**Works well (static HTML):** Sky Sports, BBC Sport, Opta Analyst, The Athletic, WhoScored, FBref

**Works with limitations (may need JavaScript):** ESPN, UEFA.com, FIFA.com
→ Install `crawl4ai` for JS-heavy sites: `pip install crawl4ai && playwright install`

**Rate limiting:** the scraper waits 3 s between requests to the same domain.

**Paywalled content:** only what's visible without a subscription will be extracted.
        """)

    url_input = st.text_input(
        "Match preview URL",
        value="",
        placeholder="https://www.skysports.com/football/preview/...",
        key="ea_url_input",
    )

    col_scrape, col_clear = st.columns([1, 5])
    scrape_clicked = col_scrape.button("⬇️ Scrape", use_container_width=True)
    if col_clear.button("✕ Clear", use_container_width=False):
        st.session_state.ea_scrape_md = ""
        st.session_state.ea_scrape_title = ""
        st.session_state.ea_scrape_url = ""
        st.session_state.ea_extracted = {}
        st.rerun()

    if scrape_clicked:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            with st.spinner("Fetching and extracting article…"):
                try:
                    md, title = scrape_url(url_input.strip())
                    st.session_state.ea_scrape_md = md
                    st.session_state.ea_scrape_title = title
                    st.session_state.ea_scrape_url = url_input.strip()
                    st.session_state.ea_extracted = {}
                    st.success(f"Scraped **{len(md):,}** characters · \"{title}\"")
                except ScrapingError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ── Show scraped content ───────────────────────────────────────────────────
    if st.session_state.ea_scrape_md:
        md_text = st.session_state.ea_scrape_md

        col_meta1, col_meta2 = st.columns(2)
        col_meta1.metric("Characters", f"{len(md_text):,}")
        col_meta2.metric("Words (approx)", f"{len(md_text.split()):,}")

        with st.expander("📄 Raw Markdown preview (first 3,000 chars)"):
            st.text(md_text[:3000] + ("…" if len(md_text) > 3000 else ""))

        st.divider()

        # ── AI Extraction ──────────────────────────────────────────────────────
        st.markdown("#### Extract Structure with AI")
        st.caption("Uses your local LLM to pull teams, injuries, probabilities, form, etc. from the text.")

        # Compact LLM config for this tab
        with st.expander("⚙️ LLM config for extraction"):
            st.text_input("Ollama base URL", key="ea_base_url",
                          help="Default: http://127.0.0.1:11434/v1")
            st.text_input("Model", key="ea_model",
                          help="e.g. deepseek-r1:1.5b  llama3  mistral")
            st.caption("Uses the same Ollama instance as the main app predictor.")

        extract_clicked = st.button("🤖 Extract structured data", use_container_width=False)
        if extract_clicked:
            base_url, model = _llm_cfg()
            if not model:
                st.warning("Set a model name in the LLM config above.")
            else:
                with st.spinner(f"Asking **{model}** to extract structure…"):
                    try:
                        from expert_analysis.rag_engine import RAGEngine
                        engine = RAGEngine(base_url=base_url, model=model)
                        extracted = engine.extract_structure(md_text)
                        st.session_state.ea_extracted = extracted
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

        # ── Display extracted structure ────────────────────────────────────────
        ex = st.session_state.ea_extracted
        if ex:
            st.markdown("**Extracted fields:**")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**🏠 Home:** {ex.get('home_team') or '—'}")
            c2.markdown(f"**✈️ Away:** {ex.get('away_team') or '—'}")
            c3.markdown(f"**📅 Date:** {ex.get('match_date') or '—'}")

            c4, c5, c6 = st.columns(3)
            c4.markdown(f"**🏟️ Venue:** {ex.get('venue') or '—'}")
            c5.markdown(f"**🏆 Tournament:** {ex.get('tournament') or '—'}")
            c6.markdown(f"**👔 Referee:** {ex.get('referee') or '—'}")

            # Injuries
            inj_home = ex.get("injured_players_home") or []
            inj_away = ex.get("injured_players_away") or []
            sus_home = ex.get("suspended_players_home") or []
            sus_away = ex.get("suspended_players_away") or []
            if inj_home or inj_away or sus_home or sus_away:
                st.markdown("**🚑 Absences:**")
                for team_label, players, colour in [
                    (f"{ex.get('home_team','Home')} injured", inj_home, "pill-red"),
                    (f"{ex.get('away_team','Away')} injured", inj_away, "pill-red"),
                    (f"{ex.get('home_team','Home')} suspended", sus_home, "pill-gold"),
                    (f"{ex.get('away_team','Away')} suspended", sus_away, "pill-gold"),
                ]:
                    if players:
                        pills = " ".join(f'<span class="pill {colour}">{p}</span>' for p in players)
                        st.markdown(f"<small>{team_label}:</small> {pills}", unsafe_allow_html=True)

            # Win probabilities
            ph = ex.get("win_probability_home")
            pd = ex.get("win_probability_draw")
            pa = ex.get("win_probability_away")
            if any(v is not None for v in [ph, pd, pa]):
                st.markdown("**📊 Win Probabilities (from article):**")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric(ex.get("home_team", "Home"), f"{ph:.0%}" if ph else "—")
                pc2.metric("Draw", f"{pd:.0%}" if pd else "—")
                pc3.metric(ex.get("away_team", "Away"), f"{pa:.0%}" if pa else "—")

            # Form
            fh = ex.get("form_home")
            fa = ex.get("form_away")
            if fh or fa:
                st.markdown(
                    f"**📈 Form:** "
                    f"{ex.get('home_team','Home')}: `{fh or '—'}` &nbsp;|&nbsp; "
                    f"{ex.get('away_team','Away')}: `{fa or '—'}`",
                    unsafe_allow_html=True,
                )

            # Key stats
            ks = ex.get("key_stats") or []
            if ks:
                st.markdown("**📌 Key stats from article:**")
                for stat in ks:
                    st.markdown(f"- {stat}")

            # Tactical narrative
            tn = ex.get("tactical_narrative", "")
            if tn:
                st.info(f"**🎯 Tactical summary:** {tn}")

        # ── Save ───────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Save Report")
        st.caption("Saves raw Markdown to `data/raw_markdown/` and structured JSON to `data/processed/`.")

        save_clicked = st.button("💾 Save report", use_container_width=False)
        if save_clicked:
            ex = st.session_state.ea_extracted
            base_url, model = _llm_cfg()
            try:
                if ex:
                    from expert_analysis.rag_engine import RAGEngine
                    engine = RAGEngine(base_url=base_url, model=model)
                    report = engine.extraction_to_report(
                        extracted=ex,
                        source_url=st.session_state.ea_scrape_url,
                        raw_markdown=st.session_state.ea_scrape_md,
                        title=st.session_state.ea_scrape_title,
                    )
                else:
                    # Save raw-only report (no structure extracted yet)
                    from expert_analysis.models import PreMatchReport
                    report = PreMatchReport(
                        source_url=st.session_state.ea_scrape_url,
                        report_title=st.session_state.ea_scrape_title,
                        raw_markdown=st.session_state.ea_scrape_md,
                    )

                json_path, md_path = save_report(report)
                st.success(
                    f"Saved as **{report.match_slug}**\n\n"
                    f"JSON → `{json_path.relative_to(json_path.parent.parent.parent)}`\n\n"
                    f"Markdown → `{md_path.relative_to(md_path.parent.parent.parent)}`"
                )
            except Exception as e:
                st.error(f"Save failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.markdown("### Saved Reports")

    lib_refresh = st.button("🔄 Refresh library", key="lib_refresh")
    reports = list_reports()

    if not reports:
        st.info("No reports saved yet. Scrape a match preview in the **Scrape** tab first.")
    else:
        st.caption(f"{len(reports)} report{'s' if len(reports) != 1 else ''} saved locally")

        for r in reports:
            slug = r["slug"]
            with st.expander(
                f"**{r['home_team']} vs {r['away_team']}** · {r['match_date']} · {r['tournament']}"
            ):
                col_info, col_actions = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**Slug:** `{slug}`")
                    st.markdown(f"**Source:** [{r['source_url'][:60]}…]({r['source_url']})"
                                if len(r["source_url"]) > 60
                                else f"**Source:** [{r['source_url']}]({r['source_url']})")
                    st.markdown(f"**Saved:** {r['extraction_timestamp'][:19].replace('T', ' ')}")
                    st.markdown(
                        f"**Structure extracted:** "
                        + ("✅ Yes" if r["has_structure"] else "❌ Raw only (re-extract in Scrape tab)")
                    )

                    # Show JSON detail
                    with st.expander("View full JSON"):
                        try:
                            rep = load_report(slug)
                            st.json(rep.model_dump(mode="json"))
                        except Exception as e:
                            st.error(str(e))

                    # Show markdown preview
                    with st.expander("View raw Markdown (first 2,000 chars)"):
                        md_content = load_markdown(slug)
                        st.text(md_content[:2000] + ("…" if len(md_content) > 2000 else ""))

                with col_actions:
                    if st.button("🗑️ Delete", key=f"del_{slug}"):
                        delete_report(slug)
                        st.success(f"Deleted `{slug}`")
                        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI ANALYST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyst:
    st.markdown("### AI Analyst")
    st.markdown(
        "<p style='color:#64748b;font-size:.85rem;'>RAG-powered interrogation of your saved scouting reports "
        "using a local LLM — no API key, no cloud.</p>",
        unsafe_allow_html=True,
    )

    # ── Dependency check ───────────────────────────────────────────────────────
    try:
        from expert_analysis.rag_engine import RAGEngine
        missing = RAGEngine.check_deps()
    except ImportError as _ie:
        missing = ["langchain", "langchain-community", "faiss-cpu"]

    if missing:
        st.warning(
            f"Missing packages: `{', '.join(missing)}`\n\n"
            "Install with:\n"
            f"```\npip install {' '.join(missing)}\n```\n\n"
            "Then restart the app."
        )
        st.stop()

    # ── LLM config ────────────────────────────────────────────────────────────
    with st.expander("⚙️ LLM config (Ollama)"):
        col_url, col_mdl = st.columns(2)
        col_url.text_input("Base URL", key="ea_base_url",
                           help="Ollama endpoint, e.g. http://127.0.0.1:11434/v1")
        col_mdl.text_input("Model", key="ea_model",
                           help="e.g. deepseek-r1:1.5b  llama3  mistral  qwen2.5")
        st.caption(
            "Make sure Ollama is running (`ollama serve`) and the model is pulled "
            "(`ollama pull deepseek-r1:1.5b`). Embeddings also use this model."
        )

    base_url, model = _llm_cfg()

    # ── Report selection ───────────────────────────────────────────────────────
    reports = list_reports()
    if not reports:
        st.info("Save at least one report in the **Scrape** tab to start analysing.")
        st.stop()

    st.markdown("#### Select Reports")
    report_options = {
        f"{r['home_team']} vs {r['away_team']} ({r['match_date']})": r["slug"]
        for r in reports
    }
    selected_labels = st.multiselect(
        "Choose reports to include in analysis",
        options=list(report_options.keys()),
        default=list(report_options.keys())[:1],
    )
    selected_slugs = [report_options[l] for l in selected_labels]

    # ── Analysis mode ──────────────────────────────────────────────────────────
    analysis_mode = st.radio(
        "Analysis mode",
        ["Direct (single report, fast)", "RAG (multi-report, vector search)"],
        horizontal=True,
        help=(
            "Direct: feeds markdown straight into LLM context. Best for one report.\n"
            "RAG: builds a FAISS index for semantic retrieval across multiple reports."
        ),
    )
    use_rag = analysis_mode.startswith("RAG")

    # ── Build index (RAG only) ─────────────────────────────────────────────────
    if use_rag:
        build_needed = set(selected_slugs) != set(st.session_state.ea_rag_built_slugs)
        col_idx1, col_idx2 = st.columns([2, 4])
        if col_idx1.button("⚡ Build / Rebuild index", use_container_width=True) or (
            use_rag and build_needed and st.session_state.ea_rag_engine is None
        ):
            if not selected_slugs:
                st.warning("Select at least one report first.")
            elif not model:
                st.warning("Configure a model in the LLM config above.")
            else:
                docs = [(s, load_markdown(s)) for s in selected_slugs]
                with st.spinner(f"Building FAISS index with **{model}** embeddings…"):
                    try:
                        engine = RAGEngine(base_url=base_url, model=model)
                        n_chunks = engine.build_index(docs)
                        st.session_state.ea_rag_engine = engine
                        st.session_state.ea_rag_built_slugs = list(selected_slugs)
                        col_idx2.success(f"Index ready — {n_chunks} chunks from {len(docs)} report(s).")
                    except Exception as e:
                        st.error(f"Index build failed: {e}")

        if st.session_state.ea_rag_built_slugs:
            st.caption(
                f"Index covers: {', '.join(st.session_state.ea_rag_built_slugs)}"
                + (" ⚠️ Rebuild needed — selection changed" if build_needed else " ✅")
            )

    st.divider()

    # ── Quick-prompt chips ─────────────────────────────────────────────────────
    st.markdown("#### Ask the Analyst")

    QUICK_PROMPTS = [
        "What are the key injury-induced tactical vulnerabilities for each side?",
        "Which defensive phase weaknesses can the away team exploit?",
        "Analyse the pressing triggers and defensive block shape of both teams.",
        "What set-piece threats should each team fear most?",
        "Which individual duel is most likely to decide this match?",
        "Compare the form and momentum of both teams over recent matches.",
        "How do the probable lineups affect the tactical balance?",
        "What is the most likely match script — high press, low block, or transitions?",
    ]

    st.markdown("<div class='qchip'>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    chosen_quick = None
    for i, prompt in enumerate(QUICK_PROMPTS):
        short = prompt[:38] + "…" if len(prompt) > 38 else prompt
        if chip_cols[i % 4].button(short, key=f"chip_{i}", use_container_width=True):
            chosen_quick = prompt
    st.markdown("</div>", unsafe_allow_html=True)

    custom_q = st.text_area(
        "Or type a custom question:",
        value=chosen_quick or "",
        height=80,
        key="ea_custom_q",
        placeholder="e.g. How will the absence of the defensive midfielder affect the pressing structure?",
    )

    # ── Run analysis ───────────────────────────────────────────────────────────
    run_col, _ = st.columns([1, 4])
    run_clicked = run_col.button("🧠 Analyse", use_container_width=True)

    if run_clicked:
        question = (custom_q or chosen_quick or "").strip()
        if not question:
            st.warning("Enter a question or click one of the quick-prompt chips.")
        elif not selected_slugs:
            st.warning("Select at least one report.")
        elif not model:
            st.warning("Configure a model in the LLM config above.")
        else:
            st.session_state.ea_chat_history.append({"role": "user", "content": question})
            with st.spinner("Analysing…"):
                try:
                    engine = RAGEngine(base_url=base_url, model=model)

                    if use_rag:
                        if st.session_state.ea_rag_engine is None:
                            st.error("Build the index first (click ⚡ Build / Rebuild index).")
                        else:
                            engine = st.session_state.ea_rag_engine
                            answer = engine.query(question)
                    else:
                        # Direct analysis — concatenate selected markdowns
                        combined_md = "\n\n---\n\n".join(
                            load_markdown(s) for s in selected_slugs
                        )
                        answer = engine.direct_analysis(combined_md, question)

                    st.session_state.ea_chat_history.append({"role": "analyst", "content": answer})
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # ── Chat history ───────────────────────────────────────────────────────────
    if st.session_state.ea_chat_history:
        st.divider()
        st.markdown("#### Analysis Thread")
        clear_col, _ = st.columns([1, 6])
        if clear_col.button("🗑️ Clear thread"):
            st.session_state.ea_chat_history = []
            st.rerun()

        for turn in reversed(st.session_state.ea_chat_history):
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="glass-card" style="border-color:rgba(99,179,237,0.25);">'
                    f'<small style="color:#63b3ed;">❓ Question</small><br>'
                    f'<span style="color:#e2e8f0;">{turn["content"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="glass-card" style="border-color:rgba(72,187,120,0.2);">'
                    f'<small style="color:#68d391;">🧠 Analyst</small><br>'
                    f'<span style="color:#cbd5e0;line-height:1.7;">{turn["content"]}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Footer info ────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ How this pipeline works"):
        st.markdown("""
**Ingestion** — `crawl4ai` / `trafilatura` + `markdownify` fetch the raw article HTML and convert it
to clean Markdown, stripping ads, nav bars and boilerplate.

**Extraction** — your local Ollama LLM reads the Markdown and returns a structured JSON matching the
`PreMatchReport` schema (teams, injuries, probabilities, form, etc.).

**Storage** — two files are written per report:
- `data/processed/<slug>.json` — structured Pydantic model (queryable)
- `data/raw_markdown/<slug>.md` — full article text (used for embeddings)

**RAG** — LangChain splits the Markdown into 700-char chunks with 120-char overlap,
embeds them with Ollama (same model), stores in a FAISS vector index, and
retrieves the *k* most relevant chunks for each question before passing them
to the LLM with the Pro Football Performance Analyst system prompt.

**Direct mode** — skips the vector index entirely and stuffs up to 10,000 characters
of Markdown directly into the LLM prompt. Faster for single-report queries.

All data stays on your machine — no external API calls are made.
        """)
