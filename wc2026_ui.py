"""
FIFA World Cup 2026 — Prediction UI  (Liquid Glass / macOS 26 design)
Run:  streamlit run wc2026_ui.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fifa_wc2026_predictor import (
    build_dataset, engineer_features, PoissonMatchPredictor,
    simulate_match, build_team_profile, WC2026_HOSTS,
    fetch_wc2026_fixtures, ESPN_TEAM_IDS,
    fetch_match_odds, blend_lambdas,
    compute_team_form, tournament_form_lambda_adjustment,
    fetch_polymarket_wc_odds_all, fetch_polymarket_team_odds,
    fetch_polymarket_match_odds, fetch_polymarket_match_odds_from_url,
    _wc_stats_for,
    HIST_PARQUET, CACHE_DIR,
)

# ── Saved-prediction store (per-session, in-memory) ───────────────────────────
# Predictions live ONLY in this browser session's memory — never written to
# disk. They survive reruns within the session but vanish when the tab closes,
# and one user's saves can never collide with another's. Keyed by match id
# (event_id if available, else "Home vs Away").
def _preds_store() -> dict:
    if "saved_predictions" not in st.session_state:
        st.session_state["saved_predictions"] = {}
    return st.session_state["saved_predictions"]


def _match_key(home, away, event_id=None):
    if event_id:
        return str(event_id)
    return f"{home} vs {away}"


def load_predictions() -> dict:
    return dict(_preds_store())


def save_prediction(key: str, record: dict):
    _preds_store()[key] = record


def delete_prediction(key: str):
    _preds_store().pop(key, None)


st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Liquid Glass Dark CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Force dark mode on html/body so Streamlit's own bg becomes dark ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  color-scheme: dark !important;
}

/* ── Apple design tokens ── */
:root {
  --apple-blue: #0A84FF;            /* SF system blue (dark mode)        */
  --apple-blue-hover: #409CFF;
  --ease-apple: cubic-bezier(0.28, 0.11, 0.32, 1);  /* Apple's signature spring ease */
}

/* ── System font stack (SF Pro on macOS) ── */
* {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
               "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ── Apple-style smooth transitions on every interactive surface ── */
.glass, .metric-card, .team-pill, .score-badge, .venue-tag, .player-ok,
.player-inj, .fix-row, button, .stSelectbox > div > div, .stExpander,
.stTabs [data-baseweb="tab"], div[data-testid="metric-container"] {
  transition: transform 0.4s var(--ease-apple),
              box-shadow 0.4s var(--ease-apple),
              background 0.3s var(--ease-apple),
              border-color 0.3s var(--ease-apple) !important;
}

/* ── Canvas: restrained near-black with whisper-subtle aurora —
   apple.com dark hero feel rather than a busy mesh ── */
.stApp {
  background:
    radial-gradient(ellipse 70% 50% at 20% 0%,  rgba(99,102,241,0.14) 0%,  transparent 60%),
    radial-gradient(ellipse 60% 50% at 90% 100%, rgba(120,90,200,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 80% 60% at 50% 50%,  rgba(40,60,120,0.06)  0%, transparent 75%),
    linear-gradient(180deg, #000000 0%, #060608 50%, #000000 100%);
  min-height: 100vh;
}

/* ── Sidebar: dark frosted glass ── */
section[data-testid="stSidebar"] {
  background: rgba(15,15,28,0.72) !important;
  backdrop-filter: blur(48px) saturate(160%) brightness(0.95) !important;
  -webkit-backdrop-filter: blur(48px) saturate(160%) brightness(0.95) !important;
  border-right: 1px solid rgba(255,255,255,0.07) !important;
  box-shadow: 4px 0 32px rgba(0,0,0,0.40) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ── Kill Streamlit's default bg ── */
.main .block-container {
  background: transparent !important;
  padding-top: 1.5rem !important;
  max-width: 1280px;
}

/* ── Typography — Apple SF Pro Display scale, tight optical tracking ── */
h1 { color: rgba(255,255,255,0.96) !important; font-weight: 700 !important;
     letter-spacing: -0.035em !important; line-height: 1.05 !important;
     font-size: 2.6rem !important; }
h2 { color: rgba(255,255,255,0.95) !important; font-weight: 650 !important;
     letter-spacing: -0.028em !important; line-height: 1.1 !important; }
h3 { color: rgba(255,255,255,0.92) !important; font-weight: 600 !important;
     letter-spacing: -0.022em !important; }
p, li             { color: rgba(255,255,255,0.62); letter-spacing: -0.005em; line-height: 1.6; }
span              { color: inherit; }
label,
.stSelectbox label,
.stSlider label   { color: rgba(255,255,255,0.58) !important; font-weight: 500 !important;
                    letter-spacing: -0.01em !important; }
.stMarkdown p     { color: rgba(255,255,255,0.62); }
caption           { color: rgba(255,255,255,0.34) !important; }

/* ── Dark glass card (the core Liquid Glass surface in dark mode) ──
   Key recipe: very dark semi-transparent bg + white hairline border +
   subtle top specular (inset) + diffused outer glow               ── */
.glass,
.metric-card {
  background: rgba(255,255,255,0.045);
  backdrop-filter: blur(40px) saturate(160%) brightness(1.08);
  -webkit-backdrop-filter: blur(40px) saturate(160%) brightness(1.08);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 24px;
  box-shadow:
    0 0.5px 0 rgba(255,255,255,0.12) inset,  /* crisp top specular     */
    0 -1px 0 rgba(0,0,0,0.18) inset,         /* bottom depth shadow    */
    0 12px 40px rgba(0,0,0,0.30),
    0 2px 8px  rgba(0,0,0,0.16);
}
.metric-card {
  padding: 22px 14px;
  text-align: center;
  margin: 4px 0;
}
.metric-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255,255,255,0.14);
  box-shadow:
    0 0.5px 0 rgba(255,255,255,0.14) inset,
    0 16px 48px rgba(0,0,0,0.38),
    0 2px 8px  rgba(0,0,0,0.18);
}
.metric-card .val {
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.05;
  margin-bottom: 5px;
}
.metric-card .lbl {
  font-size: 0.72rem;
  font-weight: 500;
  color: rgba(255,255,255,0.38);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Team name pills ── */
.team-pill {
  border-radius: 16px;
  padding: 14px 20px;
  text-align: center;
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-bottom: 6px;
  box-shadow:
    0 1px 0 rgba(255,255,255,0.12) inset,
    0 4px 20px rgba(0,0,0,0.35);
}
.home-pill {
  background: linear-gradient(135deg, rgba(99,102,241,0.28) 0%, rgba(139,92,246,0.22) 100%);
  border: 1px solid rgba(129,140,248,0.30);
  color: rgba(199,210,254,0.95);
}
.away-pill {
  background: linear-gradient(135deg, rgba(239,68,68,0.25) 0%, rgba(217,70,239,0.18) 100%);
  border: 1px solid rgba(252,165,165,0.25);
  color: rgba(254,202,202,0.95);
}

/* ── Score badge ── */
.score-badge {
  background: rgba(255,255,255,0.07);
  backdrop-filter: blur(24px) brightness(1.15);
  -webkit-backdrop-filter: blur(24px) brightness(1.15);
  border: 1.5px solid rgba(255,255,255,0.13);
  border-radius: 20px;
  padding: 14px 20px;
  text-align: center;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 6px;
  color: rgba(255,255,255,0.92);
  box-shadow:
    0 1px 0 rgba(255,255,255,0.14) inset,
    0 8px 32px rgba(0,0,0,0.40);
  margin-top: 6px;
}
.score-sub {
  text-align: center;
  font-size: 0.68rem;
  color: rgba(255,255,255,0.30);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 4px;
}

/* ── Section header ── */
.sh {
  font-size: 0.68rem;
  font-weight: 600;
  color: rgba(255,255,255,0.35);
  text-transform: uppercase;
  letter-spacing: 0.10em;
  margin: 16px 0 8px 0;
}

/* ── Venue / stage tag ── */
.venue-tag {
  display: inline-block;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: 0.75rem;
  color: rgba(255,255,255,0.42);
  font-weight: 500;
  text-align: center;
  margin: 0 auto 4px auto;
}

/* ── Expander: nuclear fix for "keyboard_arrow_right" ligature ghost text ──
   Root cause: Streamlit injects the toggle icon as a <span> with
   font-family "Material Symbols Rounded". That font uses ligatures, so
   the text "keyboard_arrow_right" renders as an arrow icon when loaded —
   but shows as raw text before/if the web font fails.
   Fix: set font-size:0 on EVERYTHING inside <summary>, then restore
   font-size only on the <p> that holds the user label. The icon span
   collapses to invisible pixels; the label stays readable.          ── */

/* 1. Kill all text rendering inside summary */
.stExpander details > summary,
.stExpander details > summary * {
  font-size: 0 !important;
  line-height: 0 !important;
}

/* 2. Restore only the label paragraph */
.stExpander details > summary p {
  font-size: 0.88rem !important;
  line-height: 1.4 !important;
  font-weight: 500 !important;
  color: rgba(255,255,255,0.65) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif !important;
}

/* 3. Add our own rotating chevron */
.stExpander details > summary {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 12px 16px !important;
  cursor: pointer !important;
}
.stExpander details > summary::before {
  content: "›";
  font-size: 1rem !important;
  line-height: 1 !important;
  font-weight: 400;
  color: rgba(255,255,255,0.30);
  flex-shrink: 0;
  display: inline-block;
  transition: transform 0.18s ease;
}
.stExpander details[open] > summary::before {
  transform: rotate(90deg);
}

/* ── Player cards ── */
.player-ok {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 12px;
  padding: 8px 12px;
  margin: 3px 0;
  font-size: 0.83rem;
  color: rgba(255,255,255,0.80);
  box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset;
}
.player-inj {
  background: rgba(239,68,68,0.12);
  border: 1px solid rgba(239,68,68,0.25);
  border-radius: 12px;
  padding: 8px 12px;
  margin: 3px 0;
  font-size: 0.83rem;
  color: rgba(252,165,165,0.90);
}

/* ── Injury / ok banner ── */
.inj-banner {
  background: rgba(239,68,68,0.12);
  border: 1px solid rgba(239,68,68,0.22);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 12px;
  color: rgba(252,165,165,0.90);
  font-size: 0.85rem;
}
.ok-banner {
  background: rgba(34,197,94,0.10);
  border: 1px solid rgba(34,197,94,0.20);
  border-radius: 14px;
  padding: 10px 16px;
  margin-bottom: 12px;
  color: rgba(134,239,172,0.90);
  font-size: 0.85rem;
}

/* ── Fixture rows in sidebar ── */
.fix-row {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 6px 10px;
  margin: 3px 0;
  font-size: 0.75rem;
  color: rgba(255,255,255,0.55);
}

/* ── Streamlit native element tweaks ── */
/* Selectbox */
.stSelectbox > div > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
  color: rgba(255,255,255,0.85) !important;
  backdrop-filter: blur(20px);
}
/* Slider track — Apple blue accent */
.stSlider > div > div > div { background: var(--apple-blue) !important; }
.stSlider [role="slider"] {
  background: #fff !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.4) !important;
  border: none !important;
}
/* Toggle text */
.stToggle > label { color: rgba(255,255,255,0.60) !important; }
/* Checkbox */
.stCheckbox > label { color: rgba(255,255,255,0.65) !important; }

/* Primary button — Apple signature blue pill (apple.com CTA style) */
button[kind="primary"] {
  background: var(--apple-blue) !important;
  border: none !important;
  border-radius: 980px !important;          /* Apple's fully-rounded pill */
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  padding: 0.55rem 1.4rem !important;
  color: #fff !important;
  box-shadow: 0 1px 8px rgba(10,132,255,0.30) !important;
}
button[kind="primary"]:hover {
  background: var(--apple-blue-hover) !important;
  transform: scale(1.015) !important;
  box-shadow: 0 2px 16px rgba(10,132,255,0.45) !important;
}
button[kind="primary"]:active { transform: scale(0.98) !important; }

/* Secondary buttons — pill, glass */
button[kind="secondary"] {
  border-radius: 980px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  color: rgba(255,255,255,0.88) !important;
}
button[kind="secondary"]:hover {
  background: rgba(255,255,255,0.10) !important;
  border-color: rgba(255,255,255,0.20) !important;
}

/* Tabs — Apple segmented control */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 980px !important;
  padding: 4px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 980px !important;
  color: rgba(255,255,255,0.42) !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(255,255,255,0.11) !important;
  color: rgba(255,255,255,0.92) !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.10) inset, 0 2px 8px rgba(0,0,0,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* Expander */
.stExpander {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(20px) !important;
}
.stExpander summary { color: rgba(255,255,255,0.60) !important; font-weight: 500 !important; }

/* Native metric widget */
div[data-testid="metric-container"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 16px !important;
  padding: 14px !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.07) inset, 0 4px 16px rgba(0,0,0,0.25) !important;
}
div[data-testid="metric-container"] label {
  color: rgba(255,255,255,0.38) !important; font-size: 0.72rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: rgba(255,255,255,0.92) !important; font-weight: 700 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 16px !important; overflow: hidden; }
[data-testid="stDataFrame"] > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 16px !important;
}

/* Alert / info boxes */
.stAlert {
  border-radius: 14px !important;
  backdrop-filter: blur(16px) !important;
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: rgba(255,255,255,0.75) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 16px 0 !important; }

/* ── Lock to the dark design ──
   Remove the theme switcher so a Light selection can't break the custom dark
   CSS, and keep Streamlit's own chrome dark + readable either way. */
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stHeader"], header[data-testid="stHeader"],
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDialog"] > div, div[role="dialog"] {
  background: #16161f !important; color: rgba(255,255,255,0.90) !important; }
[data-testid="stDialog"] label, [data-testid="stDialog"] p, [data-testid="stDialog"] span,
div[role="dialog"] label, div[role="dialog"] p, div[role="dialog"] span {
  color: rgba(255,255,255,0.85) !important; }

/* ── Sidebar expand/collapse controls (Streamlit 1.50: stSidebarCollapseButton
     when open, stExpandSidebarButton when collapsed). Their Material Symbols
     icon ghosts as raw "keyboard_double_arrow_*" text when the font fails, so
     replace both with a simple hamburger menu icon. ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] { position: relative !important; }
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"] * { font-size: 0 !important; }
[data-testid="stSidebarCollapseButton"]::after,
[data-testid="stExpandSidebarButton"]::after {
  content: "☰"; position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  font-size: 1.1rem !important; line-height: 1;
  color: rgba(255,255,255,0.6); pointer-events: none; }
</style>
""", unsafe_allow_html=True)

# ── Access gate: admin (password) = full · visitor (skip) = read-only ─────────
from auth import require_access
access_level = require_access("visitor")
is_admin = access_level == "admin"

# ── Plotly dark glass theme ───────────────────────────────────────────────────
PLOT_BG   = "rgba(0,0,0,0)"
PLOT_GRID = "rgba(255,255,255,0.07)"
PLOT_FONT = "rgba(255,255,255,0.65)"
PLOT_H    = dict(t=10, b=48, l=44, r=10)

def _plot_base(height=260):
    return dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                  size=12),
        margin=PLOT_H, height=height,
        xaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.06)",
                   tickcolor="rgba(0,0,0,0)", tickfont=dict(color=PLOT_FONT)),
        yaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.06)",
                   tickcolor="rgba(0,0,0,0)", tickfont=dict(color=PLOT_FONT)),
    )

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading live data & WC 2026 results…"):
        df_raw, elo_by_name, wc2026_results, wc_team_stats = build_dataset()
        df = engineer_features(df_raw, elo_by_name, wc_team_stats)
        predictor = PoissonMatchPredictor()
        predictor.fit(df)
        all_elos  = np.array(list(elo_by_name.values()))
        all_teams = sorted(set(df_raw["home_team"].tolist() + df_raw["away_team"].tolist()))
        import datetime
        loaded_at = datetime.datetime.now().strftime("%d %b %Y %H:%M")
        return (predictor, df, elo_by_name, all_teams,
                float(all_elos.min()), float(all_elos.max()),
                wc2026_results, loaded_at, wc_team_stats)


@st.cache_data(ttl=300, show_spinner=False)
def load_player_stats():
    from fifa_wc2026_predictor import load_player_stats_cached
    return load_player_stats_cached()

@st.cache_data(ttl=300, show_spinner=False)
def get_squad_cached(team, include_squad):
    from fifa_wc2026_predictor import fetch_squad, manual_injury_adjustment
    if not include_squad:
        return [], 1.0, 1.0, []
    squad = fetch_squad(team)
    return squad, 1.0, 1.0, []   # base — injuries applied after manual input


def apply_manual_injuries(squad, injured_names):
    from fifa_wc2026_predictor import manual_injury_adjustment
    if not injured_names:
        return 1.0, 1.0, []
    return manual_injury_adjustment(squad, injured_names)

@st.cache_data(ttl=120, show_spinner=False)
def get_fixtures():
    return fetch_wc2026_fixtures()

@st.cache_data(ttl=180, show_spinner=False)
def get_odds_cached(home, away, use_market):
    """Live betting market for a matchup (cached 3 min). None if no market."""
    if not use_market:
        return None
    return fetch_match_odds(home, away)

@st.cache_data(ttl=300, show_spinner=False)
def get_polymarket_odds_cached():
    """All Polymarket WC 2026 winner odds (cached 5 min)."""
    return fetch_polymarket_wc_odds_all()

@st.cache_data(ttl=300, show_spinner=False)
def get_polymarket_match_cached(home, away):
    """Polymarket per-match odds for this fixture (cached 5 min). None if absent."""
    return fetch_polymarket_match_odds(home, away)

@st.cache_data(ttl=300, show_spinner=False)
def get_polymarket_match_by_url_cached(url, home, away):
    """Polymarket per-match odds from a user-supplied URL (cached 5 min)."""
    return fetch_polymarket_match_odds_from_url(url, home, away)

# ── MC helper ─────────────────────────────────────────────────────────────────
def _mc(lh, la, hu, au, n, seed=42):
    rng = np.random.default_rng(seed)
    gh  = rng.poisson(np.maximum(lh + rng.normal(0, hu*lh, n), 0.05))
    ga  = rng.poisson(np.maximum(la + rng.normal(0, au*la, n), 0.05))
    return gh, ga

# ── Chart functions (light glass Plotly theme) ────────────────────────────────
def chart_donut(p_hw, p_d, p_aw, home, away):
    fig = go.Figure(go.Pie(
        labels=[f"{home} Win", "Draw", f"{away} Win"],
        values=[p_hw*100, p_d*100, p_aw*100],
        hole=0.65,
        marker=dict(
            colors=["rgba(99,102,241,0.80)", "rgba(156,163,175,0.60)", "rgba(239,68,68,0.75)"],
            line=dict(color="rgba(255,255,255,0.90)", width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=12, color="rgba(255,255,255,0.65)"),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    layout = _plot_base(height=240)
    layout.update(showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
    fig.update_layout(**layout)
    return fig

def chart_goal_dist(lh, la, hu, au, home, away, n):
    gh, ga = _mc(lh, la, hu, au, n)
    max_g  = max(int(gh.max()), int(ga.max()), 6)
    bins   = np.arange(0, max_g+2)
    fig    = go.Figure()
    fig.add_trace(go.Bar(x=bins[:-1], y=np.histogram(gh,bins=bins)[0]/n*100,
                         name=home,
                         marker=dict(color="rgba(99,102,241,0.70)",
                                     line=dict(color="rgba(255,255,255,0.6)",width=1)),
                         opacity=0.85))
    fig.add_trace(go.Bar(x=bins[:-1], y=np.histogram(ga,bins=bins)[0]/n*100,
                         name=away,
                         marker=dict(color="rgba(239,68,68,0.65)",
                                     line=dict(color="rgba(255,255,255,0.6)",width=1)),
                         opacity=0.85))
    layout = _plot_base(260)
    layout.update(
        barmode="overlay",
        xaxis=dict(**layout["xaxis"], title="Goals scored", tickmode="linear"),
        yaxis=dict(**layout["yaxis"], title="Probability (%)"),
        legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="rgba(0,0,0,0.06)",
                    borderwidth=1, font=dict(color=PLOT_FONT)),
    )
    fig.update_layout(**layout)
    return fig

def chart_goal_diff(lh, la, hu, au, home, away, ci_lo, ci_hi, n):
    gh, ga = _mc(lh, la, hu, au, n)
    gd = gh.astype(int) - ga.astype(int)
    lo, hi = int(gd.min()), int(gd.max())
    bins   = np.arange(lo, hi+2)
    counts, _ = np.histogram(gd, bins=bins)
    xs = bins[:-1];  probs = counts/n*100
    cols = [
        "rgba(99,102,241,0.75)"  if x > 0 else
        ("rgba(239,68,68,0.70)"  if x < 0 else
         "rgba(156,163,175,0.65)")
        for x in xs
    ]
    fig = go.Figure()
    fig.add_vrect(
        x0=ci_lo, x1=ci_hi,
        fillcolor="rgba(99,102,241,0.07)",
        line=dict(color="rgba(99,102,241,0.20)", width=1, dash="dot"),
        annotation_text="95% CI",
        annotation_font=dict(color="rgba(99,102,241,0.70)", size=11),
        annotation_position="top left",
    )
    fig.add_trace(go.Bar(
        x=xs, y=probs,
        marker=dict(color=cols, line=dict(color="rgba(255,255,255,0.5)", width=1)),
        hovertemplate="GD=%{x}: %{y:.1f}%<extra></extra>",
    ))
    layout = _plot_base(260)
    layout.update(
        showlegend=False,
        xaxis=dict(**layout["xaxis"],
                   title=f"← {away} wins  |  Draw  |  {home} wins →",
                   zeroline=True,
                   zerolinecolor="rgba(107,114,128,0.30)",
                   zerolinewidth=1.5),
        yaxis=dict(**layout["yaxis"], title="Probability (%)"),
    )
    fig.update_layout(**layout)
    return fig

def chart_heatmap(lh, la, hu, au, home, away, n):
    gh, ga = _mc(lh, la, hu, au, n)
    max_g = 6
    mat   = np.zeros((max_g+1, max_g+1))
    for h, a in zip(gh, ga):
        if h <= max_g and a <= max_g:
            mat[h, a] += 1
    mat  = mat / n * 100
    text = [[f"{mat[r,c]:.1f}%" if mat[r,c] > 0.4 else ""
             for c in range(max_g+1)] for r in range(max_g+1)]
    colorscale = [
        [0.0,  "rgba(255,255,255,0.0)"],
        [0.15, "rgba(199,210,254,0.5)"],
        [0.4,  "rgba(129,140,248,0.65)"],
        [0.7,  "rgba(99,102,241,0.80)"],
        [1.0,  "rgba(67,56,202,0.92)"],
    ]
    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[str(i) for i in range(max_g+1)],
        y=[str(i) for i in range(max_g+1)],
        colorscale=colorscale,
        hovertemplate=f"{home} %{{y}} – %{{x}} {away}: %{{z:.2f}}%<extra></extra>",
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="rgba(255,255,255,0.80)"),
        colorbar=dict(
            tickfont=dict(color=PLOT_FONT, size=10),
            title=dict(text="%", font=dict(color=PLOT_FONT, size=11)),
            bgcolor="rgba(255,255,255,0.04)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            outlinewidth=0,
        ),
    ))
    layout = _plot_base(320)
    layout.update(
        xaxis=dict(**layout["xaxis"], title=f"{away} goals"),
        yaxis=dict(**layout["yaxis"], title=f"{home} goals", autorange="reversed"),
        margin=dict(t=10, b=52, l=52, r=10),
    )
    fig.update_layout(**layout)
    return fig

def chart_elo_bar(home, away, h_elo, a_elo):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[h_elo], y=[home], orientation="h",
        marker=dict(color="rgba(99,102,241,0.70)",
                    line=dict(color="rgba(255,255,255,0.6)", width=1)),
        hovertemplate=f"{home}: {h_elo:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[a_elo], y=[away], orientation="h",
        marker=dict(color="rgba(239,68,68,0.65)",
                    line=dict(color="rgba(255,255,255,0.6)", width=1)),
        hovertemplate=f"{away}: {a_elo:.0f}<extra></extra>",
    ))
    layout = _plot_base(110)
    layout.update(
        barmode="overlay", showlegend=False,
        xaxis=dict(**layout["xaxis"], title="World Football ELO", range=[1200, 2300]),
        margin=dict(t=8, b=40, l=8, r=8),
    )
    fig.update_layout(**layout)
    return fig

# ── Squad panel ───────────────────────────────────────────────────────────────
def _cb_save_injuries(team_name, names):
    """Save the ticked players as this team's injured set (on_click callback)."""
    chosen = {n for n in names if st.session_state.get(f"injchk::{team_name}::{n}")}
    st.session_state[f"injured::{team_name}"] = chosen
    st.session_state["_apply_injuries"] = True


def _cb_clear_injuries(team_name, names):
    """Untick every player and clear the injured set (on_click callback).
    Runs before widgets re-instantiate, so modifying the checkbox keys is safe."""
    for n in names:
        st.session_state[f"injchk::{team_name}::{n}"] = False
    st.session_state[f"injured::{team_name}"] = set()
    st.session_state["_apply_injuries"] = True


def squad_panel(players, team_name, atk_adj, def_adj, form: dict = None,
                player_stats_df=None, editable: bool = True):
    if not players:
        st.caption(f"No ESPN squad data available for {team_name}.")
        return

    form = form or {}
    saved = st.session_state.get(f"injured::{team_name}", set())

    # ── Tournament form banner ────────────────────────────────────────────────
    gp = form.get("games_played", 0)
    if gp > 0:
        gpg  = form.get("goals_pg", 0)
        concg = form.get("goals_conceded_pg", 0)
        spg  = form.get("shots_pg", 0)
        sotpg = form.get("shots_on_target_pg", 0)
        st.markdown(
            f"<div style='background:rgba(99,179,237,0.08);border:1px solid rgba(99,179,237,0.2);"
            f"border-radius:10px;padding:8px 12px;font-size:.78rem;color:#90cdf4;margin:4px 0 8px;'>"
            f"📊 <b>WC 2026 form</b> &nbsp;·&nbsp; {gp} game{'s' if gp!=1 else ''} &nbsp;|&nbsp; "
            f"Goals: <b>{gpg:.1f}/g</b> &nbsp;·&nbsp; "
            f"Shots: <b>{spg:.1f}/g</b> &nbsp;·&nbsp; "
            f"SoT: <b>{sotpg:.1f}/g</b> &nbsp;·&nbsp; "
            f"Conceded: <b>{concg:.1f}/g</b></div>",
            unsafe_allow_html=True,
        )
        # Top scorers
        scorers = form.get("top_scorers", [])
        if scorers:
            scorer_html = " &nbsp;·&nbsp; ".join(
                f"<b>{s['player_name']}</b> {int(s['goals'])}⚽" for s in scorers
            )
            st.markdown(
                f"<div style='font-size:.76rem;color:#f6e05e;margin:-4px 0 8px;'>"
                f"🏅 {scorer_html}</div>",
                unsafe_allow_html=True,
            )

    # ── Injury / suspension banner (reflects the last SAVED selection) ────────
    if atk_adj < 1.0 or def_adj < 1.0 or saved:
        pen_a = round((1 - atk_adj) * 100, 1)
        pen_d = round((1 - def_adj) * 100, 1)
        st.markdown(
            f"<div class='inj-banner'>⚠️ <b>{len(saved)} unavailable</b> — "
            f"Attack λ −{pen_a}%&nbsp;&nbsp;·&nbsp;&nbsp;Defence λ −{pen_d}%</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='ok-banner'>✅ <b>Full squad available</b> — tick anyone who is out below</div>",
            unsafe_allow_html=True,
        )

    # ── Editing controls (admin only) ─────────────────────────────────────────
    if editable:
        _names = [p["name"] for p in players]
        st.markdown(
            "<div style='font-size:.76rem;color:rgba(255,255,255,0.42);margin:2px 0 6px;'>"
            "Tick players who are injured or unavailable, then press "
            "<b>Save &amp; re-run</b> to apply them to the prediction.</div>",
            unsafe_allow_html=True,
        )
        _sc1, _sc2, _ = st.columns([1.4, 1, 4])
        with _sc1:
            st.button("💾 Save & re-run", key=f"savechk::{team_name}",
                      use_container_width=True, type="primary",
                      on_click=_cb_save_injuries, args=(team_name, _names))
        with _sc2:
            st.button("Clear", key=f"clearchk::{team_name}",
                      use_container_width=True,
                      on_click=_cb_clear_injuries, args=(team_name, _names))

    # Build per-player WC 2026 stat lookup
    player_stat_map: dict = {}
    if player_stats_df is not None and not player_stats_df.empty:
        team_ps = player_stats_df[player_stats_df["team"] == team_name]
        for name, grp in team_ps.groupby("player_name"):
            player_stat_map[name] = {
                "goals":   int(grp["totalGoals"].sum()),
                "assists": int(grp["goalAssists"].sum()),
                "shots":   int(grp["totalShots"].sum()),
                "yellows": int(grp["yellowCards"].sum()),
                "reds":    int(grp["redCards"].sum()),
            }

    for pos_group in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        group = [p for p in players if pos_group in p["position"]]
        if not group:
            continue
        st.markdown(f"<div class='sh'>{pos_group}s</div>", unsafe_allow_html=True)
        for p in group:
            ck = f"injchk::{team_name}::{p['name']}"
            if ck not in st.session_state:
                # Seed once from the saved set (which itself starts from ESPN injuries)
                st.session_state[ck] = p["name"] in saved
            if editable:
                c1, c2 = st.columns([1, 18])
                with c1:
                    st.checkbox("unavailable", key=ck, label_visibility="collapsed")
                cell = c2
                checked = st.session_state[ck]
            else:                                    # visitor: static, no checkbox
                cell = st.container()
                checked = p["name"] in saved
            with cell:
                ps = player_stat_map.get(p["name"], {})
                badges = []
                if ps.get("goals"):   badges.append(f"⚽{ps['goals']}")
                if ps.get("assists"): badges.append(f"🎯{ps['assists']}")
                if ps.get("shots"):   badges.append(f"🔫{ps['shots']}")
                if ps.get("yellows"): badges.append(f"🟨{ps['yellows']}")
                if ps.get("reds"):    badges.append(f"🟥{ps['reds']}")
                badge_str = " ".join(badges)

                if checked:
                    detail = p.get("injury_detail") if p.get("is_injured") else "Marked unavailable"
                    st.markdown(
                        f"<div class='player-inj'>⚠️ <b>{p['name']}</b>"
                        f"{'  ' + badge_str if badge_str else ''}"
                        f"<br><span style='font-size:.75rem;opacity:.7;'>{detail}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    age = f"  ·  {p['age']}y" if p.get("age") else ""
                    inj_hint = (" <span style='font-size:.7rem;color:#f6ad55;'>"
                                "(ESPN: injury concern)</span>") if p.get("is_injured") else ""
                    st.markdown(
                        f"<div class='player-ok'>● <b>{p['name']}</b>"
                        f"<span style='font-size:.75rem;color:rgba(255,255,255,0.35);'>{age}</span>"
                        f"{badge_str and '  ' + badge_str}{inj_hint}</div>",
                        unsafe_allow_html=True,
                    )

    # Save / Clear are handled by their on_click callbacks (which run before the
    # checkbox widgets re-instantiate, so they can safely modify the keys).

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.92);margin-bottom:2px;'>"
        "⚽ WC 2026 Predictor</p>"
        "<p style='font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:0;'>FIFA World Cup 2026</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    (predictor, df, elo_by_name, all_teams,
     elo_min, elo_max, wc2026_results, loaded_at, wc_team_stats) = load_model()

    # Build WC-qualified team list from live ESPN schedule
    with st.spinner(""):
        _fixtures = get_fixtures()
    _skip = ("Winner", "Place", "Third Place", "Round of")
    if not _fixtures.empty:
        _raw = set(_fixtures["home_team"].tolist() + _fixtures["away_team"].tolist())
        team_options = sorted(t for t in _raw if not any(k in t for k in _skip))
    else:
        team_options = sorted(ESPN_TEAM_IDS.keys())

    # ── Default to the next unplayed fixture ─────────────────────────────────
    _next_home, _next_away = "France", "Morocco"   # fallback
    if not _fixtures.empty:
        _upcoming = _fixtures[~_fixtures["completed"]].copy()
        # Also exclude placeholder team names
        _upcoming = _upcoming[
            ~_upcoming["home_team"].str.contains("|".join(_skip), na=False) &
            ~_upcoming["away_team"].str.contains("|".join(_skip), na=False)
        ]
        if not _upcoming.empty:
            _next = _upcoming.iloc[0]
            _next_home = _next["home_team"] if _next["home_team"] in team_options else _next_home
            _next_away = _next["away_team"] if _next["away_team"] in team_options else _next_away

    _default_h = team_options.index(_next_home) if _next_home in team_options else 0
    _default_a = team_options.index(_next_away) if _next_away in team_options else min(1, len(team_options)-1)

    if is_admin:
        st.markdown("<div class='sh'>Match Setup</div>", unsafe_allow_html=True)
        home_team = st.selectbox("Team A", team_options, index=_default_h,
                                  label_visibility="visible")
        away_team = st.selectbox("Team B", team_options, index=_default_a,
                                  label_visibility="visible")
    else:
        # Visitor: locked to the next upcoming fixture — read-only preview.
        home_team, away_team = _next_home, _next_away
        st.markdown("<div class='sh'>Next match</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border:1px solid "
            f"rgba(255,255,255,0.10);border-radius:12px;padding:12px 14px;"
            f"font-size:0.95rem;color:rgba(255,255,255,0.88);text-align:center;'>"
            f"<b>{home_team}</b> &nbsp;vs&nbsp; <b>{away_team}</b></div>"
            "<div style='font-size:0.74rem;color:rgba(255,255,255,0.45);margin-top:8px;'>"
            "👁 Read-only preview of the next fixture. Log out and enter the admin "
            "password to simulate any matchup.</div>",
            unsafe_allow_html=True,
        )

    # Host auto-detection (both modes)
    home_is_host = home_team in WC2026_HOSTS
    away_is_host = away_team in WC2026_HOSTS

    if is_admin:
        host_labels = [t for t in [home_team, away_team] if t in WC2026_HOSTS]
        if host_labels:
            st.markdown(
                f"<div style='background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.30);"
                f"border-radius:12px;padding:8px 12px;font-size:.78rem;color:#92400e;margin:6px 0;'>"
                f"🏠 <b>Host nation{'s' if len(host_labels)>1 else ''}:</b> {', '.join(host_labels)}</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("<div class='sh'>Context</div>", unsafe_allow_html=True)
        stage   = st.selectbox("Stage", ["Group Stage","Round of 32","Round of 16",
                                          "Quarter-Final","Semi-Final","Final"])
        neutral = st.toggle("Neutral Venue", value=True)

        st.divider()
        st.markdown("<div class='sh'>Options</div>", unsafe_allow_html=True)
        include_squad = st.toggle("Squad & injury data", value=True,
                                   help="WC 2026 squads from ESPN — add injured players manually below")

        if include_squad:
            _h_squad_pre, _, _, _ = get_squad_cached(home_team, True)
            _a_squad_pre, _, _, _ = get_squad_cached(away_team, True)
            for _team, _sq in ((home_team, _h_squad_pre), (away_team, _a_squad_pre)):
                _key = f"injured::{_team}"
                if _key not in st.session_state:
                    st.session_state[_key] = {p["name"] for p in _sq if p.get("is_injured")}
            h_injury_names = sorted(st.session_state.get(f"injured::{home_team}", set()))
            a_injury_names = sorted(st.session_state.get(f"injured::{away_team}", set()))
            st.markdown(
                "<div style='font-size:.75rem;color:rgba(255,255,255,0.45);margin:6px 0 2px;'>"
                f"⚠️ Unavailable &nbsp;·&nbsp; {home_team}: <b>{len(h_injury_names)}</b>"
                f" &nbsp;·&nbsp; {away_team}: <b>{len(a_injury_names)}</b><br>"
                "<span style='opacity:.85;'>Tick players in the squad list below the "
                "results, then <b>Save</b>.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            h_injury_names, a_injury_names = [], []

        use_market = st.toggle("Betting market odds", value=True,
                                help="Blend DraftKings market odds into the prediction "
                                     "(only available for scheduled WC 2026 fixtures)")
        market_weight = st.slider(
            "Market influence", 0.0, 1.0, 0.50, step=0.05,
            help="0 = pure statistical model · 1 = pure betting market · "
                 "0.5 = equal blend",
            disabled=not use_market,
        )
        poly_match_url = st.text_input(
            "Polymarket match URL (optional)",
            key="poly_match_url",
            placeholder="https://polymarket.com/event/…",
            help="Paste the Polymarket page for this exact match to force its odds "
                 "if auto-detection misses it. Leave blank to auto-detect.",
            disabled=not use_market,
        )
        n_sims  = st.slider("Monte Carlo draws", 1_000, 50_000, 10_000, step=1_000)
        run_btn = st.button("Run Simulation ›", use_container_width=True, type="primary")
    else:
        # Visitor: fixed sensible defaults; no controls. Auto-computes the
        # prediction for the locked fixture (and refreshes if a stale result
        # from a prior admin session is in memory).
        stage = "Group Stage"
        neutral = True
        include_squad = True
        for _team in (home_team, away_team):
            _sq, _, _, _ = get_squad_cached(_team, True)
            _key = f"injured::{_team}"
            if _key not in st.session_state:
                st.session_state[_key] = {p["name"] for p in _sq if p.get("is_injured")}
        h_injury_names = sorted(st.session_state.get(f"injured::{home_team}", set()))
        a_injury_names = sorted(st.session_state.get(f"injured::{away_team}", set()))
        use_market = True
        market_weight = 0.50
        poly_match_url = ""
        n_sims = 10_000
        run_btn = (st.session_state.get("home") != home_team
                   or st.session_state.get("away") != away_team)

    # Refresh / rebuild buttons — admin only (they mutate the shared cache)
    from fifa_wc2026_predictor import HIST_PARQUET, CACHE_DIR
    if is_admin:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🔄  Check for new results", use_container_width=True,
                     help="Fetches any new WC 2026 results from ESPN and appends to local cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        if st.button("🗑️  Rebuild full cache", use_container_width=True,
                     help="Deletes historical parquet and re-downloads everything on next load"):
            if HIST_PARQUET.exists():
                HIST_PARQUET.unlink()
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared — reloading…")
            st.rerun()

    # Kaggle setup hint shown only when cache is missing (admin only)
    if is_admin and not HIST_PARQUET.exists():
        st.info(
            "**First run:** for the full 47,000-match dataset, place the Kaggle CSV at:\n\n"
            f"`{CACHE_DIR}/results.csv`\n\n"
            "Download from [kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017]"
            "(https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)  "
            "or configure `~/.kaggle/kaggle.json` for auto-download.\n\n"
            "Without it the app uses a smaller WC-only fallback dataset.",
            icon="📂",
        )

    # ── WC 2026 results incorporated into the model ───────────────────────────
    st.divider()
    n_res = len(wc2026_results) if not wc2026_results.empty else 0
    st.markdown(
        f"<div class='sh'>WC 2026 Results in Model "
        f"<span style='color:rgba(134,239,172,0.80);font-size:.7rem;'>({n_res})</span></div>",
        unsafe_allow_html=True,
    )
    if n_res:
        for _, row in wc2026_results.sort_values("date", ascending=False).iterrows():
            ht, at = row["home_team"], row["away_team"]
            hs, as_ = int(row["home_score"]), int(row["away_score"])
            if hs > as_:
                result_icon = "🔵"
            elif hs < as_:
                result_icon = "🔴"
            else:
                result_icon = "⚪"
            st.markdown(
                f"<div class='fix-row'>{result_icon} <b>{ht}</b> "
                f"<span style='color:rgba(255,255,255,0.90);font-weight:700;'>"
                f"{hs}–{as_}</span> "
                f"<b>{at}</b>"
                f"<span style='float:right;color:rgba(255,255,255,0.30);font-size:.68rem;'>"
                f"{str(row['date'])[:10]}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<p style='font-size:.68rem;color:rgba(255,255,255,0.30);margin:6px 0 0;'>"
            f"Model last updated: {loaded_at}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='fix-row' style='color:rgba(255,255,255,0.30);'>"
            "No completed matches yet.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("<div class='sh'>Upcoming Fixtures</div>", unsafe_allow_html=True)
    if not _fixtures.empty:
        for _, row in _fixtures[~_fixtures["completed"]].head(5).iterrows():
            st.markdown(
                f"<div class='fix-row'>📅 <b>{row['date']}</b>"
                f"<br>{row['home_team']} vs {row['away_team']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<p style='font-size:.65rem;color:rgba(255,255,255,0.20);margin-top:12px;'>"
        "ESPN · eloratings.net · jfjelstul/worldcup · fixturedownload.com<br>"
        "No API key required.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center;font-size:1.6rem;font-weight:700;"
    "letter-spacing:-0.02em;color:rgba(255,255,255,0.92);margin-bottom:2px;'>"
    "FIFA World Cup 2026</h2>"
    "<p style='text-align:center;color:rgba(255,255,255,0.35);font-size:.82rem;margin-top:0;'>"
    "Match Outcome Predictor</p>",
    unsafe_allow_html=True,
)

if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

# ── Simulation ────────────────────────────────────────────────────────────────
if ("result" not in st.session_state or run_btn
        or st.session_state.pop("_apply_injuries", False)):
    with st.spinner("Fetching squads & player stats…"):
        h_squad, _, _, _ = get_squad_cached(home_team, include_squad)
        a_squad, _, _, _ = get_squad_cached(away_team, include_squad)
        player_stats_df   = load_player_stats()

    h_atk_adj, h_def_adj, h_injured = apply_manual_injuries(h_squad, h_injury_names)
    a_atk_adj, a_def_adj, a_injured = apply_manual_injuries(a_squad, a_injury_names)

    # Tournament form from live WC 2026 player stats
    h_form = compute_team_form(home_team, player_stats_df)
    a_form = compute_team_form(away_team, player_stats_df)
    h_form_atk, h_form_def = tournament_form_lambda_adjustment(h_form)
    a_form_atk, a_form_def = tournament_form_lambda_adjustment(a_form)

    # Auto-add suspended players (red card) to injury list for display
    for name in h_form.get("suspended", []):
        if not any(p["name"] == name for p in h_injured):
            h_injured.append({"name": name, "position": "Unknown",
                               "status": "Suspended", "status_type": "out",
                               "is_injured": True, "injury_detail": "Red card suspension",
                               "jersey": "", "age": None})
    for name in a_form.get("suspended", []):
        if not any(p["name"] == name for p in a_injured):
            a_injured.append({"name": name, "position": "Unknown",
                               "status": "Suspended", "status_type": "out",
                               "is_injured": True, "injury_detail": "Red card suspension",
                               "jersey": "", "age": None})

    def _profile(team, is_host, neu):
        return build_team_profile(df, elo_by_name, team, is_host=is_host, neutral=neu,
                                   elo_min=elo_min, elo_max=elo_max, include_squad=False,
                                   wc_team_stats=wc_team_stats)

    hf = _profile(home_team, home_is_host, neutral)
    af = _profile(away_team, away_is_host, neutral)
    lh, la = predictor.predict_lambdas(hf, af)

    h_atk = h_atk_adj if include_squad else 1.0
    h_def = h_def_adj if include_squad else 1.0
    a_atk = a_atk_adj if include_squad else 1.0
    a_def = a_def_adj if include_squad else 1.0
    model_lh = max(lh * h_atk * h_form_atk / max(a_def * a_form_def, 0.75), 0.05)
    model_la = max(la * a_atk * a_form_atk / max(h_def * h_form_def, 0.75), 0.05)

    # ── Betting market blend ──────────────────────────────────────────────────
    with st.spinner("Fetching betting market…"):
        odds = get_odds_cached(home_team, away_team, use_market)

    if odds and use_market:
        final_lh, final_la = blend_lambdas(
            model_lh, model_la,
            odds["market_lambda_home"], odds["market_lambda_away"],
            market_weight,
        )
        blend_applied = True
    else:
        final_lh, final_la = model_lh, model_la
        blend_applied = False

    result = simulate_match(
        final_lh, final_la,
        home_uncertainty = hf["uncertainty"], away_uncertainty = af["uncertainty"],
        n_simulations    = n_sims,
    )
    model_result = simulate_match(
        model_lh, model_la,
        home_uncertainty = hf["uncertainty"], away_uncertainty = af["uncertainty"],
        n_simulations    = n_sims,
    )

    st.session_state.update(dict(
        result=result, model_result=model_result,
        home=home_team, away=away_team, hf=hf, af=af,
        model_lh=model_lh, model_la=model_la,
        final_lh=final_lh, final_la=final_la,
        odds=odds, blend_applied=blend_applied, market_weight=market_weight,
        h_squad=h_squad, h_atk_adj=h_atk_adj, h_def_adj=h_def_adj, h_injured=h_injured,
        a_squad=a_squad, a_atk_adj=a_atk_adj, a_def_adj=a_def_adj, a_injured=a_injured,
        h_form=h_form, a_form=a_form,
    ))

res          = st.session_state.result
model_res    = st.session_state.model_result
home         = st.session_state.home
away         = st.session_state.away
hf           = st.session_state.hf
af           = st.session_state.af
model_lh     = st.session_state.model_lh
model_la     = st.session_state.model_la
final_lh     = st.session_state.final_lh
final_la     = st.session_state.final_la
odds         = st.session_state.odds
blend_applied = st.session_state.blend_applied
mkt_weight   = st.session_state.market_weight
h_squad   = st.session_state.h_squad
a_squad   = st.session_state.a_squad
h_atk_adj = st.session_state.h_atk_adj
a_atk_adj = st.session_state.a_atk_adj
h_def_adj = st.session_state.h_def_adj
a_def_adj = st.session_state.a_def_adj
h_injured = st.session_state.h_injured
a_injured = st.session_state.a_injured
h_form    = st.session_state.get("h_form", {})
a_form    = st.session_state.get("a_form", {})
_player_stats_df = load_player_stats()
h_elo     = elo_by_name.get(home)
a_elo     = elo_by_name.get(away)

# ── Match header card ─────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([5, 3, 5])
with c1:
    inj = f" &nbsp;⚠️ {len(h_injured)}" if h_injured else ""
    host_tag = " &nbsp;🏠" if home_is_host else ""
    st.markdown(
        f"<div class='team-pill home-pill'>{home}{host_tag}{inj}</div>",
        unsafe_allow_html=True,
    )
    if h_elo:
        st.markdown(
            f"<p style='text-align:center;font-size:.78rem;color:#6366f1;"
            f"font-weight:600;margin:0;'>ELO {h_elo:.0f}</p>",
            unsafe_allow_html=True,
        )

with c2:
    st.markdown(
        f"<div class='score-badge'>{res['most_likely_score'][0]}–{res['most_likely_score'][1]}</div>"
        f"<div class='score-sub'>Most likely</div>",
        unsafe_allow_html=True,
    )

with c3:
    inj = f" &nbsp;⚠️ {len(a_injured)}" if a_injured else ""
    host_tag = " &nbsp;🏠" if away_is_host else ""
    st.markdown(
        f"<div class='team-pill away-pill'>{away}{host_tag}{inj}</div>",
        unsafe_allow_html=True,
    )
    if a_elo:
        st.markdown(
            f"<p style='text-align:center;font-size:.78rem;color:#ef4444;"
            f"font-weight:600;margin:0;'>ELO {a_elo:.0f}</p>",
            unsafe_allow_html=True,
        )

# Stage / venue tag
venue_label = "Neutral venue"
if not neutral:
    if home_is_host and away_is_host: venue_label = "🏠 Both are WC 2026 hosts"
    elif home_is_host:                venue_label = f"🏠 {home} host nation"
    elif away_is_host:                venue_label = f"🏠 {away} host nation"
    else:                             venue_label = "Home advantage"

blend_badge = ""
if blend_applied and odds:
    blend_badge = (
        f"<span class='venue-tag' style='background:rgba(251,191,36,0.12);"
        f"border-color:rgba(251,191,36,0.30);color:rgba(251,191,36,0.85);"
        f"margin-left:6px;'>💰 Market blended ({int(mkt_weight*100)}%)</span>"
    )
st.markdown(
    f"<div style='text-align:center;margin:10px 0 4px;'>"
    f"<span class='venue-tag'>{stage} &nbsp;·&nbsp; {venue_label}</span>{blend_badge}</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Metric cards row ──────────────────────────────────────────────────────────
gd = res["mean_goal_diff"]
m1, m2, m3, m4, m5 = st.columns(5)
cards = [
    (m1, f"{res['prob_home_win']*100:.1f}%", f"{home} Win",    "#6366f1"),
    (m2, f"{res['prob_draw']*100:.1f}%",      "Draw",           "rgba(180,180,200,0.70)"),
    (m3, f"{res['prob_away_win']*100:.1f}%",  f"{away} Win",   "#ef4444"),
    (m4, (f"+{gd:.2f}" if gd>=0 else f"{gd:.2f}"), "Goal Diff",
     "#6366f1" if gd>=0 else "#ef4444"),
    (m5, f"±{res['margin_of_error']:.2f}", "Margin of Error",  "#059669"),
]
for col, val, lbl, color in cards:
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='val' style='color:{color};'>{val}</div>"
            f"<div class='lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Model vs Market vs Blended comparison ─────────────────────────────────────
_poly_url = (st.session_state.get("poly_match_url") or "").strip()
poly_match = None
poly_url_failed = False
if use_market:
    if _poly_url:
        poly_match = get_polymarket_match_by_url_cached(_poly_url, home, away)
        poly_url_failed = poly_match is None
    else:
        poly_match = get_polymarket_match_cached(home, away)

if (blend_applied and odds) or poly_match:
    st.divider()
    _providers = []
    if blend_applied and odds:
        _providers.append(odds["provider"])
    if poly_match:
        _providers.append("Polymarket")
    _blend_note = (f" &nbsp;·&nbsp; blend {int((1-mkt_weight)*100)}/{int(mkt_weight*100)} "
                   f"(model/market)" if blend_applied and odds else "")
    st.markdown(
        f"<div class='sh'>Model vs Market &nbsp;·&nbsp; "
        f"{' + '.join(_providers)}{_blend_note}</div>",
        unsafe_allow_html=True,
    )

    def _row(label, p_h, p_d, p_a, lh, la, accent):
        return (
            f"<tr>"
            f"<td style='font-weight:600;color:{accent};'>{label}</td>"
            f"<td style='text-align:center;'>{p_h*100:.1f}%</td>"
            f"<td style='text-align:center;'>{p_d*100:.1f}%</td>"
            f"<td style='text-align:center;'>{p_a*100:.1f}%</td>"
            f"<td style='text-align:center;color:rgba(255,255,255,0.45);'>"
            f"{lh:.2f} – {la:.2f}</td>"
            f"</tr>"
        )

    table = (
        "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
        "<thead><tr style='border-bottom:1px solid rgba(255,255,255,0.10);'>"
        "<th style='text-align:left;padding:6px 8px;color:rgba(255,255,255,0.40);"
        "font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;'>Source</th>"
        f"<th style='padding:6px 8px;color:rgba(199,210,254,0.70);font-size:0.7rem;'>{home}</th>"
        "<th style='padding:6px 8px;color:rgba(180,180,200,0.70);font-size:0.7rem;'>Draw</th>"
        f"<th style='padding:6px 8px;color:rgba(252,165,165,0.70);font-size:0.7rem;'>{away}</th>"
        "<th style='padding:6px 8px;color:rgba(255,255,255,0.40);font-size:0.7rem;'>λ (xG)</th>"
        "</tr></thead><tbody>"
    )
    # Model row (always)
    table += _row("📊 Model",
                  model_res["prob_home_win"], model_res["prob_draw"], model_res["prob_away_win"],
                  model_lh, model_la, "rgba(255,255,255,0.75)")
    # DraftKings market row (when available)
    if blend_applied and odds:
        table += _row(f"💰 {odds['provider']}",
                      odds["market_prob_home"], odds["market_prob_draw"], odds["market_prob_away"],
                      odds["market_lambda_home"], odds["market_lambda_away"], "rgba(251,191,36,0.85)")
    # Polymarket per-match row (when a market exists for this fixture)
    if poly_match:
        table += _row("🔮 Polymarket",
                      poly_match["market_prob_home"], poly_match["market_prob_draw"],
                      poly_match["market_prob_away"], poly_match["market_lambda_home"],
                      poly_match["market_lambda_away"], "rgba(94,234,212,0.90)")
    # Blended (final) row — only when DraftKings is blended into the prediction
    if blend_applied and odds:
        table += (
            "<tr style='background:rgba(99,102,241,0.10);"
            "border-top:1px solid rgba(99,102,241,0.25);'>"
            f"<td style='font-weight:700;color:rgba(167,139,250,0.95);padding:6px 8px;'>✨ Blended</td>"
            f"<td style='text-align:center;font-weight:700;'>{res['prob_home_win']*100:.1f}%</td>"
            f"<td style='text-align:center;font-weight:700;'>{res['prob_draw']*100:.1f}%</td>"
            f"<td style='text-align:center;font-weight:700;'>{res['prob_away_win']*100:.1f}%</td>"
            f"<td style='text-align:center;color:rgba(255,255,255,0.55);'>{final_lh:.2f} – {final_la:.2f}</td>"
            "</tr>"
        )
    table += "</tbody></table>"

    caps = []
    if blend_applied and odds:
        ou = (f" &nbsp;·&nbsp; O/U {odds['over_under']} "
              f"(market total {odds['market_total_goals']})"
              if odds.get("over_under") else "")
        caps.append(f"Moneylines: {home} {odds['moneyline_home']:+g} · "
                    f"Draw {odds['moneyline_draw']:+g} · {away} {odds['moneyline_away']:+g}{ou}")
    if poly_match:
        caps.append(
            f"🔮 Polymarket match market <code>{poly_match['event_slug']}</code> "
            f"&nbsp;·&nbsp; overround {poly_match['overround']:.2f} "
            f"(shown for reference — not blended into the prediction)")
    cap_html = "<br>".join(caps)

    st.markdown(
        f"<div class='metric-card' style='padding:14px 18px;text-align:left;'>"
        f"{table}"
        f"<p style='margin:10px 0 0;font-size:0.72rem;color:rgba(255,255,255,0.35);'>{cap_html}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

if use_market and not odds:
    st.info("💰 No DraftKings market for this matchup "
            "(odds are only published for scheduled WC 2026 fixtures). "
            "Showing pure statistical model.")
if use_market and not poly_match:
    if poly_url_failed:
        st.warning("🔮 Couldn't read a match market from that Polymarket URL. Make sure it's "
                   "the page for **this exact fixture** (a `…/event/…` link with the two "
                   "teams), then try again — or clear the field to auto-detect.")
    else:
        st.caption("🔮 No Polymarket match market auto-detected for this fixture. If you know "
                   "there is one, paste its URL in the sidebar (**Polymarket match URL**). "
                   "Polymarket's tournament-winner odds are shown below.")

# ── Polymarket WC 2026 winner odds ────────────────────────────────────────────
_poly_all = get_polymarket_odds_cached()
_poly_h   = None
_poly_a   = None
if _poly_all:
    # Direct or fuzzy match
    def _poly_lookup(team, all_odds):
        if team in all_odds:
            return all_odds[team]
        tl = team.lower()
        for k, v in all_odds.items():
            if tl in k.lower() or k.lower() in tl:
                return v
        return None
    _poly_h = _poly_lookup(home, _poly_all)
    _poly_a = _poly_lookup(away, _poly_all)

if _poly_h is not None or _poly_a is not None:
    st.divider()
    st.markdown(
        "<div class='sh'>Polymarket · 2026 World Cup Winner Odds</div>",
        unsafe_allow_html=True,
    )

    def _poly_bar(team, prob, color):
        pct = prob * 100 if prob is not None else None
        if pct is None:
            return (
                f"<div style='flex:1;padding:14px 16px;border-radius:12px;"
                f"background:rgba(255,255,255,0.04);text-align:center;'>"
                f"<div style='font-size:0.95rem;font-weight:600;color:rgba(255,255,255,0.6);'>{team}</div>"
                f"<div style='font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:4px;'>No market</div>"
                f"</div>"
            )
        bar_w = max(pct * 2, 3)   # scale: 50% → full bar at 100px width
        return (
            f"<div style='flex:1;padding:14px 16px;border-radius:12px;"
            f"background:rgba(255,255,255,0.04);text-align:center;'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{color};'>{team}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{color};margin:6px 0;'>{pct:.1f}%</div>"
            f"<div style='height:6px;border-radius:3px;background:rgba(255,255,255,0.08);margin:0 8px;'>"
            f"<div style='height:6px;border-radius:3px;width:{min(bar_w,100):.0f}%;background:{color};'></div>"
            f"</div>"
            f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:6px;'>"
            f"to win the tournament</div>"
            f"</div>"
        )

    # Relative edge between the two teams
    edge_html = ""
    if _poly_h is not None and _poly_a is not None:
        ratio = _poly_h / max(_poly_a, 1e-6)
        if ratio >= 1.5:
            edge_html = (
                f"<div style='font-size:0.78rem;color:rgba(199,210,254,0.70);"
                f"text-align:center;margin-top:6px;'>"
                f"Market rates {home} <strong>{ratio:.1f}×</strong> more likely to win the tournament</div>"
            )
        elif ratio <= 0.667:
            edge_html = (
                f"<div style='font-size:0.78rem;color:rgba(252,165,165,0.70);"
                f"text-align:center;margin-top:6px;'>"
                f"Market rates {away} <strong>{1/ratio:.1f}×</strong> more likely to win the tournament</div>"
            )
        else:
            edge_html = (
                "<div style='font-size:0.78rem;color:rgba(255,255,255,0.35);"
                "text-align:center;margin-top:6px;'>Markets see these teams as roughly equal tournament contenders</div>"
            )

    st.markdown(
        f"<div class='metric-card' style='padding:14px 18px;'>"
        f"<div style='display:flex;gap:12px;'>"
        f"{_poly_bar(home, _poly_h, '#818cf8')}"
        f"{_poly_bar(away, _poly_a, '#f87171')}"
        f"</div>"
        f"{edge_html}"
        f"<p style='margin:10px 0 0;font-size:0.68rem;color:rgba(255,255,255,0.28);'>"
        f"Source: Polymarket · prices update every 5 min · not match-specific win probabilities</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── WC Heritage panel (jfjelstul/worldcup dataset) ───────────────────────────
_h_wc = _wc_stats_for(home, wc_team_stats) if not wc_team_stats.empty else None
_a_wc = _wc_stats_for(away, wc_team_stats) if not wc_team_stats.empty else None

if _h_wc or _a_wc:
    st.divider()
    st.markdown(
        "<div class='sh'>World Cup Heritage &nbsp;·&nbsp; Historical WC Performance (1930–2022)</div>",
        unsafe_allow_html=True,
    )

    def _stage_label(score):
        if score >= 4.5:   return "🏆 Winner / Finalist"
        elif score >= 3.0: return "🥉 Semi-finalist"
        elif score >= 2.0: return "⚽ Quarter-finalist"
        elif score >= 1.0: return "🔄 Group Stage exit"
        elif score > 0:    return "📋 Group Stage"
        else:              return "🆕 No WC history"

    def _wc_card(team, wc, color):
        if wc is None or wc.get("wc_appearances", 0) == 0:
            return (
                f"<div style='flex:1;padding:16px;border-radius:12px;"
                f"background:rgba(255,255,255,0.04);text-align:center;'>"
                f"<div style='font-size:1rem;font-weight:700;color:{color};'>{team}</div>"
                f"<div style='font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:8px;'>"
                f"No World Cup history</div></div>"
            )
        apps = wc["wc_appearances"]
        win_pct = wc["wc_win_rate"] * 100
        atk = wc["wc_attack_rate"]
        def_ = wc["wc_defence_rate"]
        stage = _stage_label(wc["wc_stage_score"])
        return (
            f"<div style='flex:1;padding:16px;border-radius:12px;"
            f"background:rgba(255,255,255,0.04);'>"
            f"<div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:10px;'>{team}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.82rem;'>"
            f"<div><span style='color:rgba(255,255,255,0.40);'>Tournaments</span><br/>"
            f"<strong style='font-size:1.1rem;'>{apps}</strong></div>"
            f"<div><span style='color:rgba(255,255,255,0.40);'>WC Win rate</span><br/>"
            f"<strong style='font-size:1.1rem;'>{win_pct:.0f}%</strong></div>"
            f"<div><span style='color:rgba(255,255,255,0.40);'>Goals/game</span><br/>"
            f"<strong style='color:#34d399;'>{atk:.2f}</strong></div>"
            f"<div><span style='color:rgba(255,255,255,0.40);'>Conceded/game</span><br/>"
            f"<strong style='color:#f87171;'>{def_:.2f}</strong></div>"
            f"</div>"
            f"<div style='margin-top:10px;font-size:0.78rem;color:rgba(255,255,255,0.55);'>"
            f"Best avg stage: {stage}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div class='metric-card' style='padding:14px 18px;'>"
        f"<div style='display:flex;gap:12px;'>"
        f"{_wc_card(home, _h_wc, '#818cf8')}"
        f"{_wc_card(away, _a_wc, '#f87171')}"
        f"</div>"
        f"<p style='margin:10px 0 0;font-size:0.68rem;color:rgba(255,255,255,0.28);'>"
        f"Source: jfjelstul/worldcup (GitHub) · 1930–2022 · "
        f"used as structural GLM features (wc_attack_rate, wc_defence_rate, wc_stage_score)</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────
cfg = {"displayModeBar": False}
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='sh'>Outcome Probability</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_donut(res["prob_home_win"], res["prob_draw"],
                                 res["prob_away_win"], home, away),
                    use_container_width=True, config=cfg)
with col2:
    st.markdown("<div class='sh'>Goal Differential  —  95% CI shaded</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_goal_diff(res["lambda_home"], res["lambda_away"],
                                     res["home_uncertainty"], res["away_uncertainty"],
                                     home, away, res["ci_low"], res["ci_high"], n_sims),
                    use_container_width=True, config=cfg)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.markdown("<div class='sh'>Goals Scored Distribution</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_goal_dist(res["lambda_home"], res["lambda_away"],
                                     res["home_uncertainty"], res["away_uncertainty"],
                                     home, away, n_sims),
                    use_container_width=True, config=cfg)
with col4:
    st.markdown("<div class='sh'>Scoreline Heatmap  (%)</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_heatmap(res["lambda_home"], res["lambda_away"],
                                   res["home_uncertainty"], res["away_uncertainty"],
                                   home, away, n_sims),
                    use_container_width=True, config=cfg)

# ── Stats strip + ELO ─────────────────────────────────────────────────────────
st.divider()
d1, d2, d3, d4 = st.columns(4)
d1.metric(f"λ {home}", f"{res['lambda_home']:.3f}", help="Expected goals after injury adj.")
d2.metric(f"λ {away}", f"{res['lambda_away']:.3f}", help="Expected goals after injury adj.")
d3.metric("95% CI", f"[{res['ci_low']:.1f}, {res['ci_high']:.1f}]")
d4.metric("MC Draws", f"{n_sims:,}")

if h_elo and a_elo:
    st.markdown("<div class='sh' style='margin-top:12px;'>World Football ELO</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_elo_bar(home, away, h_elo, a_elo),
                    use_container_width=True, config=cfg)

# ── Squad panels ──────────────────────────────────────────────────────────────
if include_squad and (h_squad or a_squad):
    st.divider()
    st.markdown(
        "<p style='font-size:1rem;font-weight:700;color:rgba(255,255,255,0.92);margin-bottom:4px;'>"
        "🎽 Squad &amp; Injury Report</p>",
        unsafe_allow_html=True,
    )
    tab_home, tab_away = st.tabs([f"  {home}  ", f"  {away}  "])
    with tab_home:
        squad_panel(h_squad, home, h_atk_adj, h_def_adj,
                    form=h_form, player_stats_df=_player_stats_df, editable=is_admin)
    with tab_away:
        squad_panel(a_squad, away, a_atk_adj, a_def_adj,
                    form=a_form, player_stats_df=_player_stats_df, editable=is_admin)

# ── AI Analysis (bring-your-own-key LLM) ──────────────────────────────────────
st.divider()
st.markdown(
    "<p style='font-size:1rem;font-weight:700;color:rgba(255,255,255,0.92);margin-bottom:4px;'>"
    "🤖 AI Match Analysis</p>"
    "<p style='font-size:0.78rem;color:rgba(255,255,255,0.35);margin-top:0;'>"
    "Add your own LLM key below and an AI analyst writes an expert breakdown. "
    "Only the structured prediction numbers are sent — never any free text.</p>",
    unsafe_allow_html=True,
)

from llm_analyst import build_match_payload, stream_analysis

# OpenAI-compatible provider presets. Qwen (and most open models) are served
# through the OpenAI chat API shape, so one code path works for all of them.
LLM_PRESETS = {
    "Ollama (local · free)": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "deepseek-r1:1.5b", "needs_key": False,
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen-2.5-72b-instruct", "needs_key": True,
    },
    "Alibaba DashScope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus", "needs_key": True,
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo", "needs_key": True,
    },
    "DeepInfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "Qwen/Qwen2.5-72B-Instruct", "needs_key": True,
    },
    "Custom (any OpenAI-compatible)": {
        "base_url": "", "model": "", "needs_key": True,
    },
}

with st.expander("⚙️  LLM settings — bring your own key"):
    provider = st.selectbox("Provider", list(LLM_PRESETS), key="llm_provider")
    preset   = LLM_PRESETS[provider]

    # When the provider changes, reset the URL/model fields to that preset's
    # defaults (set BEFORE the widgets are created so they pick it up).
    if st.session_state.get("_llm_last_provider") != provider:
        st.session_state["llm_base_url"] = preset["base_url"]
        st.session_state["llm_model"]    = preset["model"]
        st.session_state["_llm_last_provider"] = provider

    base_url = st.text_input("Base URL", key="llm_base_url",
                             placeholder="https://...")
    model    = st.text_input("Model", key="llm_model",
                             placeholder="deepseek-r1:1.5b / qwen/qwen-2.5-72b-instruct / ...")
    api_key  = st.text_input(
        "API key", type="password", key="llm_api_key",
        help="Held only in this browser session's memory. Never written to "
             "disk and cleared when you close the tab.",
        placeholder="(leave blank for local Ollama)",
    )
    st.caption(
        "🔒 Your key stays in this session only — nothing is saved to disk or "
        "shared, and it's wiped the moment you close the tab. "
        "Local Ollama is free but only works when you run this app on your own "
        "machine (not on a hosted server)."
    )

_key   = st.session_state.get("llm_api_key", "").strip()
_base  = st.session_state.get("llm_base_url", "").strip()
_model = st.session_state.get("llm_model", "").strip()
_needs_key = LLM_PRESETS[st.session_state.get("llm_provider",
                          "Ollama (local · free)")]["needs_key"]
_ready = bool(_base and _model and (_key or not _needs_key))

_analysis_key = f"analysis::{home}::{away}"

if not _ready:
    st.info("🔑 Open **LLM settings** above, choose a provider, and paste your "
            "API key (or pick Ollama for a free local run) to enable analysis.")
else:
    if st.button("✨ Analyse this prediction", type="secondary"):
        payload = build_match_payload(
            home=home, away=away,
            result=res, model_result=model_res, odds=odds,
            h_elo=h_elo, a_elo=a_elo,
            h_injured=h_injured, a_injured=a_injured,
            hf=hf, af=af, stage=stage,
            market_weight=mkt_weight, blend_applied=blend_applied,
        )
        try:
            with st.spinner("The AI analyst is reading the numbers…"):
                full_text = st.write_stream(
                    stream_analysis(payload, api_key=_key,
                                    base_url=_base, model=_model)
                )
            st.session_state[_analysis_key] = full_text
        except RuntimeError as e:
            st.error(str(e))
    elif _analysis_key in st.session_state:
        # Re-render the last analysis for this matchup after reruns
        st.markdown(st.session_state[_analysis_key])

# ── Save prediction panel ─────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='font-size:1rem;font-weight:700;color:rgba(255,255,255,0.92);margin-bottom:4px;'>"
    "💾 Save Prediction</p>",
    unsafe_allow_html=True,
)

# Find this match's event_id + schedule date (if it's a real scheduled fixture)
_event_id, _match_date = None, ""
if not _fixtures.empty:
    _teams = {home, away}
    _m = _fixtures[_fixtures.apply(
        lambda r: {r["home_team"], r["away_team"]} == _teams, axis=1)]
    if not _m.empty:
        _event_id   = _m.iloc[0]["event_id"]
        _match_date = _m.iloc[0]["date"]

_pred_key = _match_key(home, away, _event_id)
_saved    = load_predictions().get(_pred_key)

# Model's own pick derived from the (blended) result
_probs = {home: res["prob_home_win"], "Draw": res["prob_draw"], away: res["prob_away_win"]}
_model_pick   = max(_probs, key=_probs.get)
_model_conf   = _probs[_model_pick] * 100
_model_score  = f"{res['most_likely_score'][0]}–{res['most_likely_score'][1]}"
_model_summary = f"{_model_pick} ({_model_conf:.0f}%) · {home} {_model_score} {away}"

_blend_note = ""
if blend_applied and odds:
    _blend_note = ("<div style='font-size:0.72rem;color:rgba(251,191,36,0.7);"
                   "margin-top:4px;'>💰 market-blended</div>")
_gd_sign = "+" if res["mean_goal_diff"] >= 0 else ""

sp1, sp2 = st.columns([1, 1])
with sp1:
    st.markdown(
        f"<div class='metric-card' style='text-align:left;padding:14px 18px;'>"
        f"<div class='lbl'>📊 Model prediction</div>"
        f"<div style='font-size:1.05rem;font-weight:700;color:rgba(167,139,250,0.95);"
        f"margin-top:4px;'>{_model_pick} &nbsp;<span style='color:rgba(255,255,255,0.45);"
        f"font-weight:500;font-size:0.85rem;'>{_model_conf:.0f}% confidence</span></div>"
        f"<div style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-top:2px;'>"
        f"Likely score {home} {_model_score} {away} · "
        f"GD {_gd_sign}{res['mean_goal_diff']:.2f} "
        f"± {res['margin_of_error']:.2f}</div>"
        f"{_blend_note}"
        f"</div>",
        unsafe_allow_html=True,
    )

with sp2:
    st.markdown("<div class='lbl' style='margin-bottom:6px;'>✍️ Your final prediction</div>",
                unsafe_allow_html=True)
    _pick_opts = [home, "Draw", away]
    _def_pick = 0
    _def_hs, _def_as = res["most_likely_score"][0], res["most_likely_score"][1]
    if _saved:
        if _saved.get("my_pick") in _pick_opts:
            _def_pick = _pick_opts.index(_saved["my_pick"])
        _def_hs = _saved.get("my_home_score", _def_hs)
        _def_as = _saved.get("my_away_score", _def_as)

    my_pick = st.radio("Your pick", _pick_opts, index=_def_pick,
                        horizontal=True, label_visibility="collapsed")
    cs1, cs2, cs3 = st.columns([2, 1, 2])
    with cs1:
        my_hs = st.number_input(f"{home}", 0, 15, int(_def_hs), key="my_hs")
    with cs2:
        st.markdown("<p style='text-align:center;margin-top:28px;color:rgba(255,255,255,0.4);'>–</p>",
                    unsafe_allow_html=True)
    with cs3:
        my_as = st.number_input(f"{away}", 0, 15, int(_def_as), key="my_as")

sb1, sb2, sb3 = st.columns([2, 2, 3])
with sb1:
    if st.button("💾 Save prediction", use_container_width=True, type="primary",
                 disabled=not is_admin,
                 help=None if is_admin else "Unlock full access to save predictions"):
        save_prediction(_pred_key, {
            "date":           str(_match_date),
            "home":           home,
            "away":           away,
            "event_id":       str(_event_id) if _event_id else "",
            "model_pick":     _model_pick,
            "model_conf":     round(_model_conf, 1),
            "model_score":    _model_score,
            "model_gd":       round(res["mean_goal_diff"], 2),
            "market_blended": bool(blend_applied and odds),
            "my_pick":        my_pick,
            "my_home_score":  int(my_hs),
            "my_away_score":  int(my_as),
        })
        st.success("Saved to schedule ✓")
        st.rerun()
with sb2:
    if _saved and st.button("🗑️ Remove", use_container_width=True, disabled=not is_admin):
        delete_prediction(_pred_key)
        st.rerun()
with sb3:
    if not is_admin:
        st.markdown(
            "<p style='font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:8px;'>"
            "🔒 Saving is available with full access.</p>",
            unsafe_allow_html=True,
        )
    elif _saved:
        st.markdown(
            f"<p style='font-size:0.78rem;color:rgba(134,239,172,0.80);margin-top:8px;'>"
            f"✓ Saved: you picked <b>{_saved['my_pick']}</b> "
            f"{_saved['my_home_score']}–{_saved['my_away_score']}</p>",
            unsafe_allow_html=True,
        )

# ── Schedule expander ─────────────────────────────────────────────────────────
if not _fixtures.empty:
    st.divider()
    with st.expander("Full Schedule"):
        _preds = load_predictions()

        def _lookup_pred(row):
            key = _match_key(row["home_team"], row["away_team"], row.get("event_id"))
            return _preds.get(key) or _preds.get(
                _match_key(row["home_team"], row["away_team"]))

        show = _fixtures[["date","home_team","away_team","home_score",
                           "away_score","status","event_id"]].copy()

        model_col, mine_col = [], []
        for _, row in show.iterrows():
            p = _lookup_pred(row)
            if p:
                model_col.append(f"{p['model_pick']} ({p['model_conf']:.0f}%) "
                                  f"{p['model_score']}")
                mine_col.append(f"{p['my_pick']} "
                                f"{p['my_home_score']}–{p['my_away_score']}")
            else:
                model_col.append("—")
                mine_col.append("—")

        show = show.drop(columns=["event_id"])
        show.columns = ["Date","Home","Away","H","A","Status"]
        show["H"] = show["H"].apply(lambda v: "—" if pd.isna(v) else str(int(v)))
        show["A"] = show["A"].apply(lambda v: "—" if pd.isna(v) else str(int(v)))
        show["📊 Model"]      = model_col
        show["✍️ My Pick"]    = mine_col
        st.dataframe(show, use_container_width=True, hide_index=True)

        n_saved = sum(1 for m in model_col if m != "—")
        if n_saved:
            st.caption(f"💾 {n_saved} saved prediction(s) for this session. "
                       "Kept in memory only — they clear when you close the tab.")

# ── Model details link ────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center;margin-top:8px;'>"
    "<a href='./Model_Details' target='_self' style='"
    "color:rgba(199,210,254,0.60);font-size:0.78rem;text-decoration:none;"
    "border-bottom:1px solid rgba(199,210,254,0.20);padding-bottom:1px;'>"
    "📖 Full model documentation →</a></p>",
    unsafe_allow_html=True,
)
