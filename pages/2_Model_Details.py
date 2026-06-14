"""
FIFA WC 2026 Predictor - Model Details page
Full mathematical walkthrough for a statistics learner.
"""
import streamlit as st

st.set_page_config(
    page_title="Model Details - WC 2026 Predictor",
    page_icon="\U0001f4d6",
    layout="wide",
)

st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
               "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  color-scheme: dark !important; }
.stApp {
  background:
    radial-gradient(ellipse at 15% 10%,  rgba(99,102,241,0.22) 0%,  transparent 50%),
    radial-gradient(ellipse at 85% 85%,  rgba(168,85,247,0.18)  0%,  transparent 50%),
    radial-gradient(ellipse at 65% 25%,  rgba(20,184,166,0.13)  0%,  transparent 45%),
    radial-gradient(ellipse at 25% 80%,  rgba(245,158,11,0.10)  0%,  transparent 45%),
    linear-gradient(160deg, #09090f 0%, #0e0e1a 45%, #0a0f14 100%);
  min-height: 100vh; }
.main .block-container {
  background: transparent !important;
  padding-top: 2rem !important;
  max-width: 900px; }
section[data-testid="stSidebar"] {
  background: rgba(15,15,28,0.72) !important;
  backdrop-filter: blur(48px) saturate(160%) !important;
  border-right: 1px solid rgba(255,255,255,0.07) !important; }
h1,h2,h3,h4 { color: rgba(255,255,255,0.92) !important; letter-spacing:-0.02em; }
p, li       { color: rgba(255,255,255,0.66) !important; line-height: 1.75; }
strong      { color: rgba(255,255,255,0.90) !important; }
code        { background: rgba(255,255,255,0.08) !important;
              color: rgba(167,243,208,0.90) !important;
              border-radius: 5px !important; padding: 1px 5px !important; }
pre         { background: rgba(255,255,255,0.05) !important;
              border: 1px solid rgba(255,255,255,0.09) !important;
              border-radius: 14px !important; padding: 16px !important; }
pre code    { background: transparent !important; padding: 0 !important; }
table       { width: 100%; border-collapse: collapse; margin: 12px 0; }
th          { background: rgba(99,102,241,0.18) !important;
              color: rgba(199,210,254,0.90) !important;
              padding: 10px 14px; text-align: left;
              font-size: 0.80rem; letter-spacing: 0.05em; text-transform: uppercase; }
td          { color: rgba(255,255,255,0.70) !important;
              padding: 9px 14px;
              border-bottom: 1px solid rgba(255,255,255,0.06) !important; }
tr:hover td { background: rgba(255,255,255,0.03) !important; }
hr { border-color: rgba(255,255,255,0.08) !important; margin: 28px 0 !important; }
.card {
  background: rgba(255,255,255,0.05);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 20px;
  padding: 28px 32px;
  margin: 18px 0;
  box-shadow: 0 1px 0 rgba(255,255,255,0.08) inset, 0 8px 32px rgba(0,0,0,0.30); }
.step-badge {
  display: inline-block;
  background: linear-gradient(135deg,rgba(99,102,241,0.70),rgba(139,92,246,0.60));
  border-radius: 8px; padding: 3px 12px;
  font-size: 0.70rem; font-weight: 700;
  color: rgba(199,210,254,0.95);
  letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 10px; }
.plain {
  background: rgba(34,197,94,0.08);
  border-left: 3px solid rgba(34,197,94,0.55);
  border-radius: 10px; padding: 14px 18px; margin: 16px 0;
  font-size: 0.88rem; color: rgba(187,247,208,0.92); line-height: 1.65; }
.plain b { color: rgba(220,252,231,0.98); }
.math-note {
  background: rgba(99,102,241,0.07);
  border-left: 3px solid rgba(99,102,241,0.45);
  border-radius: 10px; padding: 14px 18px; margin: 16px 0;
  font-size: 0.88rem; color: rgba(199,210,254,0.85); line-height: 1.65; }
.math-note b { color: rgba(224,231,255,0.98); }
.warn {
  background: rgba(245,158,11,0.07);
  border-left: 3px solid rgba(245,158,11,0.50);
  border-radius: 10px; padding: 14px 18px; margin: 16px 0;
  font-size: 0.88rem; color: rgba(253,230,138,0.85); line-height: 1.65; }
.warn b { color: rgba(254,243,199,0.98); }
.term { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px; padding: 12px 16px; margin: 7px 0; }
.term .word { font-weight:700; color:rgba(199,210,254,0.95); font-size:0.92rem; }
.term .def  { color:rgba(255,255,255,0.66); font-size:0.85rem;
              line-height:1.6; margin-top:3px; }
.pill { display:inline-block; background:rgba(255,255,255,0.07);
        border:1px solid rgba(255,255,255,0.12); border-radius:999px;
        padding:3px 12px; font-size:0.74rem; color:rgba(255,255,255,0.55);
        margin:3px 2px; }
.flow-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:12px 0; }
.flow-box { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10);
            border-radius:10px; padding:8px 14px; font-size:0.83rem;
            color:rgba(255,255,255,0.75); white-space:nowrap; }
.flow-arrow { color:rgba(255,255,255,0.30); font-size:1.1rem; }
/* Lock to the dark design — hide the theme switcher; keep chrome dark. */
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stHeader"], header[data-testid="stHeader"],
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDialog"] > div, div[role="dialog"] {
  background: #16161f !important; color: rgba(255,255,255,0.90) !important; }
/* Sidebar collapse control: kill Material Symbols ligature ghost, draw chevron */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { position: relative !important; }
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="collapsedControl"] * { font-size: 0 !important; }
[data-testid="stSidebarCollapseButton"]::after { content: "\2039"; }
[data-testid="stSidebarCollapsedControl"]::after,
[data-testid="collapsedControl"]::after { content: "\203A"; }
[data-testid="stSidebarCollapseButton"]::after,
[data-testid="stSidebarCollapsedControl"]::after,
[data-testid="collapsedControl"]::after {
  position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; font-size: 1.15rem !important; line-height: 1;
  color: rgba(255,255,255,0.55); pointer-events: none; }
</style>
""", unsafe_allow_html=True)

# ── Authentication gate ───────────────────────────────────────────────────────
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from auth import require_login
require_login()


# ─────────────────────────────────────────────────────────────────────────────
# Content is authored in ../model_details.md so it can be edited without touching
# this file. The mini-renderer below understands a few @@directives@@ (see the
# header of model_details.md for the full list); everything else is plain
# Markdown rendered by Streamlit.
# ─────────────────────────────────────────────────────────────────────────────
import re
from pathlib import Path

_MD_PATH = Path(__file__).resolve().parent.parent / "model_details.md"
_NOTE_CLASS = {"green": "plain", "blue": "math-note", "amber": "warn"}


def _inline_md(text: str) -> str:
    """Tiny Markdown -> HTML for inside callout boxes: **bold** and line breaks."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br/>")


def _flush(buf):
    if buf:
        md = "\n".join(buf).strip()
        if md:
            st.markdown(md, unsafe_allow_html=True)
        buf.clear()


def render_model_details(path=_MD_PATH):
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error(f"Content file not found: {path}. Create model_details.md to edit this page.")
        return

    lines = raw.splitlines()
    buf, card_open = [], False
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        if s.startswith(";;"):                       # comment line
            i += 1
            continue

        if s.startswith("@@TITLE:"):
            _flush(buf)
            payload = s[len("@@TITLE:"):].rstrip("@").strip()
            title, sub = (payload.split("|", 1) + [""])[:2]
            st.markdown(
                f"<h1 style='font-size:2.1rem;margin-bottom:4px;'>{title.strip()}</h1>"
                f"<p style='color:rgba(255,255,255,0.38);font-size:0.86rem;margin-top:0;'>"
                f"{sub.strip()}</p>",
                unsafe_allow_html=True,
            )
            st.divider()
            i += 1
            continue

        if s == "@@CARD@@":
            _flush(buf)
            if card_open:
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            card_open = True
            i += 1
            continue

        if s.startswith("@@BADGE:"):
            _flush(buf)
            st.markdown(
                f"<div class='step-badge'>{s[len('@@BADGE:'):].rstrip('@').strip()}</div>",
                unsafe_allow_html=True,
            )
            i += 1
            continue

        if s.startswith("@@PILLS:"):
            _flush(buf)
            tags = [t.strip() for t in s[len("@@PILLS:"):].rstrip("@").split("|") if t.strip()]
            st.markdown("".join(f"<span class='pill'>{t}</span>" for t in tags),
                        unsafe_allow_html=True)
            i += 1
            continue

        if s.startswith("@@FLOW:"):
            _flush(buf)
            boxes = [b.strip() for b in s[len("@@FLOW:"):].rstrip("@").split("|") if b.strip()]
            html = "<div class='flow-row'>"
            for j, b in enumerate(boxes):
                b = b.replace("&", "&amp;").replace("->", "&#8594;").replace("lambda", "&#955;")
                html += f"<div class='flow-box'>{b}</div>"
                if j < len(boxes) - 1:
                    html += "<span class='flow-arrow'>&#8594;</span>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
            i += 1
            continue

        if s == "@@LATEX@@":
            _flush(buf)
            i += 1
            body = []
            while i < n and lines[i].strip() != "@@END@@":
                body.append(lines[i])
                i += 1
            i += 1
            st.latex("\n".join(body))
            continue

        if s == "@@CODE@@":
            _flush(buf)
            i += 1
            body = []
            while i < n and lines[i].strip() != "@@END@@":
                body.append(lines[i])
                i += 1
            i += 1
            st.code("\n".join(body), language="text")
            continue

        if s.startswith("@@NOTE:"):
            _flush(buf)
            cls = _NOTE_CLASS.get(s[len("@@NOTE:"):].rstrip("@").strip(), "plain")
            i += 1
            body = []
            while i < n and lines[i].strip() != "@@END@@":
                body.append(lines[i])
                i += 1
            i += 1
            st.markdown(f"<div class='{cls}'>{_inline_md(chr(10).join(body).strip())}</div>",
                        unsafe_allow_html=True)
            continue

        if s == "@@GLOSSARY@@":
            _flush(buf)
            i += 1
            entries = []
            while i < n and lines[i].strip() != "@@END@@":
                row = lines[i].strip()
                if row.startswith(";;"):
                    i += 1
                    continue
                if "::" in row:
                    w, d = row.split("::", 1)
                    entries.append((w.strip(), d.strip()))
                i += 1
            i += 1
            with st.expander("Key terms (click to expand)", expanded=False):
                for w, d in entries:
                    st.markdown(
                        f"<div class='term'><div class='word'>{w}</div>"
                        f"<div class='def'>{d}</div></div>",
                        unsafe_allow_html=True,
                    )
            continue

        if s.startswith("@@FOOTER:"):
            _flush(buf)
            if card_open:
                st.markdown("</div>", unsafe_allow_html=True)
                card_open = False
            text = s[len("@@FOOTER:"):].rstrip("@").strip()
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align:center;font-size:0.72rem;color:rgba(255,255,255,0.20);'>"
                f"{text}</p>",
                unsafe_allow_html=True,
            )
            i += 1
            continue

        buf.append(line)                              # default: plain Markdown
        i += 1

    _flush(buf)
    if card_open:
        st.markdown("</div>", unsafe_allow_html=True)


render_model_details()
