"""
About Me — renders ../about_me.md (bio + photo / LinkedIn / email settings).
Edit about_me.md to change the content; no need to touch this file.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="About Me - WC 2026 Predictor",
                   page_icon="\U0001f464", layout="wide")

st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
      "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] { color-scheme: dark !important; }
.stApp { background:
    radial-gradient(ellipse 70% 50% at 20% 0%,  rgba(99,102,241,0.14) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 90% 100%, rgba(120,90,200,0.10) 0%, transparent 60%),
    linear-gradient(180deg, #000 0%, #060608 50%, #000 100%); min-height: 100vh; }
.main .block-container { background: transparent !important; padding-top: 2.4rem !important; max-width: 880px; }
section[data-testid="stSidebar"] { background: rgba(15,15,28,0.72) !important;
    backdrop-filter: blur(48px) saturate(160%) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important; }
h1 { color: rgba(255,255,255,0.96) !important; font-weight:700 !important;
     letter-spacing:-0.035em !important; font-size:2.6rem !important; line-height:1.05; }
h2,h3 { color: rgba(255,255,255,0.93) !important; letter-spacing:-0.025em; }
p, li { color: rgba(255,255,255,0.66); line-height:1.75; font-size:1.02rem; }
strong { color: rgba(255,255,255,0.92) !important; }
a { color: #6ea8ff !important; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border-color: rgba(255,255,255,0.08) !important; margin: 22px 0 !important; }
blockquote { border-left: 3px solid rgba(99,102,241,0.5); margin: 8px 0; padding: 4px 16px;
    color: rgba(255,255,255,0.7) !important; font-style: italic; }
.photo {
  width: 200px; height: 200px; border-radius: 24px; object-fit: cover;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0.5px 0 rgba(255,255,255,0.12) inset; }
.photo-ph {
  width: 200px; height: 200px; border-radius: 24px;
  display: flex; align-items: center; justify-content: center; flex-direction: column;
  background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.2);
  color: rgba(255,255,255,0.4); text-align: center; font-size: 0.72rem; gap: 6px; }
.photo-ph .initials { font-size: 2.4rem; font-weight: 700; color: rgba(255,255,255,0.55);
  letter-spacing: -0.02em; }
.contact { display:inline-flex; align-items:center; gap:7px; margin: 6px 8px 0 0;
  padding: 7px 16px; border-radius: 980px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
  color: rgba(255,255,255,0.85) !important; font-size: 0.88rem; font-weight: 500; }
.contact:hover { background: rgba(10,132,255,0.18); border-color: rgba(10,132,255,0.5);
  text-decoration: none !important; }
.joke { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px; padding: 18px 24px; margin: 8px 0 6px; color: rgba(255,255,255,0.72);
  font-size: 0.98rem; line-height: 1.7; }
/* Lock to the dark design — hide the theme switcher; keep chrome dark. */
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stHeader"], header[data-testid="stHeader"],
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDialog"] > div, div[role="dialog"] {
  background: #16161f !important; color: rgba(255,255,255,0.90) !important; }
/* Sidebar expand/collapse controls -> hamburger icon */
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

from auth import require_login
require_login()

_ROOT = Path(__file__).resolve().parent.parent
_MD = _ROOT / "about_me.md"


def _parse(path: Path):
    """Return (meta dict, markdown body). Meta = KEY: value lines before first ---."""
    meta, body_lines, in_body = {}, [], False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}, ""
    for line in lines:
        s = line.strip()
        if in_body:
            body_lines.append(line)
            continue
        if s.startswith(";;") or not s:
            continue
        if s == "---":
            in_body = True
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            meta[k.strip().upper()] = v.strip()
    return meta, "\n".join(body_lines).strip()


meta, body = _parse(_MD)

if not body:
    st.error(f"Could not read {_MD.name}. Create about_me.md in the project root to edit this page.")
    st.stop()

st.markdown("<h1>About Me</h1>", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

left, right = st.columns([1, 2.3], gap="large")

with left:
    photo = meta.get("PHOTO", "").strip()
    photo_path = (_ROOT / photo) if photo else None
    if photo_path and photo_path.exists():
        import base64
        _mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                 ".gif": "gif", ".webp": "webp"}.get(photo_path.suffix.lower(), "jpeg")
        _b64 = base64.b64encode(photo_path.read_bytes()).decode()
        st.markdown(
            f"<img class='photo' src='data:image/{_mime};base64,{_b64}'/>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='photo-ph'><span class='initials'>SR</span>"
            f"<span>Add your photo<br/>at <code>{photo or 'assets/about_me.jpg'}</code></span></div>",
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        "<div style='font-size:1.5rem;font-weight:700;color:rgba(255,255,255,0.95);"
        "letter-spacing:-0.02em;'>Samrat Roy</div>"
        "<div style='font-size:0.95rem;color:rgba(255,255,255,0.45);margin-top:2px;'>"
        "Supply Chain professional &amp; mathematics geek</div>",
        unsafe_allow_html=True,
    )
    links = ""
    li = meta.get("LINKEDIN", "").strip()
    em = meta.get("EMAIL", "").strip()
    if li:
        links += f"<a class='contact' href='{li}' target='_blank'>in&nbsp; LinkedIn</a>"
    if em:
        links += f"<a class='contact' href='mailto:{em}'>✉&nbsp; {em}</a>"
    if links:
        st.markdown(f"<div style='margin-top:14px'>{links}</div>", unsafe_allow_html=True)

st.divider()
st.markdown(body, unsafe_allow_html=True)
