"""
auth.py — access control for the WC 2026 Predictor.

The site opens for everyone in **read-only visitor** mode — no gate, no login.
To unlock the full model, click **Access full model** (in the sidebar, or on an
admin-only page) and enter the admin password in the pop-up window.

  • visitor (default) — read-only: the main predictor (WC2026 UI) shows the next
                        fixture, plus the About Me page. No controls.
  • admin             — full access to every page and control, after entering
                        the password.

The admin password is checked against a SHA-256 hash so the plaintext is not
stored in this (public) repo. Override it by setting `admin_password` in
st.secrets.
"""
from __future__ import annotations

import hashlib

import streamlit as st

_ACCESS = "access_level"                  # session_state: "admin" | "visitor" | unset

# sha256("@dm!nis-traitor") — the default admin password, stored hashed.
_ADMIN_HASH = "405633ea226b6c859d8b1d163f2505bd2dd570825671e6f49f09cdd843c4ac1c"


def _password_ok(pw: str) -> bool:
    if not pw:
        return False
    try:
        secret = st.secrets.get("admin_password")
    except Exception:
        secret = None
    if secret:
        return pw == secret
    return hashlib.sha256(pw.encode("utf-8")).hexdigest() == _ADMIN_HASH


def current_access() -> str:
    """Current access level — defaults to 'visitor' (the site is open)."""
    return st.session_state.get(_ACCESS) or "visitor"


@st.dialog("Access full model")
def _unlock_dialog():
    """Pop-up window: enter the admin password to unlock full access."""
    st.write("Enter the admin password to unlock the full model — every page, "
             "match selection, injuries, and the simulation controls.")
    pw = st.text_input("Admin password", type="password", key="acc_unlock_pw",
                       placeholder="••••••••")
    if st.button("Unlock", type="primary", use_container_width=True,
                 key="acc_unlock_submit"):
        if _password_ok(pw):
            st.session_state[_ACCESS] = "admin"
            st.rerun()
        else:
            st.error("Incorrect password.")


def _access_controls(level: str):
    """Sidebar badge + the Access-full-model / Log-out button."""
    logout = unlock = False
    with st.sidebar:
        if level == "admin":
            st.markdown(
                "<div style='font-size:0.78rem;color:rgba(134,239,172,0.85);"
                "margin-bottom:6px;'>🔓 Admin access</div>",
                unsafe_allow_html=True)
            logout = st.button("Log out", key="acc_logout", use_container_width=True)
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:rgba(255,255,255,0.5);"
                "margin-bottom:6px;'>👁 Read-only preview</div>",
                unsafe_allow_html=True)
            unlock = st.button("🔓 Access full model", key="acc_unlock",
                               use_container_width=True)
    # Trigger outside the sidebar context so the modal renders as a centred overlay.
    if logout:
        st.session_state.pop(_ACCESS, None)
        st.rerun()
    if unlock:
        _unlock_dialog()


def _admin_required_panel():
    st.markdown("""
    <style>
    .gate-wrap { max-width: 460px; margin: 7vh auto 0; text-align: center; }
    .gate-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em;
        color: rgba(255,255,255,0.96); margin-bottom: 4px; }
    .gate-sub { font-size: 0.92rem; color: rgba(255,255,255,0.45); margin-bottom: 22px; }
    </style>
    <div class='gate-wrap'>
      <div class='gate-title'>🔒 Part of the full model</div>
      <div class='gate-sub'>This page is available with full access. Unlock it with
      the admin password, or head back to the live predictor.</div>
    </div>
    """, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🔓 Access full model", type="primary",
                     use_container_width=True, key="acc_unlock_page"):
            _unlock_dialog()


def require_access(required: str = "visitor") -> str:
    """
    Return the current access level ("admin" or "visitor"). The site is open in
    read-only visitor mode by default — no blocking gate. Pages that pass
    required="admin" show an unlock panel (and st.stop()) for visitors.
    """
    level = st.session_state.get(_ACCESS) or "visitor"
    _access_controls(level)

    if required == "admin" and level != "admin":
        _admin_required_panel()
        st.stop()

    return level


# Backwards-compatible alias (old call sites used require_login()).
def require_login():
    return require_access("visitor")
