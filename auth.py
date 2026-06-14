"""
auth.py — access gate for the WC 2026 Predictor.

Two access levels, chosen on a single password page:

  • admin    — full access to every page and every control.
               Requires the admin password.
  • visitor  — read-only. May view the main predictor (WC2026 UI) and the
               About Me page, but NOT Model Details or Expert Analysis, and
               admin-only controls are hidden.

The admin password is checked against a SHA-256 hash so the plaintext is not
stored in this (public) repo. To change it, set `admin_password` in
st.secrets — that takes precedence over the built-in hash.
"""
from __future__ import annotations

import hashlib

import streamlit as st

_ACCESS = "access_level"                  # session_state: None | "admin" | "visitor"

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


def current_access():
    """Return the current access level, or None if not chosen yet."""
    return st.session_state.get(_ACCESS)


def _gate_styles():
    st.markdown("""
    <style>
    .gate-wrap { max-width: 460px; margin: 7vh auto 0; text-align: center; }
    .gate-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.03em;
        color: rgba(255,255,255,0.96); margin-bottom: 4px; }
    .gate-sub { font-size: 0.92rem; color: rgba(255,255,255,0.45); margin-bottom: 26px; }
    </style>
    """, unsafe_allow_html=True)


def _login_panel(key_prefix: str = "gate"):
    """Render the password / skip page. Sets access on submit and reruns."""
    _gate_styles()
    st.markdown(
        "<div class='gate-wrap'>"
        "<div class='gate-title'>⚽ WC 2026 Predictor</div>"
        "<div class='gate-sub'>Enter the admin password for full access, "
        "or continue as a visitor (read-only).</div></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        pw = st.text_input("Admin password", type="password",
                           key=f"{key_prefix}_pw", placeholder="••••••••")
        c1, c2 = st.columns(2)
        with c1:
            enter = st.button("Enter", type="primary", use_container_width=True,
                              key=f"{key_prefix}_enter")
        with c2:
            skip = st.button("Continue as visitor", use_container_width=True,
                             key=f"{key_prefix}_skip")
        if enter:
            if _password_ok(pw):
                st.session_state[_ACCESS] = "admin"
                st.rerun()
            else:
                st.error("Incorrect password.")
        if skip:
            st.session_state[_ACCESS] = "visitor"
            st.rerun()


def _admin_required_panel():
    """Shown when a visitor opens an admin-only page; offers to unlock."""
    _gate_styles()
    st.markdown(
        "<div class='gate-wrap'>"
        "<div class='gate-title'>🔒 Admins only</div>"
        "<div class='gate-sub'>This page requires the admin password. "
        "Enter it to unlock, or go back to the main predictor.</div></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        pw = st.text_input("Admin password", type="password", key="gate_upgrade_pw",
                           placeholder="••••••••")
        if st.button("Unlock", type="primary", use_container_width=True,
                     key="gate_upgrade_btn"):
            if _password_ok(pw):
                st.session_state[_ACCESS] = "admin"
                st.rerun()
            else:
                st.error("Incorrect password.")


def _sidebar_badge(level: str):
    with st.sidebar:
        if level == "admin":
            st.markdown(
                "<div style='font-size:0.78rem;color:rgba(134,239,172,0.85);"
                "margin-bottom:6px;'>🔓 Admin access</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:rgba(255,255,255,0.5);"
                "margin-bottom:6px;'>👁 Read-only visitor</div>",
                unsafe_allow_html=True)
        if st.button("Log out", key="gate_logout", use_container_width=True):
            st.session_state.pop(_ACCESS, None)
            st.rerun()


def require_access(required: str = "visitor") -> str:
    """
    Ensure the visitor has at least `required` access ("visitor" or "admin").

    Renders the password page (or an admin-unlock prompt) and st.stop()s until
    the requirement is met. Returns the granted level ("admin" or "visitor").
    """
    level = st.session_state.get(_ACCESS)

    if level is None:                      # first visit — choose access
        _login_panel()
        st.stop()

    if required == "admin" and level != "admin":
        _sidebar_badge(level)
        _admin_required_panel()
        st.stop()

    _sidebar_badge(level)
    return level


# Backwards-compatible alias (old call sites used require_login()).
def require_login():
    return require_access("visitor")
