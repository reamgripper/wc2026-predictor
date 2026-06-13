"""
auth.py — username/password gate for the WC 2026 simulator.

Users are read from ``st.secrets["auth"]`` (never committed). If no ``[auth]``
section is present — e.g. running locally without a secrets file — the app runs
OPEN so local development is never blocked.

Call ``require_login()`` near the top of every page (after ``st.set_page_config``
and after the page's own CSS, so the login form inherits the styling).

To create accounts, run locally::

    python make_users.py

and paste the printed block into ``.streamlit/secrets.toml`` (local) or the
Streamlit Community Cloud "Secrets" editor (deployment).
"""
from __future__ import annotations

import streamlit as st


def _credentials_from_secrets():
    """Return (credentials_dict, auth_section) or None if auth isn't configured."""
    try:
        has_auth = "auth" in st.secrets
    except Exception:
        # No secrets.toml at all (local dev) -> run open.
        return None
    if not has_auth:
        return None

    auth = st.secrets["auth"]
    users = auth.get("credentials", {}).get("usernames", {})
    creds = {"usernames": {}}
    for username, info in users.items():
        creds["usernames"][str(username)] = {
            "name": info.get("name", str(username)),
            "email": info.get("email", ""),
            "password": info["password"],          # pre-hashed (bcrypt)
        }
    if not creds["usernames"]:
        return None
    return creds, auth


def require_login():
    """
    Render the login gate. Returns the signed-in username, or None when auth is
    not configured (open access). Halts the script via ``st.stop()`` until the
    visitor is authenticated.
    """
    bundle = _credentials_from_secrets()
    if bundle is None:
        return None                                # open access (no [auth] secret)

    try:
        import streamlit_authenticator as stauth
    except ModuleNotFoundError:
        st.warning("Login is configured but `streamlit-authenticator` is not "
                   "installed — running without authentication.")
        return None

    creds, auth = bundle
    authenticator = stauth.Authenticate(
        creds,
        auth.get("cookie_name", "wc2026_auth"),
        auth["cookie_key"],
        int(auth.get("cookie_expiry_days", 7)),
        auto_hash=False,                           # passwords are already hashed
    )

    authenticator.login(
        location="main",
        fields={"Form name": "Sign in to the WC 2026 Simulator"},
    )
    status = st.session_state.get("authentication_status")

    if status is False:
        st.error("Incorrect username or password.")
        st.stop()
    if status is None:
        st.info("Please sign in to run simulations.")
        st.stop()

    # Authenticated — show who, and a logout control in the sidebar.
    with st.sidebar:
        st.markdown(
            f"<div style='font-size:0.8rem;color:rgba(255,255,255,0.5);"
            f"margin-bottom:6px;'>Signed in as "
            f"<b style='color:rgba(255,255,255,0.85)'>"
            f"{st.session_state.get('name')}</b></div>",
            unsafe_allow_html=True,
        )
        authenticator.logout("Log out", "sidebar")

    return st.session_state.get("username")
