# Deploying the WC 2026 Simulator

The app runs on **Streamlit Community Cloud**. Access is controlled by a simple
**password gate** with two levels:

- **Admin** — enter the admin password for full access to every page and control.
- **Visitor** — choose *Continue as visitor* for read-only access, limited to the
  **WC2026 UI** and **About Me** pages.

Many people can use it concurrently — each browser gets its own isolated session,
and nothing is written to disk per user.

---

## 1. Push to GitHub

```bash
# create an empty repo on github.com first (e.g. wc2026-predictor), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

The real `.streamlit/secrets.toml` is git-ignored and will **not** be pushed.

## 2. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick your repo, branch `main`, **main file `wc2026_ui.py`**.
3. Open **Advanced settings → Python version → 3.12**. This matters: the
   scientific stack (statsmodels / scipy / numpy) has no stable wheels for
   Python 3.14 yet, and the build will fail on it.
4. Deploy. The first build installs `requirements.txt` and takes a few minutes.

## 3. (Optional) Set the admin password via Secrets

The admin password is **not** stored in plaintext — `auth.py` checks input
against a SHA-256 hash, so the app works out of the box with the built-in
default. To use your own password without touching code, add this in the app's
**Settings → Secrets** (see `.streamlit/secrets.toml.example`):

```toml
admin_password = "your-admin-password"
```

Share the app URL with everyone. Give the admin password only to people who need
full access; everyone else clicks **Continue as visitor**.

---

## Notes & known limitations

- **Python version** — pin **3.12** (see step 2). 3.13 also works; 3.14 does not.
- **LLM / Expert Analysis** needs a reachable Ollama server, which Streamlit
  Cloud does not provide — the AI Analyst tab will error there (expected). The
  scraping deps are commented out in `requirements.txt` to keep the build fast;
  uncomment them only if you want the Scrape/Library tabs.
- **Live data is resilient** — ELO, fixtures, results and odds are fetched live,
  but every loader falls back to the committed `data/cache/*.parquet` if a source
  is unreachable from the cloud (e.g. eloratings.net blocking a datacenter IP),
  so the app boots even when a fetch fails.
- **Resource limits** — the free tier gives ~1 GB RAM and sleeps after
  inactivity (wakes on next visit). Fine for a small group running simulations.
