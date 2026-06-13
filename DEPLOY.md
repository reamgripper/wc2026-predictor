# Deploying the WC 2026 Simulator

The app is ready for **Streamlit Community Cloud** with **username/password login**.
Multiple people can log in and simulate concurrently — each browser gets its own
isolated session, and nothing is written to disk per user.

---

## 1. Create login accounts (local, one-time)

```bash
pip install streamlit-authenticator==0.4.2
python make_users.py
```

Answer the prompts for each person. It prints an `[auth]` block with
**bcrypt-hashed** passwords. Copy that whole block — you'll paste it in step 4.

> Test locally first: paste the block into `.streamlit/secrets.toml`
> (git-ignored), run `streamlit run wc2026_ui.py`, and confirm the login works.
> With no `secrets.toml`, the app runs open — that's expected for local dev.

## 2. Push to GitHub

```bash
# create an empty repo on github.com first (e.g. wc2026-simulator), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

The real `secrets.toml` is git-ignored and will **not** be pushed — correct.

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick your repo, branch `main`, **main file `wc2026_ui.py`**.
3. Deploy. The first build installs `requirements.txt` and may take a few minutes.

## 4. Add the secrets (this is the login)

In the app's **Settings → Secrets**, paste the `[auth]` block from step 1, then
save. The app reboots and the login gate goes live. Example shape (see
`.streamlit/secrets.toml.example`):

```toml
[auth]
cookie_name = "wc2026_auth"
cookie_key = "<random hex>"
cookie_expiry_days = 7

[auth.credentials.usernames.alice]
name = "Alice"
email = "alice@example.com"
password = "$2b$12$...bcrypt-hash..."
```

Share the app URL + each person's username/password. Done.

---

## Adding or removing users later

Re-run `python make_users.py`, then update the Secrets editor on Streamlit Cloud
(keep the same `cookie_key` so existing sessions stay valid). No redeploy needed.

## Notes & known limitations

- **LLM / Expert Analysis** needs a reachable Ollama server, which Streamlit
  Cloud does not provide — the AI Analyst tab will error there (expected/accepted).
  The scraping deps are commented out in `requirements.txt` to keep the build
  fast and reliable; uncomment them only if you want the Scrape/Library tabs.
- **Data freshness** — ELO, fixtures, results and odds are fetched live and
  cached. The committed `data/cache/*.parquet` just makes the first load instant;
  it refreshes on its normal TTL.
- **Resource limits** — the free tier gives ~1 GB RAM and sleeps after
  inactivity (wakes on next visit). Fine for a small group running simulations.
