"""
make_users.py — generate a secrets.toml [auth] block with bcrypt-hashed passwords.

Run locally:

    python make_users.py

Then paste the printed block into:
  • .streamlit/secrets.toml         (for local testing), or
  • Streamlit Community Cloud  →  app  →  Settings  →  Secrets  (for deployment).

Never commit real passwords or the generated secrets.toml.
"""
import getpass
import secrets as pysecrets

import streamlit_authenticator as stauth


def main():
    print("Create login accounts for the WC 2026 simulator.")
    print("Press Enter on an empty username when you're done.\n")

    users = []
    while True:
        username = input("Username: ").strip()
        if not username:
            break
        name = input("  Display name: ").strip() or username
        email = input("  Email (optional): ").strip()
        pw = getpass.getpass("  Password: ")
        pw2 = getpass.getpass("  Confirm : ")
        if not pw:
            print("  x Empty password, skipping.\n")
            continue
        if pw != pw2:
            print("  x Passwords didn't match, try again.\n")
            continue
        hashed = stauth.Hasher.hash(pw)
        users.append((username, name, email, hashed))
        print("  + added\n")

    if not users:
        print("No users created.")
        return

    cookie_key = pysecrets.token_hex(16)
    sep = "=" * 72
    print("\n" + sep)
    print("Paste everything below into .streamlit/secrets.toml or Cloud Secrets:")
    print(sep + "\n")
    print("[auth]")
    print('cookie_name = "wc2026_auth"')
    print(f'cookie_key = "{cookie_key}"   # random signing key — keep secret')
    print("cookie_expiry_days = 7\n")
    for username, name, email, hashed in users:
        print(f"[auth.credentials.usernames.{username}]")
        print(f'name = "{name}"')
        print(f'email = "{email}"')
        print(f'password = "{hashed}"\n')


if __name__ == "__main__":
    main()
