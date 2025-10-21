"""
Helper script to generate and save a FYERS access token.

Usage:
  1) Ensure your .env has FYERS_APP_ID, FYERS_SECRET_KEY, and FYERS_REDIRECT_URI
  2) Run: python tools/generate_fyers_token.py
  3) Log in via the opened URL, copy the redirect URL, paste it back when prompted
  4) The script will write FYERS_ACCESS_TOKEN to your .env
"""
import os
import sys
import webbrowser
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

try:
    from fyers_apiv3 import fyersModel
except Exception as e:
    print("Missing dependency: fyers-apiv3. Install with: pip install fyers-apiv3")
    raise


def load_required_env() -> dict:
    load_dotenv()
    app_id = os.getenv("FYERS_APP_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri")

    missing = [name for name, val in [
        ("FYERS_APP_ID", app_id),
        ("FYERS_SECRET_KEY", secret_key),
    ] if not val]

    if missing:
        print(f"Error: Missing required env vars: {', '.join(missing)}. Edit your .env and try again.")
        sys.exit(1)

    return {
        "client_id": app_id,
        "secret_key": secret_key,
        "redirect_uri": redirect_uri,
    }


def write_access_token_to_env(access_token: str, env_path: str = ".env") -> None:
    # Update FYERS_ACCESS_TOKEN in .env (append if not present)
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    key_found = False
    for i, line in enumerate(lines):
        if line.startswith("FYERS_ACCESS_TOKEN="):
            lines[i] = f"FYERS_ACCESS_TOKEN={access_token}\n"
            key_found = True
            break

    if not key_found:
        lines.append(f"FYERS_ACCESS_TOKEN={access_token}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    env = load_required_env()

    session = fyersModel.SessionModel(
        client_id=env["client_id"],
        secret_key=env["secret_key"],
        redirect_uri=env["redirect_uri"],
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    print("\nOpen this URL to authorize (we will try to open it for you):\n")
    print(auth_url)
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    redirect_response = input("\nPaste the FULL redirect URL after login and press Enter:\n> ").strip()
    if not redirect_response:
        print("No URL entered. Exiting.")
        sys.exit(1)

    # Extract auth_code from the redirect URL
    try:
        parsed = urlparse(redirect_response)
        query = parse_qs(parsed.query)
        auth_code_list = query.get("auth_code") or query.get("auth_code")
        if not auth_code_list:
            # Some flows might return as fragment; fallback to manual parse
            # e.g., ...#auth_code=XXX&state=...
            frag = parse_qs(parsed.fragment)
            auth_code_list = frag.get("auth_code")

        if not auth_code_list:
            raise ValueError("auth_code not found in URL")

        auth_code = auth_code_list[0]
        print(f"Found auth_code: {auth_code}")
    except Exception as e:
        print(f"Failed to parse auth_code from redirect URL: {e}")
        sys.exit(1)

    session.set_token(auth_code)
    token_response = session.generate_token()

    if not isinstance(token_response, dict) or "access_token" not in token_response:
        print("Failed to generate access token. Response:")
        print(token_response)
        sys.exit(1)

    access_token = token_response["access_token"]
    print("\nAccess token generated successfully.\n")
    print(access_token)

    write_access_token_to_env(access_token)
    print("\nSaved FYERS_ACCESS_TOKEN to .env. You can now run the system.")


if __name__ == "__main__":
    main()


