import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

# 1) .env einlesen
load_dotenv()  # lädt HUGGINGFACEHUB_API_TOKEN aus .env

# 2) Token auslesen
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError(
        "♨️ Kein Token gefunden. Lege eine .env an mit\n"
        "   HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>"
    )

# 3) Hugging Face API‐Client initialisieren
api = HfApi()

try:
    # 4) whoami() benutzt deinen Token, um die Account‐Infos abzurufen
    user_info = api.whoami(token=hf_token)
    print("✅ Token ist gültig! Folgende Account-Infos kamen zurück:")
    # Nur ausgewählte Felder anzeigen, damit es nicht zu lang wird
    print(f"   Benutzername: {user_info.get('name')}")
    print(f"   E-Mail:       {user_info.get('email', '<nicht gesetzt>')}")
    print(f"   Username:     {user_info.get('username')}")
    print(f"   User ID:      {user_info.get('userId')}")
except HTTPError as e:
    # 5) HTTP‐Fehler (z. B. 401 Unauthorized)
    print("❌ Token ungültig oder kein Zugriff möglich.")
    print("   HTTP‐Fehler:", e)
except Exception as e:
    # 6) Alle anderen Fehler (Netzwerk, JSON‐Parsing o. Ä.)
    print("❌ Ein unerwarteter Fehler ist aufgetreten:")
    print("   ", e)
