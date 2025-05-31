import os
import time
import pandas as pd
from dotenv import load_dotenv

from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------------
# 1) .env einlesen
#    - Die Datei ".env" muss im selben Verzeichnis wie main.py liegen.
#    - In .env steht exakt:
#        HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>
# --------------------------------------------------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError(
        "♨️ Kein Token gefunden. Lege eine `.env`-Datei an mit:\n"
        "   HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>"
    )

# --------------------------------------------------------
# 2) Modell‐ID & Tokenizer initialisieren
#    - Wir nutzen dasselbe Modell wie zuvor: HuggingFaceH4/zephyr-7b-beta
# --------------------------------------------------------
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def count_tokens(text: str) -> int:
    """
    Zählt, wie viele Tokens im übergebenen Text durch den HF‐Tokenizer erzeugt werden.
    """
    return len(tokenizer.encode(text, add_special_tokens=True))


# --------------------------------------------------------
# 3) Funktion: LLM-Aufruf mit Latenz‐ und Token‐Messung
# --------------------------------------------------------
def measure_llm(
    llm: HuggingFaceEndpoint,
    prompt: str,
    stop_sequences: list[str] = None
) -> dict:
    """
    Führt einen LLM-Aufruf aus und misst dabei:
      - die Antwortzeit (Latenz) in Sekunden
      - die Anzahl der Input- und Output-Tokens

    Parameter:
      - llm: die HuggingFaceEndpoint-Instanz
      - prompt: der fertige Prompt-String
      - stop_sequences: Liste von Stoppsequenzen (optional)

    Rückgabe:
      {
        "latency": float,        # gemessene Zeit in Sekunden
        "input_tokens": int,     # Anzahl der Tokens im Prompt
        "output_tokens": int,    # Anzahl der Tokens in der generierten Antwort
        "response_text": str     # der eigentliche generierte Text
      }
    """
    # Anzahl der Tokens im Prompt zählen
    n_input_tokens = count_tokens(prompt)

    # Stop-Parameter anfügen, falls vorhanden
    invocation_kwargs = {}
    if stop_sequences:
        invocation_kwargs["stop"] = stop_sequences

    # Latenz messen
    start = time.time()
    response = llm.invoke(prompt, **invocation_kwargs)
    latency = time.time() - start

    generated_text = response
    # Anzahl der Tokens in der Antwort zählen
    n_output_tokens = count_tokens(generated_text)

    return {
        "latency": latency,
        "input_tokens": n_input_tokens,
        "output_tokens": n_output_tokens,
        "response_text": generated_text
    }


# --------------------------------------------------------
# 4) PromptTemplate und LLMChain definieren
#    - Mit zusätzlicher Instruktion für eine einzelne, sachliche Antwort auf Deutsch
# --------------------------------------------------------
template = PromptTemplate(
    template=(
        "Gib in einem einzigen, klaren Satz und auf Deutsch eine sachliche Antwort.\n"
        "Frage: {frage}\n"
        "Antwort:"
    ),
    input_variables=["frage"]
)

# --------------------------------------------------------
# 5) Parameter‐Sets vorbereiten
#    - Jede Kombination aus temperature und max_new_tokens testen
#    - Hinweis: temperature muss strikt positiv sein, daher kein 0.0
# --------------------------------------------------------
parameter_list = [
    {"temperature": 0.1, "max_new_tokens": 50},
    {"temperature": 0.1, "max_new_tokens": 100},
    {"temperature": 0.3, "max_new_tokens": 50},
    {"temperature": 0.3, "max_new_tokens": 100},
    {"temperature": 0.7, "max_new_tokens": 50},
    {"temperature": 0.7, "max_new_tokens": 100},
]

# --------------------------------------------------------
# 6) Messergebnisse sammeln
# --------------------------------------------------------
results = []
frage = "Warum ist der Himmel blau?"
prompt_string = template.format(frage=frage)

for params in parameter_list:
    # --------------------------------------------------------
    # 6.1) LLM-Instanz erstellen (HuggingFaceEndpoint) mit den jeweiligen Parametern
    #    - stop=["\nFrage:", "\nQuestion:"] sorgt dafür, dass das Modell stoppt,
    #      sobald es eine neue "Frage:" oder "Question:" generieren will
    # --------------------------------------------------------
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        huggingfacehub_api_token=hf_token,
        provider="auto",
        task="text-generation",
        temperature=params["temperature"],
        max_new_tokens=params["max_new_tokens"],
        stop=["\nFrage:", "\nQuestion:"]
    )

    # --------------------------------------------------------
    # 6.2) LLM-Aufruf + Messung
    # --------------------------------------------------------
    measurement = measure_llm(
        llm=llm,
        prompt=prompt_string,
        stop_sequences=["\nFrage:", "\nQuestion:"]
    )

    # --------------------------------------------------------
    # 6.3) Messergebnis protokollieren
    # --------------------------------------------------------
    results.append({
        "temperature": params["temperature"],
        "max_new_tokens": params["max_new_tokens"],
        "latency_sec": round(measurement["latency"], 3),
        "input_tokens": measurement["input_tokens"],
        "output_tokens": measurement["output_tokens"],
        "response_text": measurement["response_text"]
    })

# --------------------------------------------------------
# 7) Ergebnisse in DataFrame umwandeln und ausgeben
# --------------------------------------------------------
df = pd.DataFrame(results)

print("\n=== Messergebnisse als Tabelle ===")
# Tabelle ohne tabulate-Dependency ausgeben
print(
    df[["temperature", "max_new_tokens", "latency_sec", "input_tokens", "output_tokens"]]
    .to_string(index=False)
)

# --------------------------------------------------------
# 8) Detaillierte Ausgabe der vollständigen Antworten
# --------------------------------------------------------
for row in results:
    print(f"\n--- Parameter-Set: temp={row['temperature']}, max_new_tokens={row['max_new_tokens']} ---")
    print("Antwort:", row["response_text"])
