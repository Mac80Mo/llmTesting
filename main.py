import os
from dotenv import load_dotenv

# --------------------------------------------------------
# 1) .env einlesen
#    - Die Datei ".env" muss sich im selben Verzeichnis wie main.py befinden.
#    - In .env steht genau:
#        HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>
# --------------------------------------------------------
load_dotenv()  # Liest Umgebungsvariablen aus der Datei .env

# --------------------------------------------------------
# 2) Token aus Umgebungsvariablen auslesen
# --------------------------------------------------------
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    # Abbrechen, falls kein Token gesetzt ist
    raise ValueError(
        "♨️ Kein Token gefunden. Bitte lege eine Datei '.env' an mit:\n"
        "   HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>"
    )

# --------------------------------------------------------
# 3) Wichtige Importe für LangChain
#    - HuggingFaceEndpoint: für die Verbindung zum Hugging Face Inference-API
#    - LLMChain und PromptTemplate: für einfache Prompt-Workflows
# --------------------------------------------------------
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------------
# 4) LLM-Instanz erstellen
#
#    Wir verwenden hier das Modell "HuggingFaceH4/zephyr-7b-beta", da es als
#    text-generation-Inference-Endpoint verfügbar ist.
#
#    Parameter:
#      - repo_id="HuggingFaceH4/zephyr-7b-beta"  
#          • exakte Modellbezeichnung auf Hugging Face  
#      - huggingfacehub_api_token=hf_token          
#          • Übergibt deinen API-Key für die Autorisierung  
#      - provider="auto"                            
#          • LangChain wählt automatisch den passenden Inference-Provider  
#      - task="text-generation"                     
#          • Pipeline-Typ: Causal Language Model (Text-Generierung)  
#      - temperature=0.7                             
#          • Steuert die Kreativität (0.0 bis 1.0)  
#      - max_new_tokens=100                          
#          • Maximale Anzahl neu generierter Tokens pro Anfrage  
# --------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=hf_token,
    provider="auto",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=50,
    stop=["\nFrage:"]
)

# --------------------------------------------------------
# 5) Direkter Prompt-Abruf (ohne Chain)
# --------------------------------------------------------
#prompt_text = "Erkläre den Dopplereffekt in einfachen Worten."
# Mit .invoke(prompt) rufen wir das Modell auf und erhalten die generierte Antwort
#response = llm.invoke(prompt_text)

#print("=== Direkte LLM-Antwort ===")
#print(response)

# --------------------------------------------------------
# 6) LLMChain mit PromptTemplate
#
#    Wir bauen eine einfache Prompt-Vorlage, die später mit einer Variable {frage}
#    gefüllt wird. LLMChain verbindet das LLM und das Template zu einem Workflow.
# --------------------------------------------------------
template = PromptTemplate(
    template="Frage: {frage}\nAntwort:",
    input_variables=["frage"]
)
chain = LLMChain(llm=llm, prompt=template)

# Beispiel-Eingabe für die Chain
frage = "Warum ist der Himmel blau?"
# Mit .invoke({"frage": frage}) setzen wir den Wert für {frage} ein und führen die Anfrage aus
ausgabe = chain.invoke({"frage": frage})

print("\n=== LLMChain-Antwort ===")
print(ausgabe)
