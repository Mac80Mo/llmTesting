# LLMHuggingFace-Testsuite

Eine Sammlung von Skripten und Tools, um verschiedene große Sprachmodelle (LLMs) über die Hugging Face Inference-API systematisch zu testen. Mit diesem Projekt kannst du:

- **Latenz (Antwortgeschwindigkeit)** messen  
- **Token-Verbrauch** (Input/Output) zählen  
- **Antwortqualitätsmetriken** (z. B. Toxizität) automatisiert prüfen  
- Verschiedene **Modelle & Parameter** vergleichen

> **Wichtig:** Für alle Skripte wird ein gültiger Hugging Face API‐Token benötigt. Lege eine Datei `.env` an und trage deinen Token wie folgt ein:  
> ```text
> HUGGINGFACEHUB_API_TOKEN=hf_<dein_token>
> ```

---


## Projektbeschreibung

In modernen NLP-Anwendungen spielt die Auswahl und Konfiguration eines großen Sprachmodells (LLM) eine entscheidende Rolle für Performance, Kosten und inhaltliche Qualität. Dieses Projekt bietet eine **modulare Testumgebung**, um LLMs über die Hugging Face Inference-API zu evaluieren.  

Du kannst damit:

- **Unterschiedliche Modelle** (z. B. `gpt2`, `HuggingFaceH4/zephyr-7b-beta` usw.) direkt aus HuggingFace testen.  
- **Antwortgeschwindigkeit (Latenz)** messen und mit verschiedenen Parametern vergleichen.  
- **Token‐Verbrauch** (Anzahl der Input- und Output‐Tokens) protokollieren, um Kostenabschätzungen zu erleichtern.  
- **Toxizität und unangemessene Inhalte** automatisiert erkennen (z. B. toxische, beleidigende oder diskriminierende Antworten).  
- Ein **Readme‐Gerüst** für eigene Erweiterungen oder Tests verwenden.

Dieses Repository richtet sich an Entwickler:innen, die den **Leistungs- und Qualitätsvergleich** von LLM-Endpunkten automatisieren möchten (z. B. in Forschung, Prototyping oder Produktionsmonitoring).

---

## Voraussetzungen

- **Python 3.10 oder 3.11** (frühere Versionen können auf Fehlermeldungen stoßen)  
- Ein **Hugging Face API-Token** (hf_…)  
- Git (zum Klonen des Projekts und Pushen von Anpassungen)  
- (Optional) Eine GPU, falls du sehr große Modelle lokal oder über HF GPU-Endpunkte testen möchtest  

---

