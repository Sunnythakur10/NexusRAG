# 🎌 NexusRAG: Autonomous Agentic Localization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_supported.svg)](https://nexusrag-2kbzprgxw77xsyzwars8tc.streamlit.app)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-orange)
![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Memory-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**An enterprise-grade, multi-agent AI pipeline for high-fidelity manga and webtoon localization.**

NexusRAG replaces fragile, linear prompt chains with a deterministic **LangGraph state machine**. It utilizes a dual-model Actor-Critic architecture, semantic vector memory (RAG), and ReAct tool-calling to ensure translation accuracy, cultural adaptation, continuous character voice, and strict physical typesetting constraints.

---

> **The Problem:** Standard LLM translation workflows suffer from context amnesia, hallucinated formatting, and tone inconsistency across chapters. 
> 
> **The Solution (NexusRAG):** A cyclic, self-healing pipeline that translates, critiques its own work, retrieves historical character profiles via ChromaDB, and mechanically compresses text to fit physical speech bubbles.
---

## 📑 Table of Contents
- [System Architecture](#️-system-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Diagnostics & Tooling](#️-diagnostics--tooling)
- [Roadmap](#-roadmap-future-enhancements)
- [License & Author](#-license)

---

## 🏗️ System Architecture

The pipeline processes dialogue panel-by-panel through a cyclic, self-healing state machine. 

### 1. LangGraph State Machine
* **Deterministic Routing:** Replaces standard `while` loops with a compiled `StateGraph`. A central `PanelState` (TypedDict) manages inputs, outputs, retry counters, and evaluation scores across all nodes.
* **Self-Healing API Limits:** Integrated `tenacity` for exponential backoff. The pipeline runs at maximum throughput and dynamically pauses/retries only when catching `HTTP 429: Too Many Requests` errors from the inference provider.

### 2. Dual-Model Actor-Critic Loop
To optimize latency and compute cost, the system utilizes two distinct LLMs:
* **The Actor (70B Model):** Handles heavy cognitive tasks—nuanced Japanese translation, cultural adaptation, and stylistic compression.
* **The Critic (8B Model):** A rapid-evaluation model that grades the Actor's output on a 1-10 scale for metrics like `meaning_preservation`, `tone`, and `naturalness`. If a node scores below a 6/10, the state machine routes the text back to the Actor for regeneration (up to 3 retries).

### 3. Continuity RAG (ChromaDB)
* **Semantic Retrieval:** Successfully translated lines are embedded and stored in a local Chroma vector database (`approved_lines`).
* **Dynamic Character Profiling:** Autonomously extracts and stores character personality traits and speech styles.
* **Context Injection:** Before translating a new panel, the Continuity Agent queries ChromaDB via mathematical similarity search to pull relevant historical dialogue, ensuring character voices remain consistent across chapters. Isolated by `manga_id` to prevent memory bleed.

### 4. ReAct Typesetting Agent
* **Tool Calling:** Operates as a ReAct (Reason + Act) agent interacting with a deterministic Python tool (`check_bubble_fit`).
* **Creative Compression:** If the text exceeds the physical speech bubble limit, the LLM iteratively compresses the text using aggressive contractions and synonym swapping without destroying the core emotional register.

---

## ⚙️ Key Features

- **Strict Fidelity Guards:** Prompt engineering rigidly enforces Japanese-to-English verb tense alignment, first/third-person speaker perspective, and emotional intensity.
- **Automated Flagging:** Panels that fail to meet the Critic's threshold after maximum retries are gracefully allowed through but flagged (`"flagged": true`) for human-in-the-loop editorial review.
- **Smart Narration Routing:** Intelligent logic ensures narration panels bypass cultural and continuity nodes to save compute and prevent hallucinated dialogue formatting.
- **Version-Proof Prompting:** `SystemMessage` injection circumvents library dependency conflicts, making the agent immune to underlying LangGraph framework updates.
- **Full Observability:** Integrated with LangSmith for real-time tracing of the LangGraph state machine, allowing granular debugging of token usage, latency, and agent routing logic.

---

## 📂 Repository Structure

```text
NexusRAG/
├── agents/
│   ├── translation_agent.py   # Step 0/1: Batch & Panel Translation
│   ├── cultural_agent.py      # Step 2: Localization & Idiom Adaptation
│   ├── continuity_agent.py    # Step 3: RAG-based Voice Consistency
│   └── typesetting_agent.py   # Step 4: ReAct Tool-Calling Compression
├── memory/
│   └── vector_store.py        # ChromaDB initialization and CRUD ops
├── utils/
│   ├── rate_limiter.py        # Tenacity exponential backoff wrappers
│   └── groq_client.py         # LLM instantiation
├── ui/
│   └── app.py                 # (Optional) Streamlit Dashboard
├── main.py                    # LangGraph Compilation & Execution CLI
├── inspect_db.py              # Diagnostic tool for ChromaDB
├── requirements.txt           
└── .gitignore                 
🚀 Getting Started
Prerequisites
Python 3.10+

A Groq API Key (for LLM inference)

Installation
1. Clone the repository

Bash
git clone [https://github.com/Sunnythakur10/NexusRAG.git](https://github.com/Sunnythakur10/NexusRAG.git)
cd NexusRAG
2. Create and activate a virtual environment

Bash
# Windows
python -m venv env
.\env\Scripts\activate

# macOS/Linux
python3 -m venv env
source env/bin/activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory. Do not commit this file.

Bash
# Copy the example file
cp .env.example .env
Open .env and add your API key:

Plaintext
GROQ_API_KEY=your_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="NexusRAG"

Usage
Run the CLI Pipeline:
Execute the main script to process chapters defined in your JSON data directory.

Bash
python main.py
Run the Streamlit Dashboard (Optional):
To view the output, flag rates, and pipeline execution visually:

Bash
streamlit run ui/app.py
🛠️ Diagnostics & Tooling
Cache Reset: To force fresh inference and bypass LangChain SQLite caching, delete lumina_cache.db.

Vector DB Inspection: Run python inspect_db.py to X-ray the Chroma database, verify embedded document counts, and audit the continuity memory without running the full pipeline.

🗺️ Roadmap (Future Enhancements)
[ ] Human-in-the-Loop (HITL) UI: Expand the Streamlit dashboard to allow editors to manually review, edit, and approve "flagged": true panels directly within the application before final export.

[ ] Vision-Language Model (VLM) Integration: Incorporate multimodal vision models (e.g., LLaVA or GPT-4o) to autonomously extract raw Japanese text and physical bubble coordinate limits directly from raw image files, eliminating manual JSON pre-processing.

[ ] Multi-Language Expansion: Abstract translation prompts to dynamically support Korean (Manhwa) and Chinese (Manhua) source texts.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Sunny Thakur

GitHub: @Sunnythakur10

Built as an independent research project focusing on multi-agent LLM orchestration, retrieval-augmented generation, and automated natural language processing.