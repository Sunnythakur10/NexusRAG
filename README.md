# NexusRAG: Autonomous Agentic Localization Engine

**An enterprise-grade, multi-agent AI pipeline for high-fidelity manga and webtoon localization.**

NexusRAG replaces traditional, fragile generation loops with a deterministic LangGraph state machine. It utilizes a dual-model Actor-Critic architecture, semantic vector memory (RAG), and ReAct tool-calling to ensure translation accuracy, cultural adaptation, continuous character voice, and strict physical typesetting constraints.

---

## 🏗️ System Architecture

The pipeline processes dialogue panel-by-panel through a cyclic, self-healing state machine. 

### 1. LangGraph State Machine
* **Deterministic Routing:** Replaces standard `while` loops with a compiled `StateGraph`. A central `PanelState` (TypedDict) manages inputs, outputs, retry counters, and evaluation scores across all nodes.
* **Self-Healing API Limits:** Integrated `tenacity` for exponential backoff. The pipeline runs at maximum throughput and dynamically pauses/retries only when catching `HTTP 429: Too Many Requests` errors from the inference provider.

### 2. Dual-Model Actor-Critic Loop
To optimize latency and compute cost, the system utilizes two distinct LLMs:
* **The Actor (70B Parameter Model):** Handles heavy cognitive tasks—nuanced Japanese translation, cultural adaptation, and stylistic compression.
* **The Critic (8B Parameter Model):** A rapid-evaluation model that grades the Actor's output on a 1-10 scale for metrics like `meaning_preservation`, `tone`, and `naturalness`. If a node scores below a 6/10, the state machine routes the text back to the Actor for regeneration (up to 3 retries).

### 3. Continuity RAG (ChromaDB)
* **Semantic Retrieval:** Successfully translated lines are embedded and stored in a local Chroma vector database (`approved_lines` collection).
* **Dynamic Character Profiling:** The system autonomously extracts and stores character personality traits and speech styles.
* **Context Injection:** Before translating a new panel, the Continuity Agent queries ChromaDB via mathematical similarity search to pull relevant historical dialogue, ensuring character voices remain consistent across chapters. Isolated by `manga_id` to prevent memory bleed.

### 4. ReAct Typesetting Agent
* **Tool Calling:** The Typesetting node operates as a ReAct (Reason + Act) agent. It interacts with a deterministic Python tool (`check_bubble_fit`) to verify if the English text physically fits within the speech bubble's character limit.
* **Creative Compression:** If the tool returns a failure, the LLM iteratively compresses the text using aggressive contractions and synonym swapping without destroying the core emotional register, looping until the physical constraint is satisfied.

---

## ⚙️ Key Features

* **Strict Fidelity Guards:** Prompt engineering rigidly enforces Japanese-to-English verb tense alignment, first/third-person speaker perspective, and emotional intensity preservation.
* **Automated Flagging:** Panels that fail to meet the Critic's threshold after maximum retries are gracefully allowed through but flagged (`"flagged": true`) in the JSON output for human-in-the-loop editorial review.
* **Narration Routing:** Intelligent logic ensures narration panels are translated but bypass cultural and continuity nodes to save compute and prevent hallucinated dialogue formatting.
* **Version-Proof Prompting:** `SystemMessage` injection circumvents library dependency conflicts (`messages_modifier` vs `state_modifier`), making the agent immune to underlying LangGraph framework updates.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* A Groq API Key (for LLM inference)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Sunnythakur10/NexusRAG.git
cd NexusRAG
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv env
.\env\Scripts\activate

# macOS/Linux
python3 -m venv env
source env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
*(Ensure `langgraph`, `langchain`, `chromadb`, `tenacity`, and `streamlit` are included in your environment).*

**4. Configure Environment Variables**
Create a `.env` file in the root directory. **Do not commit this file.**
```bash
# Copy the example file
cp .env.example .env
```
Open `.env` and add your API key:
```text
GROQ_API_KEY=your_api_key_here
```

### Usage

**Run the CLI Pipeline:**
Execute the main script to process chapters defined in your JSON data directory.
```bash
python main.py
```

**Run the Streamlit Dashboard (Optional):**
To view the output, flag rates, and pipeline execution visually:
```bash
streamlit run ui/app.py
```

---

## 🛠️ Diagnostics & Tooling

* **Cache Reset:** If you are testing new prompts and want to force fresh inference, delete the local `lumina_cache.db` file.
* **Vector DB Inspection:** Run `python inspect_db.py` to X-ray the Chroma database, verify embedded document counts, and audit the continuity memory without running the full pipeline.

## 🗺️ Roadmap (Future Enhancements)

While the core multi-agent pipeline is stable and production-ready, future iterations will focus on:
* **Human-in-the-Loop (HITL) UI:** Expanding the Streamlit dashboard to allow editors to manually review, edit, and approve `"flagged": true` panels directly within the application before final export.
* **Vision-Language Model (VLM) Integration:** Incorporating multimodal vision models (like LLaVA or GPT-4o) to autonomously extract raw Japanese text and physical bubble coordinate limits directly from raw image files, eliminating the need for manual JSON pre-processing.
* **Multi-Language Expansion:** Abstracting the translation node prompts to dynamically support Korean (Manhwa) and Chinese (Manhua) source texts based on pipeline configuration.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sunny Thakur**
* GitHub: [@Sunnythakur10](https://github.com/Sunnythakur10)
* LinkedIn: [Insert your LinkedIn URL here]

---
*Built as an independent research project focusing on multi-agent LLM orchestration, retrieval-augmented generation, and automated natural language processing.*
