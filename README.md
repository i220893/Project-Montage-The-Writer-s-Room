# 🎬 Project Montage: The Writer's Room (Phase 1)

**The Writer's Room** is an autonomous, multi-agent creative system built using **LangGraph** and the **Model Context Protocol (MCP)**. It takes a raw human story prompt and orchestrates a team of AI agents to produce a fully structured screenplay, persistent character profiles, and cinematic character visuals.

---

## 🏗️ Architecture & Tech Stack

This project strictly adheres to a decentralized **Supervisor-Worker** design pattern. The agents themselves hold zero API keys or external hardcoded calls; they query and discover tools entirely at runtime through the centralized **MCP server**.

- **Workflow Orchestration**: LangGraph (`StateGraph`)
- **Tool Protocol**: FastMCP (`mcp[server]`, `langchain-mcp-adapters`) via `streamable_http`
- **UI & Interrupts**: Chainlit (with Human-in-the-Loop checkpoints)
- **Agent LLMs**: Groq (Llama), Google (Gemini), and native local Ollama integration (UI Switchable)
- **Image Generation**: Google Gemini (`gemini-2.0-flash-preview-image-generation` via `google-genai` SDK)
- **Semantic Memory**: FAISS + HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)

---

## 🤖 The Agent Cast

1. **ARIA (Scriptwriter)** ✍️ — Expands short prompts into multi-scene screenplays.
2. **VERA (Validator)** 🔍 — Audits the script for structural correctness (Scene Headers, Cues).
3. **CASSIDY (Character Designer)** 👥 — Extracts deep character identity profiles (Arc, traits).
4. **PIXEL (Image Synthesizer)** 🎨 — Loops over profiles to cast characters visually using Gemini.

---

## ⚙️ Installation

1. **Clone & Setup Environment**
   ```bash
   # We recommend using a virtual environment (e.g., conda or venv)
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   Copy the example environment file and add your keys:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and fill out your credentials:
   - `GROQ_API_KEY`: Required for Llama/Groq logic.
   - `GEMINI_API_KEY`: Required for Image Generation.

> **Note**: You can dynamically swap your agent brains and image generators directly from the **Chainlit Settings Gear** in the UI. No code resets required!

---

## 🧠 Stateful Memory (FAISS RAG)

This system actively uses **HuggingFace Embeddings** and a local **FAISS Vector Database** to maintain continuity!
- When you ask to write a sequel, the agents actively search past sessions via `search_memory` to seamlessly import returning characters and plots.
- The system automatically anchors past character identities so the agents do not hallucinate brand new appearances for recurring cast members.

---

## 🚀 How to Run

Because this is a true decoupled MCP architecture, the tools server acts as an independent service from the frontend application workflow. You must run them simultaneously.

### Step 1: Start the MCP Tools Server 🛠️
In your first terminal, run:
```bash
python mcp_server.py
```
*This spins up 8 unique creative tools, 2 resources, and 2 prompts on port `8100`.*

### Step 2: Launch the Writer's Room UI 🖥️
Open a second terminal, activate the same environment, and run:
```bash
chainlit run app.py -p 8001
```
*Your browser will pop open automatically. You can chat directly to start a pipeline (`generate` or `validate`).*

---

## 🗂️ Outputs

The persistent knowledge graph and final outputs all save automatically to the `./outputs` directory:

| Filename / Folder | Description |
| :--- | :--- |
| `scene_manifest.json` | The structured script. Contains acts, scene headings, dialogue arrays, and visual cues. |
| `character_db.json` | Key-value store of every named character's role, traits, visual appearance, and emotional arcs. |
| `image_assets/` | Cinematic 1024x1024 `.png` concept portraits for every defined character. |
| `faiss_index/` | Local vector index holding cross-session memory chunks. |

---

## 🛑 Human-in-the-Loop (HITL)

LangGraph's memory features (`MemorySaver`) actively control the workflow. After generation, if the **Validator node** uncovers critical structural problems with the script generation, the Graph halts. 

In the Chainlit UI, a prompt will appear letting the user view the errors and manually choose to either **Approve & Continue** or **Reject & Retry**. 
