# 🎬 Project Montage: The Writer's Room (Phase 2)

**The Writer's Room** is an autonomous, multi-agent creative system built using **LangGraph** and the **Model Context Protocol (MCP)**. It takes a raw human story prompt and orchestrates a team of AI agents to produce a fully structured screenplay, persistent character profiles, and cinematic character visuals.

---

## 🏗️ Architecture & Tech Stack

This project strictly adheres to a decentralized **Supervisor-Worker** design pattern. The agents themselves hold zero API keys or external hardcoded calls; they query and discover tools entirely at runtime through the centralized **MCP server**.

- **Workflow Orchestration**: LangGraph (`StateGraph` w/ strictly typed Redux schema)
- **Tool Protocol**: FastMCP (`mcp[server]`, `langchain-mcp-adapters`) via `streamable_http`
- **UI & Interrupts**: Chainlit (with Human-in-the-Loop checkpoints and dynamic system settings)
- **Agent LLMs**: Qwen-2.5 7B via Ollama (Local), Groq (Llama), Google (Gemini)
- **Image Generation**: Cloud via Pollinations.ai (Free Flux, Hardware-Agnostic default) with support for Local ComfyUI (GPU) and Gemini (Paid API fallback)
- **Semantic Memory**: FAISS + HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Resource Management**: Custom VRAM orchestration layer for concurrent 8GB GPU pipelines

---

## 🤖 The Agent Cast

1. **ARIA (Scriptwriter)** ✍️ — Expands short prompts into multi-scene screenplays.
2. **VERA (Validator)** 🔍 — Audits the script for structural correctness (Scene Headers, Cues).
3. **CASSIDY (Character Designer)** 👥 — Extracts deep character identity profiles (Arc, traits).
4. **PIXEL (Image Synthesizer)** 🎨 — Loops over profiles to cast characters visually via Flux.

---

## ⚙️ Installation

1. **Clone & Setup Environment**
   ```bash
   # We recommend using a virtual environment (e.g., conda or venv)
   pip install -r requirements.txt
   ```

2. **Configure API Keys (Optional Fallbacks)**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Add fallback keys if you plan to use Cloud APIs instead of pure Local generation:
   - `GROQ_API_KEY`: Required for Llama/Groq text generation.
   - `GEMINI_API_KEY`: Required only if you select Gemini as your image fallback (Pollinations.ai is 100% free and requires no key).

3. *(Optional)* **ComfyUI Integration**
   - Ensure a local instance of ComfyUI is running on port `8188`.
   - Install the `flux-2-klein-9b-fp8` model architecture in your models directory.

> **Note**: You can dynamically swap your agent brains, models, and image generators (Local vs Cloud) directly from the **Chainlit Settings Gear** in the UI. No code restarts required!

---

## 🧠 Stateful Memory & VRAM Engine

### Context Continuity
This system actively uses **HuggingFace Embeddings** and a local **FAISS Vector Database** to maintain continuity!
- Fast cross-session memory retrieval via `search_memory` naturally imports returning characters and plotlines.
- Ensures the image synthesis team does not hallucinate brand new appearances for recurring cast members.

### Thread-Safe Hardware Orchestration
A custom `vram_manager` module seamlessly rotates heavy local models to fit inside constrained 8GB VRAM budgets (like an RTX 5060):
- Intercepts Ollama payloads to evict Qwen from VRAM automatically.
- Bridges the ComfyUI `/free` REST endpoints to dump FLUX arrays prior to script analysis.
- Multi-threaded execution hooks shield ASGI runtime loops from `asyncio` collision deadlocks during heavy inference calls.

---

## 🚀 How to Run

Because this is a true decoupled MCP architecture, the tools server acts as an independent service from the frontend application workflow. You must run them simultaneously across two identical virtual environments.

### Step 1: Start the MCP Tools Server 🛠️
In your first terminal, run:
```bash
python mcp_server.py
```
*This spins up 8 unique creative tools, 2 resources, and 2 prompts asynchronously on port `8000`.*

### Step 2: Launch the Writer's Room UI 🖥️
Open a second terminal, activate the same environment, and run:
```bash
chainlit run app.py --port 8001
```
*Your browser will pop open automatically. The server performs a handshake and uses Pollinations.ai by default, with automatic failovers between ComfyUI and Gemini based on your settings.*

---

## 🗂️ Outputs

The persistent knowledge graph and final outputs all auto-save to the `./outputs` directory:

| Filename / Folder | Description |
| :--- | :--- |
| `scene_manifest.json` | The structured script. Contains acts, scene headings, dialogue arrays, and visual cues. |
| `character_db.json` | Key-value store of every named character's role, traits, visual appearance, and emotional arcs. |
| `image_assets/` | Cinematic 1024x1024 `.png` concept portraits for every defined character. |
| `faiss_index/` | Local vector index holding cross-session memory chunks. |

---

## 🛑 Human-in-the-Loop (HITL)

LangGraph's strict typed memory states feature actively controls the workflow. After generation, if the **Validator node** uncovers structural problems with the script generation, the Graph safely dumps a persistent checkpoint and halts execution stream cleanly. 

In the Chainlit UI, a prompt will securely appear letting the user inspect the traceback block, giving explicit controls to either **Approve & Continue** (resolving to character synthesis) or **Reject & Retry** (striking the session).
