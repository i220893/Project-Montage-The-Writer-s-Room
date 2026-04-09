# ============================================================
# config.py — Central configuration for The Writer's Room
#
# HOW TO SWAP MODELS:
#   Set ACTIVE_PROVIDER in your .env to:
#     "groq"    → Groq Llama 3.x (default)
#     "google"  → Gemini via langchain-google-genai
#     "ollama"  → Local Ollama (future)
# ============================================================

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SETTINGS_FILE = Path("outputs/settings.json")

def get_dynamic_setting(key: str, default: str = "") -> str:
    """Read a setting dynamically so Chainlit UI changes reflect in MCP server instantly."""
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if key in settings:
                return settings[key]
        except Exception:
            pass
    return os.getenv(key, default)

# ── Keys (usually static) ────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str  = os.getenv("GEMINI_API_KEY", "")

# ── Endpoints (usually static) ───────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MCP_SERVER_URL: str  = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
MCP_SERVER_NAME: str = "writers_room"
MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8000"))

# ── Embeddings & FAISS ───────────────────────────────────────────────────────
EMBED_MODEL: str      = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH: str = "./outputs/faiss_index"
CHUNK_SIZE: int        = 1000
CHUNK_OVERLAP: int     = 150

# ── Output Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR: str       = "./outputs"
IMAGE_ASSETS_DIR: str = "./outputs/image_assets"
SCENE_MANIFEST: str   = "./outputs/scene_manifest.json"
CHARACTER_DB: str     = "./outputs/character_db.json"


def get_llm(temperature: float = 0.7):
    """
    Factory function — returns the correct LangChain chat model based on
    dynamically loaded settings from the Chainlit UI or .env.
    """
    active_provider = get_dynamic_setting("ACTIVE_PROVIDER", "ollama")

    if active_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=get_dynamic_setting("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )

    elif active_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=get_dynamic_setting("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
        )

    elif active_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=get_dynamic_setting("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown ACTIVE_PROVIDER='{active_provider}'. "
            "Choose from: groq | google | ollama"
        )


def model_info() -> str:
    """Return a human-readable string describing the active model config."""
    active_provider = get_dynamic_setting("ACTIVE_PROVIDER", "ollama")
    if active_provider == "groq":
        return f"Groq / {get_dynamic_setting('GROQ_MODEL', 'llama-3.1-8b-instant')}"
    elif active_provider == "google":
        return f"Google Gemini / {get_dynamic_setting('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')}"
    elif active_provider == "ollama":
        return f"Ollama (local) / {get_dynamic_setting('OLLAMA_MODEL', 'qwen2.5:7b-instruct-q4_K_M')} @ {OLLAMA_BASE_URL}"
    return active_provider
