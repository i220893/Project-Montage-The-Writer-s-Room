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

# ── ComfyUI (local image generation) ─────────────────────────────────────────
COMFYUI_BASE_URL: str   = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
COMFYUI_WORKFLOW: str   = os.getenv("COMFYUI_WORKFLOW", "./comfyui/image_flux2_text_to_image_9b.json")
COMFYUI_TIMEOUT: int    = int(os.getenv("COMFYUI_TIMEOUT", "300"))  # seconds

# ── Image generation backend ─────────────────────────────────────────────────
# "comfyui"      — Local GPU via ComfyUI + Flux 9B (best quality, needs GPU)
# "pollinations" — Free cloud API, no key/GPU required (recommended default)
# "gemini"       — Google Gemini image API (paid, requires GEMINI_API_KEY)
IMAGE_GEN_BACKEND: str     = os.getenv("IMAGE_GEN_BACKEND", "pollinations")
POLLINATIONS_MODEL: str    = os.getenv("POLLINATIONS_MODEL", "flux")

import logging as _logging
_config_log = _logging.getLogger(__name__)


def _resolve_text_provider() -> str:
    """
    Resolve which text-generation provider to actually use.

    The ComfyUI + Ollama combination causes an irrecoverable RAM conflict on
    systems with ≤ 16 GB RAM:
      - ComfyUI loads Flux 9B → holds ~4–5 GB of system RAM in its Python heap.
      - Python never returns heap pages to the OS, so free_comfyui_vram() only
        clears GPU VRAM — system RAM stays consumed.
      - Qwen 7B (Q4_K_M) needs ~2.2 GB of system RAM to load.
      - With only ~1–1.5 GB free, Ollama throws "model requires more system
        memory than is available".

    Fix: when ComfyUI is the image backend AND Ollama is requested AND a Groq
    API key is present, silently promote the provider to "groq".  This routes
    all text generation through the cloud (zero local RAM) while keeping image
    generation fully local on the GPU.

    Override: set FORCE_TEXT_PROVIDER=ollama in .env to disable this behaviour.
    """
    requested = get_dynamic_setting("ACTIVE_PROVIDER", "ollama")

    if (
        requested == "ollama"
        and get_dynamic_setting("IMAGE_GEN_BACKEND", IMAGE_GEN_BACKEND) == "comfyui"
        and GROQ_API_KEY
        and get_dynamic_setting("FORCE_TEXT_PROVIDER", "") != "ollama"
    ):
        _config_log.info(
            "[Config] ComfyUI+Ollama RAM conflict detected: auto-promoting text "
            "provider from Ollama → Groq (cloud). ComfyUI holds ~4–5 GB of system "
            "RAM that Python cannot release; Qwen 7B needs 2.2 GB more. "
            "To force Ollama anyway, set FORCE_TEXT_PROVIDER=ollama in .env."
        )
        return "groq"

    return requested


def get_llm(temperature: float = 0.7):
    """
    Factory function — returns the correct LangChain chat model based on
    dynamically loaded settings from the Chainlit UI or .env.

    When ComfyUI is the image backend and Ollama is configured, this function
    automatically promotes the provider to Groq to avoid the RAM conflict
    between Flux 9B (held by ComfyUI's Python process) and Qwen 7B.
    See _resolve_text_provider() for full details.
    """
    active_provider = _resolve_text_provider()

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
    active_provider = _resolve_text_provider()
    if active_provider == "groq":
        groq_model = get_dynamic_setting("GROQ_MODEL", "llama-3.1-8b-instant")
        requested   = get_dynamic_setting("ACTIVE_PROVIDER", "ollama")
        suffix = " [auto: ComfyUI RAM conflict]" if requested == "ollama" else ""
        return f"Groq / {groq_model}{suffix}"
    elif active_provider == "google":
        return f"Google Gemini / {get_dynamic_setting('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')}"
    elif active_provider == "ollama":
        return f"Ollama (local) / {get_dynamic_setting('OLLAMA_MODEL', 'qwen2.5:7b-instruct-q4_K_M')} @ {OLLAMA_BASE_URL}"
    return active_provider
