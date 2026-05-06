# ============================================================
# agents/character_designer.py — Character Designer Agent Node
#
# Responsibilities:
#   - Read the scene_manifest from state
#   - Call extract_characters MCP tool to build detailed profiles
#   - Call save_character_db MCP tool to persist the profiles
#   - Call store_in_memory MCP tool to save character data to FAISS
#   - Return updated state with character_db
# ============================================================

import json
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_SERVER_NAME, model_info

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
CHARACTER_DESIGNER_SYSTEM = """\
You are CASSIDY — a seasoned casting director and character analyst in The Writer's Room.

Your task is to:
1. Call the `extract_characters` tool with the full scene manifest JSON.

IMPORTANT:
- Always use tools — never write character profiles directly.
- The 'appearance' field in each profile must be richly detailed for image generation.
  Include: approximate age, build, ethnicity cues, hair, clothing, distinguishing features.
- The server will automatically save the characters to disk, so you do NOT need to call save_character_db or store_in_memory.
- After the tool succeeds, summarise the characters found (names and roles only).
- Extract ONLY characters from the scene manifest provided — do not invent or import characters from other stories.
"""


async def character_designer_node(state: dict) -> dict:
    """
    LangGraph node: Character Designer Agent.

    Reads scene_manifest from state, extracts character profiles via MCP tool,
    saves them to disk and FAISS memory.
    """
    logger.info("[CharacterDesigner] Starting | Model: %s", model_info())

    # ── Free ComfyUI VRAM before loading Qwen for orchestration ───────────────
    from config import IMAGE_GEN_BACKEND, COMFYUI_BASE_URL, get_dynamic_setting
    if get_dynamic_setting("IMAGE_GEN_BACKEND", IMAGE_GEN_BACKEND) == "comfyui":
        try:
            from comfyui.vram_manager import free_comfyui_vram, wait_for_vram_clear
            freed = await free_comfyui_vram(
                get_dynamic_setting("COMFYUI_BASE_URL", COMFYUI_BASE_URL)
            )
            if freed:
                await wait_for_vram_clear(4.0)
        except ImportError:
            pass  # comfyui package absent — skip

    scene_manifest = state.get("scene_manifest")

    # ── Disk fallback: recover scene_manifest if state lost it across HITL boundary ─
    if not scene_manifest:
        from config import OUTPUT_DIR
        from pathlib import Path
        import json as _json
        session_id = state.get("session_id", "default")
        base = Path(OUTPUT_DIR)
        manifest_path = base / f"session_{session_id}" / "scene_manifest.json" if session_id and session_id != "default" else base / "latest" / "scene_manifest.json"
        if manifest_path.exists():
            try:
                scene_manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
                logger.info("[CharacterDesigner] Recovered scene_manifest from disk: %s", str(manifest_path))
            except Exception as e:
                logger.warning("[CharacterDesigner] Could not read scene_manifest from disk: %s", e)

    if not scene_manifest:
        return {
            **state,
            "error": "CharacterDesigner: scene_manifest not found in state or on disk.",
            "current_agent": "character_designer",
        }

    manifest_json = json.dumps(scene_manifest, indent=2)
    character_db  = None
    messages      = []  # kept for state compatibility

    try:
        mcp = MultiServerMCPClient({
            MCP_SERVER_NAME: {
                "url":       MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        tools = await mcp.get_tools()

        # ── Direct tool call: bypass LLM-as-orchestrator entirely ────────────
        #
        # WHY: The only thing the ReAct loop ever does here is call
        # `extract_characters(scene_manifest_json=<full screenplay JSON>)`.
        # To make that tool call, the LLM must re-encode ~8 000 characters of
        # JSON as an escaped string argument inside another JSON envelope.
        # Smaller models (llama-3.1-8b-instant, Qwen 7B) reliably produce
        # malformed escaping → Groq 400 `tool_use_failed` / Ollama garbling.
        #
        # The actual AI analysis still runs — it happens INSIDE the MCP tool
        # (`extract_characters` calls `get_llm()` in mcp_server.py). We just
        # skip the fragile intermediary "LLM constructs the call arguments" step.
        tool_obj = next((t for t in tools if t.name == "extract_characters"), None)
        if tool_obj is None:
            raise ValueError("extract_characters tool not found on MCP server.")

        logger.info("[CharacterDesigner] Calling extract_characters directly (no LLM orchestration).")
        result = await tool_obj.ainvoke({
            "scene_manifest_json": manifest_json,
            "session_id":          state.get("session_id", "default"),
        })

        # ── Unwrap MCP Adapter List Format ────────────────────────────────────
        result_str = str(result)
        if result_str.startswith("[{") and "'text':" in result_str:
            try:
                import ast
                parsed = ast.literal_eval(result_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    result_str = parsed[0].get("text", result_str)
            except Exception:
                pass

        if result_str.startswith("ERROR"):
            raise ValueError(f"extract_characters returned an error: {result_str[:300]}")

        character_db = json.loads(result_str)
        logger.info("[CharacterDesigner] Extracted %d character(s).", len(character_db))

    except Exception as exc:
        logger.error("[CharacterDesigner] Error: %s", exc, exc_info=True)
        return {**state, "error": str(exc), "current_agent": "character_designer"}

    return {
        "messages":      messages,
        "character_db":  character_db,
        "current_agent": "character_designer",
    }
