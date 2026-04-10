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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_SERVER_NAME, get_llm, model_info

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
CHARACTER_DESIGNER_SYSTEM = """\
You are CASSIDY — a seasoned casting director and character analyst in The Writer's Room.

Your task is to:
1. Call the `extract_characters` tool with the full scene manifest JSON.
2. Call the `store_in_memory` tool to save a character summary for future recall.
   - Use metadata: {{"type": "characters", "title": "<story title>"}}

IMPORTANT:
- Always use tools — never write character profiles directly.
- The 'appearance' field in each profile must be richly detailed for image generation.
  Include: approximate age, build, ethnicity cues, hair, clothing, distinguishing features.
- The server will automatically save the characters to disk, so you do NOT need to call the save_character_db tool.
- After tools succeed, summarise the characters found (names and roles only).
"""


async def character_designer_node(state: dict) -> dict:
    """
    LangGraph node: Character Designer Agent.

    Reads scene_manifest from state, extracts character profiles via MCP tool,
    saves them to disk and FAISS memory.
    """
    logger.info("[CharacterDesigner] Starting | Model: %s", model_info())

    scene_manifest = state.get("scene_manifest")

    # ── Disk fallback: recover scene_manifest if state lost it across HITL boundary ─
    if not scene_manifest:
        from config import SCENE_MANIFEST
        from pathlib import Path
        import json as _json
        manifest_path = Path(SCENE_MANIFEST)
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

    try:
        mcp = MultiServerMCPClient({
            MCP_SERVER_NAME: {
                "url":       MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        if True:
            tools     = await mcp.get_tools()
            llm       = get_llm(temperature=0.3)
            llm_bound = llm.bind_tools(tools)

            story_title = scene_manifest.get("title", "Untitled")

            messages = [
                SystemMessage(content=CHARACTER_DESIGNER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Extract all characters from this screenplay for '{story_title}':\n\n"
                        f"{manifest_json[:8000]}"
                    )
                ),
            ]

            max_steps = 6

            for step in range(max_steps):
                ai_msg = await llm_bound.ainvoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    logger.info("[CharacterDesigner] Agent finished in %d step(s).", step + 1)
                    break

                for tc in ai_msg.tool_calls:
                    logger.info("[CharacterDesigner] Calling tool: %s", tc["name"])
                    tool_obj = next((t for t in tools if t.name == tc["name"]), None)
                    if tool_obj is None:
                        result = f"ERROR: tool '{tc['name']}' not found."
                    else:
                        # ── Fix: serialize metadata dict→str for store_in_memory ─
                        args = dict(tc["args"])
                        if tc["name"] == "store_in_memory" and isinstance(args.get("metadata"), dict):
                            args["metadata"] = json.dumps(args["metadata"])
                        try:
                            result = await tool_obj.ainvoke(args)
                        except Exception as tool_exc:
                            # store_in_memory is optional — log but don't crash
                            logger.warning("[CharacterDesigner] Tool %s failed (non-fatal): %s", tc["name"], tool_exc)
                            result = f"Tool {tc['name']} failed: {tool_exc}"

                    # ── Unwrap MCP Adapter List Format ────────────────────────
                    result_str = str(result)
                    if result_str.startswith("[{") and "'text':" in result_str:
                        try:
                            import ast
                            parsed = ast.literal_eval(result_str)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                result_str = parsed[0].get("text", result_str)
                        except Exception:
                            pass
                    result_content = result_str

                    logger.info("[CharacterDesigner] Tool result: %s", result_content[:200])
                    messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))

                    # ── Capture character_db from extract_characters ───────
                    if tc["name"] == "extract_characters":
                        try:
                            character_db = json.loads(result_content)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("[CharacterDesigner] Could not parse character JSON.")

        # ── Direct fallback: if LLM didn't call extract_characters, do it yourself ─
        if not character_db:
            logger.warning("[CharacterDesigner] LLM skipped tool call — running extract_characters directly.")
            tool_obj = next((t for t in tools if t.name == "extract_characters"), None)
            if tool_obj:
                try:
                    result = await tool_obj.ainvoke({"scene_manifest_json": manifest_json})
                    result_str = str(result)
                    if result_str.startswith("[{") and "'text':" in result_str:
                        import ast
                        parsed = ast.literal_eval(result_str)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            result_str = parsed[0].get("text", result_str)
                    character_db = json.loads(result_str)
                    logger.info("[CharacterDesigner] Fallback extraction succeeded: %d characters.", len(character_db))
                except Exception as fallback_exc:
                    logger.error("[CharacterDesigner] Fallback extraction failed: %s", fallback_exc)
                    raise ValueError(f"Character extraction failed on fallback: {fallback_exc}\nLLM returned: {result_str[:200]}")

    except Exception as exc:
        logger.error("[CharacterDesigner] Error: %s", exc, exc_info=True)
        return {**state, "error": str(exc), "current_agent": "character_designer"}

    return {
        "messages":      messages,
        "character_db":  character_db,
        "current_agent": "character_designer",
    }
