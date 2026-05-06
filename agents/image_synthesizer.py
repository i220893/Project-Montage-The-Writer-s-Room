# ============================================================
# agents/image_synthesizer.py — Image Synthesizer Agent Node
#
# Responsibilities:
#   - Read character_db from state
#   - For EACH character, call generate_character_image MCP tool
#     (which internally calls Gemini image generation API)
#   - Collect all returned image file paths
#   - Return updated state with image_paths
#
# NOTE: Image generation is rate-limited by the Gemini API.
#       The agent processes characters sequentially to avoid errors.
# ============================================================

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_SERVER_NAME, get_llm, model_info

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
IMAGE_SYNTHESIZER_SYSTEM = """\
You are PIXEL — a visual director responsible for generating character portraits.

Your task is to:
1. For EACH character provided, call the `generate_character_image` tool.
   - Use the character's name as character_name.
   - Use the character's 'appearance' field as visual_description.
2. After generating ALL images, list each character name and their image path.

IMPORTANT:
- Always call generate_character_image for EVERY character — do not skip any.
- Do not generate images yourself — always use the tool.
- Process one character at a time (sequential tool calls).
"""


async def image_synthesizer_node(state: dict) -> dict:
    """
    LangGraph node: Image Synthesizer Agent.

    Iterates over character_db, calls generate_character_image MCP tool
    for each character, collects image paths.
    """
    logger.info("[ImageSynthesizer] Starting | Model: %s", model_info())

    character_db = state.get("character_db")

    # ── Disk fallback: recover character_db if state lost it ─────────────────
    if not character_db:
        from config import OUTPUT_DIR
        from pathlib import Path
        import json as _json
        session_id = state.get("session_id", "default")
        base = Path(OUTPUT_DIR)
        char_db_path = base / f"session_{session_id}" / "character_db.json" if session_id and session_id != "default" else base / "latest" / "character_db.json"
        if char_db_path.exists():
            try:
                character_db = _json.loads(char_db_path.read_text(encoding="utf-8"))
                logger.info("[ImageSynthesizer] Recovered character_db from disk: %s", str(char_db_path))
            except Exception as e:
                logger.warning("[ImageSynthesizer] Could not read character_db from disk: %s", e)

    if not character_db:
        return {
            **state,
            "error": "ImageSynthesizer: character_db not found in state or on disk.",
            "current_agent": "image_synthesizer",
        }

    # ── Build a concise character list for the agent prompt ──────────────────
    char_list_str = json.dumps({
        name: {
            "appearance": profile.get("appearance", ""),
            "role":       profile.get("role", ""),
        }
        for name, profile in character_db.items()
    }, indent=2)

    image_paths: list[str] = []

    try:
        mcp = MultiServerMCPClient({
            MCP_SERVER_NAME: {
                "url":       MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        if True:
            tools     = await mcp.get_tools()
            llm       = get_llm(temperature=0.5)
            llm_bound = llm.bind_tools(tools)

            n_chars = len(character_db)
            messages = [
                SystemMessage(content=IMAGE_SYNTHESIZER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Generate portrait images for all {n_chars} character(s):\n\n"
                        f"{char_list_str}"
                    )
                ),
            ]

            # One tool call per character + one final LLM turn (avoided when all done early)
            max_steps = (n_chars * 2) + 2

            for step in range(max_steps):
                # ── Early-exit: all images collected — skip the final Ollama call ──────
                # After the last image is generated ComfyUI's Flux model is still hot in
                # VRAM.  The next llm_bound.ainvoke() would try to reload Qwen, but there
                # isn't enough free RAM for both models simultaneously.  The final call
                # only produces a human-readable summary we never use, so we bail here.
                if len(image_paths) >= n_chars:
                    logger.info(
                        "[ImageSynthesizer] All %d image(s) collected — skipping final "
                        "LLM wrap-up call to avoid Ollama/ComfyUI RAM collision.",
                        n_chars,
                    )
                    break

                ai_msg = await llm_bound.ainvoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    logger.info("[ImageSynthesizer] Agent finished in %d step(s).", step + 1)
                    break

                for tc in ai_msg.tool_calls:
                    logger.info(
                        "[ImageSynthesizer] Calling tool: %s | character: %s",
                        tc["name"],
                        tc["args"].get("character_name", "?"),
                    )
                    tool_obj = next((t for t in tools if t.name == tc["name"]), None)
                    if tool_obj is None:
                        result = f"ERROR: tool '{tc['name']}' not found."
                    else:
                        tc_args = dict(tc["args"])
                        tc_args["session_id"] = state.get("session_id", "default")
                        result = await tool_obj.ainvoke(tc_args)

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

                    logger.info("[ImageSynthesizer] Result: %s", result_content[:300])
                    messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))

                    # ── Collect image paths ────────────────────────────────────
                    if tc["name"] == "generate_character_image":
                        if result_content and not result_content.startswith("ERROR"):
                            image_paths.append(result_content)
                        else:
                            logger.warning("[ImageSynthesizer] Image gen failed: %s", result_content)

    except Exception as exc:
        logger.error("[ImageSynthesizer] Error: %s", exc, exc_info=True)
        return {**state, "error": str(exc), "current_agent": "image_synthesizer"}

    logger.info("[ImageSynthesizer] Generated %d image(s).", len(image_paths))

    return {
        "messages":      messages,
        "image_paths":   image_paths,
        "current_agent": "image_synthesizer",
    }
