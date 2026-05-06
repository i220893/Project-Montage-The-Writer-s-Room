# ============================================================
# agents/scriptwriter.py — Scriptwriter Agent Node
#
# Responsibilities:
#   - Receive the user's story prompt from state
#   - Dynamically discover tools from the MCP server
#   - Call generate_screenplay MCP tool via ReAct loop
#   - Call save_scene_manifest MCP tool to persist output
#   - Call store_in_memory MCP tool to save to FAISS
#   - Return updated state with scene_manifest
# ============================================================

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_SERVER_NAME, get_llm, model_info

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
SCRIPTWRITER_SYSTEM = """\
You are ARIA — a professional Hollywood screenwriter working in The Writer's Room.

Your task is to:
1. Call the `generate_screenplay` tool with the user's story prompt.
2. Wait for it to succeed, then politely confirm completion.

IMPORTANT:
- The server will automatically save the script, you do NOT need to call save tools.
- Use num_scenes=3 unless the user specifies otherwise.
- Generate ONLY characters that belong to the user's new story — do not reference or import characters from other stories.
"""


async def scriptwriter_node(state: dict) -> dict:
    """
    LangGraph node: Scriptwriter Agent.

    Connects to MCP server, discovers tools at runtime, then runs a
    ReAct loop (bind_tools → ainvoke → ToolMessage) to:
      1. Generate the screenplay JSON via Groq through the MCP tool.
      2. Persist it to disk.
      3. Store a summary in FAISS memory.
    """
    logger.info("[Scriptwriter] Starting | Model: %s", model_info())

    # ── Free ComfyUI VRAM before loading Qwen for orchestration ───────────────
    # The MCP tools free VRAM internally, but the agent's own LLM calls (which
    # decide which tool to invoke) are NOT protected. If ComfyUI's Flux model
    # is still hot from a previous image-gen run, Qwen can't load → OOM.
    # Agents are async so we can await directly (no ThreadPoolExecutor needed).
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

    user_prompt = state.get("user_prompt", "")
    if not user_prompt:
        return {**state, "error": "Scriptwriter: no user_prompt in state.", "current_agent": "scriptwriter"}

    try:
        mcp = MultiServerMCPClient({
            MCP_SERVER_NAME: {
                "url":       MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        if True:
            # ── Discover tools at runtime (MCP constraint) ────────────────────
            tools     = await mcp.get_tools()
            llm       = get_llm(temperature=0.8)
            llm_bound = llm.bind_tools(tools)

            messages = [
                SystemMessage(content=SCRIPTWRITER_SYSTEM),
                HumanMessage(content=f"Write a screenplay about: {user_prompt}"),
            ]

            scene_manifest = state.get("scene_manifest")
            max_steps      = 6

            for step in range(max_steps):
                ai_msg = await llm_bound.ainvoke(messages)
                messages.append(ai_msg)

                # ── No tool calls → agent is done ─────────────────────────────
                if not ai_msg.tool_calls:
                    logger.info("[Scriptwriter] Agent finished in %d step(s).", step + 1)
                    break

                # ── Execute each tool call ────────────────────────────────────
                for tc in ai_msg.tool_calls:
                    logger.info("[Scriptwriter] Calling tool: %s | args: %s", tc["name"], tc["args"])
                    tool_obj = next((t for t in tools if t.name == tc["name"]), None)
                    if tool_obj is None:
                        result = f"ERROR: tool '{tc['name']}' not found on MCP server."
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

                    logger.info("[Scriptwriter] Tool result: %s", result_content[:200])
                    messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))

                    # ── Capture scene_manifest from generate_screenplay ────
                    if tc["name"] == "generate_screenplay":
                        try:
                            scene_manifest = json.loads(result_content)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("[Scriptwriter] Could not parse screenplay JSON.")

            # NOTE: Auto-save to FAISS is intentionally disabled here.
            # Storing scripts in shared FAISS memory caused the character_designer
            # to pull characters from old sessions during `extract_characters`.
            # Cross-session continuity can be re-enabled in a future feature
            # with session-scoped FAISS indexes.

    except Exception as exc:
        logger.error("[Scriptwriter] Error: %s", exc, exc_info=True)
        return {**state, "error": str(exc), "current_agent": "scriptwriter"}

    return {
        "messages":       messages,
        "scene_manifest": scene_manifest,
        "current_agent":  "scriptwriter",
    }
