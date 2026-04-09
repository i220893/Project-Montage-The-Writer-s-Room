# ============================================================
# agents/validator.py — Script Validator Agent Node
#
# Responsibilities:
#   - Accept either raw_script (validate mode) or scene_manifest
#     (generated mode) from state
#   - Invoke validate_script_structure MCP tool
#   - Parse validation result and populate validation_errors
#   - If no errors AND scene_manifest was just generated,
#     also call save_scene_manifest to persist it
# ============================================================

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MCP_SERVER_URL, MCP_SERVER_NAME, get_llm, model_info

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
VALIDATOR_SYSTEM = """\
You are VERA — a meticulous script supervisor and story analyst in The Writer's Room.

Your task is to:
1. Call the `validate_script_structure` tool on the provided screenplay text.
2. Report the results clearly:
   - List all ERRORS (critical structural problems).
   - List all WARNINGS (minor issues).
   - State whether the script is valid or not.

IMPORTANT:
- Always call the tool — never evaluate the script yourself.
- Be precise and direct in your report.
"""


async def validator_node(state: dict) -> dict:
    """
    LangGraph node: Script Validator Agent.

    Validates either raw_script (validate mode) or the generated
    scene_manifest (generate mode). Populates validation_errors list.
    """
    logger.info("[Validator] Starting | Model: %s", model_info())

    # ── Determine what to validate ────────────────────────────────────────────
    if state.get("raw_script"):
        script_text = state["raw_script"]
    elif state.get("scene_manifest"):
        script_text = json.dumps(state["scene_manifest"], indent=2)
    else:
        return {
            **state,
            "validation_errors": ["Validator: no script or manifest found in state."],
            "current_agent": "validator",
        }

    validation_errors  = []
    validation_warnings = []

    try:
        mcp = MultiServerMCPClient({
            MCP_SERVER_NAME: {
                "url":       MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        })
        if True:
            tools     = await mcp.get_tools()
            llm       = get_llm(temperature=0.0)
            llm_bound = llm.bind_tools(tools)

            messages = [
                SystemMessage(content=VALIDATOR_SYSTEM),
                HumanMessage(
                    content=(
                        f"Validate this screenplay:\n\n"
                        f"{script_text[:6000]}"  # Truncate to keep within token limits
                    )
                ),
            ]

            raw_validation_json = None
            max_steps = 4

            for step in range(max_steps):
                ai_msg = await llm_bound.ainvoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    logger.info("[Validator] Agent finished in %d step(s).", step + 1)
                    break

                for tc in ai_msg.tool_calls:
                    logger.info("[Validator] Calling tool: %s", tc["name"])
                    tool_obj = next((t for t in tools if t.name == tc["name"]), None)
                    if tool_obj is None:
                        result = f"ERROR: tool '{tc['name']}' not found."
                    else:
                        result = await tool_obj.ainvoke(tc["args"])

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

                    messages.append(ToolMessage(content=result_content, tool_call_id=tc["id"]))

                    if tc["name"] == "validate_script_structure":
                        raw_validation_json = result_content

            # ── Parse validation JSON ─────────────────────────────────────────
            if raw_validation_json:
                try:
                    report = json.loads(raw_validation_json)
                    validation_errors   = report.get("errors",   [])
                    validation_warnings = report.get("warnings", [])
                    logger.info(
                        "[Validator] Valid: %s | Errors: %d | Warnings: %d",
                        report.get("is_valid"),
                        len(validation_errors),
                        len(validation_warnings),
                    )
                except json.JSONDecodeError:
                    validation_errors = ["Validator could not parse validation result."]

    except Exception as exc:
        logger.error("[Validator] Error: %s", exc, exc_info=True)
        return {**state, "error": str(exc), "current_agent": "validator"}

    return {
        "messages":          messages,
        "validation_errors": validation_errors,
        "current_agent":     "validator",
    }
