# ============================================================
# graph/state.py — Shared State for the Writer's Room Pipeline
#
# WritersRoomState is the single object passed between all
# LangGraph nodes. Every agent reads from it and returns a
# partial update dict. LangGraph merges updates automatically.
# ============================================================

import operator
from typing import Annotated, List, Literal, Optional
from langchain_core.messages import BaseMessage


class WritersRoomState(dict):
    """
    Typed shared state for the Writer's Room LangGraph workflow.

    Fields:
        messages         — Full conversation history (auto-appended via operator.add)
        mode             — "generate" (LLM writes script) or "validate" (user uploads script)
        user_prompt      — Raw user story idea (generate mode)
        raw_script       — Pasted/uploaded script text (validate mode)
        scene_manifest   — Structured screenplay dict (output of Scriptwriter)
        validation_errors — List of structural errors found by Validator
        script_approved  — Human-in-the-loop approval flag
        hitl_feedback    — Optional notes provided by human reviewer
        character_db     — Dict of character name → profile (output of Character Designer)
        image_paths      — List of absolute paths to generated character images
        current_agent    — Name of the last agent that ran (for logging/UI)
        error            — Error message if a node failed
    """

    messages:          Annotated[List[BaseMessage], operator.add]
    mode:              Literal["generate", "validate"]

    # ── Inputs ───────────────────────────────────────────────────────────────
    user_prompt:       Optional[str]
    raw_script:        Optional[str]

    # ── Intermediate outputs ──────────────────────────────────────────────────
    scene_manifest:    Optional[dict]
    validation_errors: Optional[List[str]]
    script_approved:   bool
    hitl_feedback:     Optional[str]

    # ── Final outputs ─────────────────────────────────────────────────────────
    character_db:      Optional[dict]
    image_paths:       Optional[List[str]]

    # ── Metadata ─────────────────────────────────────────────────────────────
    current_agent:     Optional[str]
    error:             Optional[str]


def initial_state(
    mode: Literal["generate", "validate"],
    user_prompt: Optional[str] = None,
    raw_script: Optional[str]  = None,
) -> dict:
    """
    Return a clean initial state dict for starting a new pipeline run.

    Usage:
        state = initial_state(mode="generate", user_prompt="A heist in Tokyo")
        result = await graph.ainvoke(state, config=THREAD_CONFIG)
    """
    from langchain_core.messages import HumanMessage

    content = user_prompt or raw_script or ""
    return {
        "messages":          [HumanMessage(content=content)],
        "mode":              mode,
        "user_prompt":       user_prompt,
        "raw_script":        raw_script,
        "scene_manifest":    None,
        "validation_errors": None,
        "script_approved":   False,
        "hitl_feedback":     None,
        "character_db":      None,
        "image_paths":       [],
        "current_agent":     None,
        "error":             None,
    }
