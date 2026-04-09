# ============================================================
# graph/workflow.py — LangGraph StateGraph for Writer's Room
#
# Graph topology:
#
#   START
#     │
#     ├─[generate]──► Scriptwriter ──► Validator
#     │                                    │
#     └─[validate]──►─────────────────────►│
#                                           │
#                               ┌───────────┴────────────┐
#                          [has errors]            [no errors]
#                               │                        │
#                          ── INTERRUPT ──        Character Designer
#                          (Chainlit HITL)               │
#                               │                 Image Synthesizer
#                     [approved] [retry]                 │
#                               │                       END
#                          continue / restart
#
# Human-in-the-Loop:
#   The graph is compiled with interrupt_before=["character_designer"].
#   When validation finds errors, graph pauses at that interrupt point.
#   Chainlit shows the errors and asks user to approve or retry.
#   On approval: graph.aupdate_state({"script_approved": True}) + resume.
#   On retry:    restart with a new invocation.
# ============================================================

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import WritersRoomState
from agents.scriptwriter import scriptwriter_node
from agents.validator import validator_node
from agents.character_designer import character_designer_node
from agents.image_synthesizer import image_synthesizer_node

logger = logging.getLogger(__name__)


# ── Routing Functions ────────────────────────────────────────────────────────

def route_entry(state: dict) -> str:
    """
    Entry-point router.
    "generate" mode → Scriptwriter writes the script first.
    "validate" mode → jump straight to Validator.
    """
    mode = state.get("mode", "generate")
    logger.info("[Router:entry] mode=%s", mode)
    return "scriptwriter" if mode == "generate" else "validator"


def route_after_validator(state: dict) -> str:
    """
    After validation:
      - Errors found  → pause for human review (HITL interrupt fires here).
      - No errors     → proceed directly to Character Designer.

    NOTE: The HITL interrupt is implemented at the graph compile level
    via interrupt_before=["character_designer"]. So even if we route to
    "character_designer", the graph PAUSES before entering that node
    and waits for Chainlit to call graph.aupdate_state() + resume.
    """
    errors = state.get("validation_errors") or []
    approved = state.get("script_approved", False)

    if errors and not approved:
        logger.info("[Router:post-validator] %d error(s) found. HITL pause.", len(errors))
    else:
        logger.info("[Router:post-validator] No blocking errors. Continuing.")

    # Always route to character_designer — the interrupt_before handles the pause.
    return "character_designer"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(enable_hitl: bool = True) -> StateGraph:
    """
    Build and compile the Writer's Room StateGraph.

    Args:
        enable_hitl: If True (default), compiles with interrupt_before=["character_designer"]
                     for human-in-the-loop review. Set to False for automated testing.

    Returns:
        A compiled LangGraph CompiledGraph ready for ainvoke / astream.
    """
    builder = StateGraph(dict)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("scriptwriter",        scriptwriter_node)
    builder.add_node("validator",           validator_node)
    builder.add_node("character_designer",  character_designer_node)
    builder.add_node("image_synthesizer",   image_synthesizer_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_conditional_entry_point(
        route_entry,
        {
            "scriptwriter": "scriptwriter",
            "validator":    "validator",
        },
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Scriptwriter always feeds into Validator for quality check
    builder.add_edge("scriptwriter", "validator")

    # After validation → conditional route (errors pause at character_designer interrupt)
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {"character_designer": "character_designer"},
    )

    # Linear pipeline: character profiles → image generation → done
    builder.add_edge("character_designer", "image_synthesizer")
    builder.add_edge("image_synthesizer",  END)

    # ── Compile with Human-in-the-Loop checkpoint ─────────────────────────────
    memory    = MemorySaver()
    interrupt = ["character_designer"] if enable_hitl else []

    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=interrupt,
    )

    logger.info(
        "[Workflow] Graph compiled. HITL=%s | Interrupt before: %s",
        enable_hitl, interrupt,
    )
    return graph


# ── Thread Config Helper ──────────────────────────────────────────────────────

def make_thread_config(session_id: str = "writers-room") -> dict:
    """
    Return a LangGraph thread config dict for a given session ID.

    Usage:
        config = make_thread_config(cl.user_session.get("id"))
        await graph.ainvoke(state, config=config)
    """
    return {"configurable": {"thread_id": session_id}}
