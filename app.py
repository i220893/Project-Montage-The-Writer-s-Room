# ============================================================
# app.py — The Writer's Room: Chainlit UI + HITL Orchestrator
#
# Run with:
#   chainlit run app.py
#
# Requires the MCP server to be running first:
#   python mcp_server.py
# ============================================================

import json
import logging
import os
from pathlib import Path

import chainlit as cl
from langchain_core.messages import HumanMessage

from config import model_info
from graph.state import initial_state
from graph.workflow import build_graph, make_thread_config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Build the graph once at startup ──────────────────────────────────────────
graph = build_graph(enable_hitl=True)


# ─────────────────────────────────────────────────────────────────────────────
# ON CHAT START
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    """Initialize a new user session."""
    session_id = cl.user_session.get("id")
    cl.user_session.set("mode", None)
    cl.user_session.set("thread_config", make_thread_config(session_id))
    cl.user_session.set("awaiting_hitl", False)

    from config import get_dynamic_setting
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="ACTIVE_PROVIDER",
            label="Agent LLM Provider",
            values=["groq", "google", "ollama"],
            initial_index=["groq", "google", "ollama"].index(get_dynamic_setting("ACTIVE_PROVIDER", "ollama")),
        ),
        cl.input_widget.Select(
            id="OLLAMA_MODEL",
            label="Ollama Model",
            values=["qwen2.5:7b-instruct-q4_K_M", "llama3.2"],
            initial_index=0,
        ),
        cl.input_widget.Select(
            id="IMAGE_GEN_MODEL",
            label="Image Generation Model (Gemini)",
            values=["gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image", "gemini-3.0-pro-image"],
            initial_index=1,
        )
    ]).send()

    await cl.Message(
        content=(
            "# 🎬 The Writer's Room\n"
            f"*Powered by {model_info()} · MCP Tool Discovery · FAISS Memory*\n\n"
            "---\n\n"
            "I'm your autonomous creative production assistant. I can:\n"
            "- 📝 **Generate** a full screenplay from your story idea\n"
            "- ✅ **Validate** a script you've already written\n\n"
            "Both paths produce:\n"
            "→ `scene_manifest.json` · `character_db.json` · `image_assets/`\n\n"
            "---\n\n"
            "**Please select a mode below to begin:**"
        ),
        actions=[
            cl.Action(name="select_mode", value="generate", label="📝 Generate Script", payload={"action": "generate"}),
            cl.Action(name="select_mode", value="validate", label="✅ Validate Script", payload={"action": "validate"}),
        ]
    ).send()

@cl.action_callback("select_mode")
async def on_select_mode(action: cl.Action):
    """Handle mode selection button clicks."""
    # Fallback to payload or value depending on the Chainlit version
    mode = getattr(action, "value", None) or action.payload.get("action")
    cl.user_session.set("mode", mode)
    
    if mode == "generate":
        await cl.Message(content="✅ **Generate Mode** selected.\n\nWhat's your story idea? (e.g. *A detective thriller set in 1940s Cairo*)").send()
    else:
        await cl.Message(content="✅ **Validate Mode** selected.\n\nPaste your screenplay below and I'll analyse its structure:").send()
        
    await action.remove()

@cl.on_settings_update
async def setup_agent(settings):
    """Save updated settings to JSON so the MCP Server can read them dynamically."""
    from config import SETTINGS_FILE
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    await cl.Message(content=f"⚙️ **Settings updated!** Restarting pipeline will use new models.").send()



# ─────────────────────────────────────────────────────────────────────────────
# ON MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    mode           = cl.user_session.get("mode")
    thread_config  = cl.user_session.get("thread_config")
    awaiting_hitl  = cl.user_session.get("awaiting_hitl", False)

    # ── HITL: user typed a response to the validation review ─────────────────
    if awaiting_hitl:
        await _handle_hitl_text_response(message.content, thread_config)
        return

    # ── Mode selection ────────────────────────────────────────────────────────
    if mode is None:
        await cl.Message(
            content="Please select a mode using the buttons ('Generate Script' or 'Validate Script') to begin."
        ).send()
        return

    # ── Run the pipeline ──────────────────────────────────────────────────────
    await _run_pipeline(message.content, mode, thread_config)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def _run_pipeline(user_input: str, mode: str, thread_config: dict):
    """Build initial state and stream through the LangGraph workflow."""

    state = initial_state(
        mode        = mode,
        user_prompt = user_input if mode == "generate" else None,
        raw_script  = user_input if mode == "validate" else None,
    )

    await cl.Message(
        content=f"🚀 **Pipeline Started** | Mode: `{mode.upper()}` | Model: `{model_info()}`"
    ).send()

    try:
        async for chunk in graph.astream(state, config=thread_config):
            for node_name, node_state in chunk.items():

                # ── Skip internal LangGraph system events ──────────────────
                if not isinstance(node_state, dict):
                    continue

                # ── Per-node progress update ───────────────────────────────
                emoji = {
                    "scriptwriter":       "✍️",
                    "validator":          "🔍",
                    "character_designer": "👥",
                    "image_synthesizer":  "🎨",
                }.get(node_name, "⚙️")

                await cl.Message(
                    content=f"{emoji} **[{node_name.replace('_', ' ').title()}]** — completed",
                    author=node_name,
                ).send()

                # ── Error handling ──────────────────────────────────────────
                if node_state.get("error"):
                    await cl.Message(
                        content=f"❌ **Error in {node_name}:**\n```\n{node_state['error']}\n```"
                    ).send()
                    return

                # ── HITL: validator found errors → display buttons + stop ──
                if node_name == "validator" and node_state.get("validation_errors"):
                    await _trigger_hitl(node_state["validation_errors"], thread_config)
                    return  # stop processing — wait for button click in action callback

        # ── Stream finished — check if graph is truly done or just interrupted ──
        state_snapshot = graph.get_state(thread_config)
        if state_snapshot.next:
            # Graph is paused at an interrupt — do NOT show deliverables
            logger.info("[Pipeline] Graph paused at: %s — waiting for HITL.", state_snapshot.next)
        else:
            # Graph ran to completion — show deliverables
            await _send_deliverables(state_snapshot.values)

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        await cl.Message(content=f"❌ **Pipeline error:**\n```\n{exc}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# HITL — Human-in-the-Loop Review
# ─────────────────────────────────────────────────────────────────────────────

async def _trigger_hitl(errors: list[str], thread_config: dict):
    """Pause the graph and ask the user to approve or retry using standard Chainlit Actions."""
    cl.user_session.set("awaiting_hitl", True)
    error_str = "\n".join(f"  - {e}" for e in errors)

    await cl.Message(
        content=(
            f"⚠️ **Validation Issues Found** ({len(errors)} issue(s)):\n"
            f"{error_str}\n\n"
            "How would you like to proceed?"
        ),
        actions=[
            cl.Action(
                name    = "hitl_approve",
                value   = "approve",
                label   = "✅ Approve & Continue to Character Design",
                payload = {"action": "approve"},
            ),
            cl.Action(
                name    = "hitl_retry",
                value   = "retry",
                label   = "🔄 Reject & Start Over",
                payload = {"action": "retry"},
            ),
        ],
    ).send()


@cl.action_callback("hitl_approve")
async def on_approve(action: cl.Action):
    """Callback when user clicks Approve."""
    # Only process if we are currently awaiting HITL
    if not cl.user_session.get("awaiting_hitl"):
        return
        
    await cl.Message(content="*Selected: ✅ Approve & Continue*").send()
    # Remove the buttons so they can't be clicked again
    await action.remove()
    
    thread_config = cl.user_session.get("thread_config")
    await _resume_after_hitl(approved=True, thread_config=thread_config)


@cl.action_callback("hitl_retry")
async def on_retry(action: cl.Action):
    """Callback when user clicks Retry."""
    if not cl.user_session.get("awaiting_hitl"):
        return
        
    await cl.Message(content="*Selected: 🔄 Reject & Start Over*").send()
    await action.remove()
    
    cl.user_session.set("mode", None)
    cl.user_session.set("awaiting_hitl", False)
    await cl.Message(content="🔄 Starting over. What's your story idea?").send()


async def _handle_hitl_text_response(text: str, thread_config: dict):
    """Fallback if user types instead of clicking."""
    text = text.lower().strip()
    if "approve" in text:
        await _resume_after_hitl(approved=True, thread_config=thread_config)
    elif "retry" in text:
        cl.user_session.set("mode", None)
        cl.user_session.set("awaiting_hitl", False)
        await cl.Message(content="🔄 Starting over. What's your story idea?").send()
    else:
        await cl.Message(content="Please click a button or type **approve** or **retry**.").send()


async def _resume_after_hitl(approved: bool, thread_config: dict):
    """Resume the paused graph after human approval."""
    cl.user_session.set("awaiting_hitl", False)

    # ── Recover scene_manifest from disk if state lost it across HITL boundary ─
    from config import SCENE_MANIFEST
    current_state = graph.get_state(thread_config).values
    scene_manifest = current_state.get("scene_manifest")
    if not scene_manifest:
        manifest_path = Path(SCENE_MANIFEST)
        if manifest_path.exists():
            try:
                scene_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                logger.info("[HITL] Recovered scene_manifest from disk: %s", str(manifest_path))
            except Exception as e:
                logger.warning("[HITL] Could not read scene_manifest from disk: %s", e)

    # Update state: mark as approved + restore scene_manifest if needed
    update_payload = {"script_approved": approved, "hitl_feedback": "Approved by user."}
    if scene_manifest:
        update_payload["scene_manifest"] = scene_manifest

    await graph.aupdate_state(thread_config, update_payload)

    await cl.Message(content="✅ **Approved!** Continuing to character design and image generation...").send()

    # Resume the graph from the checkpoint (pass None as input to continue)
    error_occurred = False
    try:
        stream = graph.astream(None, config=thread_config)
        async for chunk in stream:
            for node_name, node_state in chunk.items():
                emoji = {
                    "character_designer": "👥",
                    "image_synthesizer":  "🎨",
                }.get(node_name, "⚙️")
                await cl.Message(
                    content=f"{emoji} **[{node_name.replace('_', ' ').title()}]** — completed",
                    author=node_name,
                ).send()

                if isinstance(node_state, dict) and node_state.get("error"):
                    await cl.Message(
                        content=f"❌ **Error in {node_name}:**\n```\n{node_state['error']}\n```"
                    ).send()
                    error_occurred = True
                    break  # break inner loop
            if error_occurred:
                await stream.aclose()  # Properly close the generator
                break  # break outer loop

        if not error_occurred:
            final = graph.get_state(thread_config).values
            await _send_deliverables(final)

    except Exception as exc:
        logger.error("Resume error: %s", exc, exc_info=True)
        await cl.Message(content=f"❌ **Error while resuming:**\n```\n{exc}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLES DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

async def _send_deliverables(state: dict):
    """Render all three deliverables in the Chainlit chat UI."""
    await cl.Message(
        content="---\n## 🎬 Writer's Room — Pipeline Complete!\n\nHere are your deliverables:"
    ).send()

    # ── 1. Scene Manifest ─────────────────────────────────────────────────────
    manifest = state.get("scene_manifest")
    if manifest:
        title   = manifest.get("title", "Untitled")
        logline = manifest.get("logline", "")
        n_scenes = len(manifest.get("scenes", []))
        preview = json.dumps(manifest, indent=2)[:3000]

        await cl.Message(
            content=(
                f"### 📄 Scene Manifest — *{title}*\n"
                f"**Logline:** {logline}\n"
                f"**Scenes:** {n_scenes}\n\n"
                f"```json\n{preview}\n```\n"
                f"*Full file saved → `outputs/scene_manifest.json`*"
            )
        ).send()
    else:
        await cl.Message(content="⚠️ No scene manifest was generated.").send()

    # ── 2. Character Database ──────────────────────────────────────────────────
    char_db = state.get("character_db")
    if char_db:
        char_summary = "\n".join(
            f"- **{name}** ({profile.get('role', 'unknown')})"
            for name, profile in char_db.items()
        )
        preview = json.dumps(char_db, indent=2)[:2000]

        await cl.Message(
            content=(
                f"### 👥 Character Database\n"
                f"{char_summary}\n\n"
                f"```json\n{preview}\n```\n"
                f"*Full file saved → `outputs/character_db.json`*"
            )
        ).send()
    else:
        await cl.Message(content="⚠️ No character database was generated.").send()

    # ── 3. Generated Images ────────────────────────────────────────────────────
    image_paths = state.get("image_paths") or []
    if image_paths:
        await cl.Message(content=f"### 🖼️ Character Portraits ({len(image_paths)} generated)").send()
        for img_path in image_paths:
            if os.path.exists(img_path):
                char_name = Path(img_path).stem.replace("_", " ")
                elements  = [cl.Image(name=char_name, path=img_path, display="inline")]
                await cl.Message(
                    content=f"**{char_name}**",
                    elements=elements,
                ).send()
            else:
                await cl.Message(content=f"⚠️ Image not found: `{img_path}`").send()
    else:
        await cl.Message(content="⚠️ No character images were generated.").send()

    # ── Summary footer ─────────────────────────────────────────────────────────
    await cl.Message(
        content=(
            "---\n"
            "✅ **All outputs saved to `outputs/`.**\n"
            "Type `generate` or `validate` to start a new project."
        )
    ).send()
    # Reset mode so user can start a new session
    cl.user_session.set("mode", None)
