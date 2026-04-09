# ============================================================
# mcp_server.py — The Writer's Room MCP Tool Server
#
# This is the SINGLE SOURCE OF TRUTH for all tools.
# Agents NEVER call APIs directly — they discover and invoke
# tools via this MCP server at runtime.
#
# Run this FIRST in a separate terminal:
#   python mcp_server.py
#
# Server starts at: http://127.0.0.1:8100/mcp
# ============================================================

import os
import json
import base64
import asyncio
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    SCENE_MANIFEST,
    CHARACTER_DB,
    IMAGE_ASSETS_DIR,
    MCP_SERVER_PORT,
    FAISS_INDEX_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── Ensure output directories exist ──────────────────────────────────────────
Path(IMAGE_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
Path("./outputs").mkdir(parents=True, exist_ok=True)

mcp = FastMCP("writers_room")


# ============================================================
# TOOL 1 — generate_screenplay
# Expands a story prompt into a structured JSON scene manifest
# using Groq (Llama 3.x). The JSON schema is enforced via
# a strict system prompt.
# ============================================================

@mcp.tool()
def generate_screenplay(prompt: str, num_scenes: int = 3) -> str:
    """
    Transform a story prompt into a structured multi-scene screenplay JSON.

    Returns a JSON string with schema:
    {
      "title": str,
      "genre": str,
      "logline": str,
      "scenes": [
        {
          "scene_number": int,
          "heading":      str,   // e.g. "INT. DETECTIVE OFFICE - NIGHT"
          "action":       str,   // action/description block
          "dialogue": [
            {"character": str, "line": str, "direction": str}
          ],
          "visual_cues":  str,   // cinematography notes for image gen
          "mood":         str
        }
      ]
    }
    """
    from config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.8)

    system_prompt = f"""You are a professional Hollywood screenwriter.
Generate a {num_scenes}-scene screenplay in STRICT JSON format.

The JSON MUST follow this exact schema (no markdown, no extra text):
{{
  "title": "<story title>",
  "genre": "<genre>",
  "logline": "<one-sentence story summary>",
  "scenes": [
    {{
      "scene_number": 1,
      "heading": "INT./EXT. LOCATION - TIME OF DAY",
      "action": "<action description paragraph>",
      "dialogue": [
        {{"character": "NAME", "line": "<dialogue>", "direction": "(beat)"}}
      ],
      "visual_cues": "<cinematography and visual style notes for AI image generation>",
      "mood": "<emotional tone>"
    }}
  ]
}}

Rules:
- scene_number starts at 1
- heading MUST start with INT. or EXT.
- visual_cues should be detailed enough for image generation
- Return ONLY valid JSON. No markdown fences."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Write a screenplay about: {prompt}"),
    ])
    
    raw = str(response.content).strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

    # Robust JSON extraction for smaller models
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx+1]
        
    try:
        data = json.loads(raw)
        path = Path(SCENE_MANIFEST)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return raw
    except Exception as e:
        return f"ERROR: Failed to parse JSON. Error: {e}\nRaw output: {raw[:200]}..."


# ============================================================
# TOOL 2 — validate_script_structure
# Checks a screenplay (raw text or JSON string) for structural
# correctness: scene headings, character cues, dialogue labels.
# ============================================================

@mcp.tool()
def validate_script_structure(script_text: str) -> str:
    """
    Validate a screenplay for structural correctness.

    Returns a JSON string:
    {
      "is_valid": bool,
      "errors": ["<error description>", ...],
      "warnings": ["<warning>", ...],
      "scene_count": int,
      "character_list": ["NAME", ...]
    }
    """
    from config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.0)

    system_prompt = """You are a professional script supervisor and story analyst.
Analyse the provided screenplay and return a STRICT JSON validation report.

Check for:
1. Scene headings (must start with INT. or EXT.)
2. Character cue labels (must be ALL-CAPS before dialogue)
3. Action descriptions (present in each scene?)
4. Dialogue structure (character name followed by lines)
5. Continuity issues (unnamed characters, orphaned scenes)

Return ONLY this JSON (no markdown):
{
  "is_valid": true/false,
  "errors": ["<critical structural error>", ...],
  "warnings": ["<minor issue>", ...],
  "scene_count": <int>,
  "character_list": ["<CHAR NAME>", ...]
}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Validate this screenplay:\n\n{script_text}"),
    ])
    
    raw = str(response.content).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

    # Robust JSON extraction
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx+1]

    try:
        json.loads(raw)
        return raw
    except Exception as e:
        return f"ERROR: Failed to parse validation JSON. Error: {e}\nRaw output: {raw[:300]}..."


# ============================================================
# TOOL 3 — extract_characters
# Parses the scene_manifest JSON and produces a rich character
# profile database for each named character.
# ============================================================

@mcp.tool()
def extract_characters(scene_manifest_json: str) -> str:
    """
    Extract structured character profiles from a scene manifest JSON.

    Returns a JSON string:
    {
      "CHARACTER_NAME": {
        "role":        "protagonist | antagonist | supporting | minor",
        "first_scene": int,
        "traits":      ["trait1", "trait2"],
        "appearance":  "<detailed visual description for image generation>",
        "voice":       "<speech pattern and tone>",
        "arc":         "<character arc summary>",
        "relationships": {"OTHER_NAME": "relationship type"}
      }
    }
    """
    from config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.3)

    system_prompt = """You are a character analyst and casting director.
Given a screenplay JSON, extract detailed character profiles for every named character.

Return ONLY this JSON (no markdown, no extra text):
{
  "CHARACTER_NAME": {
    "role": "protagonist | antagonist | supporting | minor",
    "first_scene": <scene_number: int>,
    "traits": ["<personality trait>", ...],
    "appearance": "<highly detailed visual description suitable for AI image generation: age, build, clothing, hair, distinguishing features>",
    "voice": "<speech pattern, accent, tone>",
    "arc": "<brief character arc>",
    "relationships": {"OTHER_CHAR": "<relationship>"}
  }
}

Be as visually descriptive as possible in 'appearance' — it will be used to generate images."""

    # ── [Continuity Memory] Auto-fetch past characters from FAISS ─────────────
    # We use the logline or the whole text as a query to find relevant past chars
    try:
        manifest_dict = json.loads(scene_manifest_json)
        query = manifest_dict.get("logline", "main characters")
    except Exception:
        query = "main characters"

    past_context = search_memory(query, k=3)
    if past_context and past_context != "[]":
        system_prompt += f"\n\n[MEMORY/CONTINUITY CONTEXT]\nHere are characters/scripts from the past:\n{past_context}\nIf any character from the current script matches a past character identity above, rigorously use their previous appearance, traits, and role to maintain continuity!"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract characters from:\n\n{scene_manifest_json}"),
    ])
    
    raw = str(response.content).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

    # Robust JSON extraction
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx+1]

    try:
        data = json.loads(raw)
        path = Path(CHARACTER_DB)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        
        # ── [Continuity Memory] Save these new profiles to FAISS ──────────────
        try:
            for char_name, char_data in data.items():
                store_in_memory(
                    text=f"Character Name: {char_name}\nProfile: {json.dumps(char_data)}",
                    metadata={"type": "character", "name": char_name}
                )
        except Exception as memory_err:
            print(f"Warning: Failed to save character to memory: {memory_err}")
            
        return raw
    except Exception as e:
        return f"ERROR: Failed to parse character JSON. Error: {e}\nRaw: {raw[:200]}..."


# ============================================================
# TOOL 4 — generate_character_image
# Calls the Gemini image generation API to create a cinematic
# character portrait and saves it to outputs/image_assets/.
# ============================================================

@mcp.tool()
def generate_character_image(character_name: str, visual_description: str) -> str:
    """
    Generate a cinematic character portrait using the Gemini image generation API.

    Args:
        character_name:    The character's name (used as filename).
        visual_description: Detailed visual description from the character profile.

    Returns:
        Absolute path to the saved PNG file, or an error message string.
    """
    from google import genai
    from google.genai import types
    from config import get_dynamic_setting

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    image_model_name = get_dynamic_setting("IMAGE_GEN_MODEL", "gemini-2.5-flash-image")

    image_prompt = (
        f"Cinematic character portrait, professional film still. "
        f"Character: {character_name}. "
        f"Description: {visual_description}. "
        f"Style: dramatic lighting, high detail, photorealistic, movie poster quality."
    )

    response = client.models.generate_content(
        model=image_model_name,
        contents=image_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extract the first image part from the response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_bytes = base64.b64decode(part.inline_data.data)
            safe_name = character_name.replace(" ", "_").replace("/", "_")
            out_path  = Path(IMAGE_ASSETS_DIR) / f"{safe_name}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)
            return str(out_path.resolve())

    return f"ERROR: Image generation produced no image part for '{character_name}'."


# ============================================================
# TOOL 5 — save_scene_manifest
# Persists the scene manifest JSON to disk.
# ============================================================

@mcp.tool()
def save_scene_manifest(manifest_json: str) -> str:
    """
    Save the scene manifest JSON. (This is now handled automatically).
    """
    return "Scene manifest saved."


# ============================================================
# TOOL 6 — save_character_db
# Persists the character database JSON to disk.
# ============================================================

@mcp.tool()
def save_character_db(character_db_json: str) -> str:
    """
    Save the character database JSON. (This is now handled automatically).
    """
    return "Character DB saved."


# ============================================================
# TOOL 7 — search_memory
# Semantic search over the FAISS vector store for past
# scripts, character references, and image metadata.
# ============================================================

@mcp.tool()
def search_memory(query: str, k: int = 4) -> str:
    """
    Semantic similarity search over the Writer's Room FAISS memory.

    Returns a JSON array of matching chunks:
    [{"content": str, "metadata": dict, "score": float}, ...]
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    if not Path(FAISS_INDEX_PATH).exists():
        return json.dumps([])

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"trust_remote_code": True},
    )
    store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    results = store.similarity_search_with_score(query, k=k)

    return json.dumps([
        {
            "content":  doc.page_content,
            "metadata": doc.metadata,
            "score":    float(score),
        }
        for doc, score in results
    ])


# ============================================================
# TOOL 8 — store_in_memory
# Embeds a text chunk and stores it in the FAISS vector store
# for future retrieval across sessions.
# ============================================================

@mcp.tool()
def store_in_memory(text: str, metadata: str | dict = "{}") -> str:
    """
    Embed and store a text chunk in the Writer's Room FAISS memory.

    Args:
        text:     The text to embed and store.
        metadata: JSON string of metadata (e.g. {"type": "script", "title": "..."}).

    Returns:
        Confirmation message.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"trust_remote_code": True},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([text], metadatas=[meta_dict])

    faiss_path = Path(FAISS_INDEX_PATH)
    if faiss_path.exists():
        store = FAISS.load_local(
            str(faiss_path), embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embeddings)

    faiss_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(faiss_path))
    return f"Stored {len(docs)} chunk(s) in FAISS memory at {faiss_path.resolve()}"


# ============================================================
# MCP RESOURCES — Static context the agents can read
# ============================================================

@mcp.resource("resource://output-schema")
def get_output_schema() -> str:
    """The Writer's Room output schema documentation."""
    return """
Writer's Room — Output Schema Reference
========================================

scene_manifest.json:
  title, genre, logline, scenes[]
    → scene_number, heading, action, dialogue[], visual_cues, mood

character_db.json:
  CHARACTER_NAME → role, first_scene, traits[], appearance, voice, arc, relationships{}

image_assets/:
  <CHARACTER_NAME>.png  (cinematic portrait, ~1024x1024)

faiss_index/:
  FAISS vector store for semantic memory across sessions
"""


@mcp.resource("resource://prompt-templates")
def get_prompt_templates() -> str:
    """Reusable prompt templates for the Writer's Room agents."""
    return """
SCRIPTWRITER_SYSTEM:
  You are a professional Hollywood screenwriter. Use the generate_screenplay
  MCP tool to transform the user's story idea into a structured screenplay.
  Always invoke the tool — never generate the screenplay directly.

VALIDATOR_SYSTEM:
  You are a script structure auditor. Use the validate_script_structure MCP
  tool on the provided screenplay. Report errors clearly.

CHARACTER_DESIGNER_SYSTEM:
  You are a casting director and character analyst. Use the extract_characters
  MCP tool on the scene manifest. Produce rich, visually detailed profiles.

IMAGE_SYNTHESIZER_SYSTEM:
  You are a visual director. For each character in the character_db, use the
  generate_character_image MCP tool with their appearance description.
  Save each image and collect all file paths.
"""


# ============================================================
# MCP PROMPTS — Templates agents can request by name
# ============================================================

@mcp.prompt()
def screenplay_prompt(story_idea: str, num_scenes: int = "3") -> str:
    """Prompt template for the Scriptwriter agent."""
    return f"""
You are a professional Hollywood screenwriter working on a new project.

Story Idea: {story_idea}
Number of Scenes: {num_scenes}

Use the generate_screenplay tool to expand this into a full structured screenplay.
After generating, use the save_scene_manifest tool to persist it.
Finally, use store_in_memory to save the script for future reference.
"""


@mcp.prompt()
def character_design_prompt(scene_manifest_json: str) -> str:
    """Prompt template for the Character Designer agent."""
    return f"""
You are a casting director and character analyst for a new film.

Scene Manifest:
{scene_manifest_json[:500]}...

Use the extract_characters tool to build a detailed profile for every named character.
Ensure the 'appearance' field is rich enough for AI image generation.
Then use save_character_db to persist the profiles.
"""


# ============================================================
# ENTRY POINT — Run the server
# ============================================================

if __name__ == "__main__":
    print(f"Starting Writer's Room MCP Server on port {MCP_SERVER_PORT}...")
    print(f"  URL: http://127.0.0.1:{MCP_SERVER_PORT}/mcp")
    print(f"  Tools: 8 registered")
    print(f"  Resources: 2 registered")
    print(f"  Prompts: 2 registered")
    print(f"  Press Ctrl+C to stop.\n")
    mcp.run(transport="streamable-http")
