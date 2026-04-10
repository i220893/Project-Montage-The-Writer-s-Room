# ComfyUI + Flux Local Image Generation Integration

## Overview

This plan integrates your local **ComfyUI** server (running Flux Dev/Schnell) as the **default** image generation backend for the Writer's Room, while retaining Gemini as a fallback. Because your RTX 5060 only has **8 GB VRAM**, Qwen 3.5 (or any other Ollama LLM) and Flux **cannot share VRAM simultaneously** — so the pipeline must offload the LLM before calling ComfyUI, then optionally reload it afterward.

---

## User Review Required

> [!IMPORTANT]
> **Confirm the model name you installed.** "Flux 2 Klein 9B" is not a standard Flux release name. The current public Flux family is:
> - **FLUX.1-schnell** (fast, 4-step, 12B params — community quantized versions can fit in 8 GB)
> - **FLUX.1-dev** (high quality, 28-step, 12B params — needs quantization to fit in 8 GB)
>
> Quantized versions (GGUF, NF4, Q8) are commonly used with ComfyUI. Please confirm which model file (`.safetensors` / `.gguf`) you installed and the exact ComfyUI workflow JSON name you plan to use.

> [!CAUTION]
> **Ollama offloading requires a brief wait.** Calling `ollama stop` (or the API equivalent) releases VRAM, but there is a short gap (~1-3 seconds) before VRAM is fully freed. The ComfyUI call will be delayed by this pause. We'll handle this with a small async sleep + retry.

> [!WARNING]
> **ComfyUI must be running before the pipeline reaches the image synthesis step.** The Writer's Room MCP server will check if ComfyUI is reachable at startup and warn the user if it is not. Users must start ComfyUI separately (or we can auto-launch it as a subprocess — see Open Questions).

---

## Architecture Overview

```
User submits story idea
         │
         ▼
  [Qwen 3.5 / Groq / Gemini LLM]   ← All text agents run here
  • Scriptwriter
  • Validator
  • Character Designer
         │
         ▼
 ┌───────────────────────────────────┐
 │  Pre-Image-Gen: VRAM Offload Step │
 │  • ollama stop <model>             │
 │  • Wait for VRAM to clear         │
 └───────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Image Synthesizer Node            │
  │  • Route: comfyui (default) OR     │
  │           gemini (fallback)        │
  │  • Calls generate_character_image  │
  │    MCP tool for each character     │
  └─────────────────────────────────────┘
         │
         ▼
 ┌───────────────────────────────────┐
 │  Post-Image-Gen: LLM Reload       │
 │  (Optional — skip if session over) │
 │  • Warm up Ollama model again      │
 └───────────────────────────────────┘
```

---

## Phase 1: ComfyUI Setup (Manual Steps — You Do This)

### Step 1.1 — Enable the ComfyUI API server

The ComfyUI desktop app runs on `http://127.0.0.1:8188` by default. Confirm it is accessible.

If using the portable/standalone ComfyUI:
```powershell
# In PowerShell, from your ComfyUI install directory:
python main.py --listen 127.0.0.1 --port 8188
```

If using the **ComfyUI Desktop App**, the server starts automatically when you open the app.

### Step 1.2 — Export the API workflow JSON

1. Open ComfyUI in browser at `http://127.0.0.1:8188`
2. Load your Flux workflow (the one with the model you installed)
3. Click the gear ⚙️ → enable **"Dev Mode"**
4. A new **"Save (API format)"** button appears — click it
5. Save the file as `comfyui_flux_workflow.json` inside the Writer's Room project at:
   ```
   Writer's Room/
   └── comfyui/
       └── comfyui_flux_workflow.json
   ```

> [!NOTE]
> The API format JSON is different from the normal save format. You MUST use API format — the normal format will not work with the HTTP `/prompt` endpoint.

### Step 1.3 — Identify the key node IDs in your workflow

The Python code needs to know which node IDs to inject the text prompt and seed into. Open the API JSON and look for:
- The **`CLIPTextEncode`** node (or `CLIPTextEncodeFlux`) — this gets the image prompt
- The **`KSampler`** node — this contains the seed
- The **`SaveImage`** node — outputs the final file

We'll write the code to handle standard Flux workflow node names automatically.

---

## Phase 2: New Files to Create

### `comfyui/comfyui_client.py` — [NEW]

A dedicated async Python client that:
1. Checks if ComfyUI is reachable (`GET /system_stats`)
2. Loads the workflow JSON template
3. Injects the character prompt + a random seed
4. POSTs to `/prompt` to queue the generation
5. Monitors via WebSocket `/ws?clientId=<uuid>` for completion
6. Fetches the image via `/view?filename=<name>` 
7. Saves the image to `outputs/image_assets/`
8. Returns the saved file path

**Key function signatures:**
```python
async def check_comfyui_reachable(base_url: str = "http://127.0.0.1:8188") -> bool
async def generate_image_comfyui(
    character_name: str,
    prompt: str,
    workflow_path: str,
    base_url: str = "http://127.0.0.1:8188",
    output_dir: str = "./outputs/image_assets",
    timeout: int = 300,
) -> str  # returns saved file path or "ERROR: ..."
```

### `comfyui/vram_manager.py` — [NEW]

Handles VRAM management for the RTX 5060 8 GB constraint:
```python
async def offload_ollama_model(model_name: str, base_url: str) -> bool
async def reload_ollama_model(model_name: str, base_url: str) -> bool
async def wait_for_vram_clear(wait_seconds: float = 3.0) -> None
```

Uses the Ollama REST API:
- `POST /api/generate` with `{"model": "<name>", "keep_alive": 0}` → unloads model
- The existing Ollama warm-up pattern for reloading

---

## Phase 3: Modified Files

### `mcp_server.py` — [MODIFY]

Modify **TOOL 4 (`generate_character_image`)** to:
1. Read `IMAGE_GEN_BACKEND` setting (new) from `outputs/settings.json`
2. If `"comfyui"` (default): call `comfyui_client.generate_image_comfyui()`
3. If `"gemini"` (fallback): use existing Gemini code path
4. **Before calling ComfyUI**: call `vram_manager.offload_ollama_model()` if provider is `"ollama"`

New tool signature remains identical (backward compatible):
```python
@mcp.tool()
def generate_character_image(character_name: str, visual_description: str) -> str:
```

The routing logic is entirely internal to the tool — no changes needed in agents.

### `config.py` — [MODIFY]

Add new configuration constants:
```python
# ── ComfyUI (local image generation) ─────────────────────────────────────────
COMFYUI_BASE_URL: str   = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
COMFYUI_WORKFLOW: str   = os.getenv("COMFYUI_WORKFLOW", "./comfyui/comfyui_flux_workflow.json")
COMFYUI_TIMEOUT: int    = int(os.getenv("COMFYUI_TIMEOUT", "300"))  # seconds

# ── Image generation backend ─────────────────────────────────────────────────
# Default: "comfyui" | Fallback: "gemini"
IMAGE_GEN_BACKEND: str  = os.getenv("IMAGE_GEN_BACKEND", "comfyui")
```

### `app.py` — [MODIFY]

Update the Chainlit Settings widget in `on_start()`:

**Replace the current `IMAGE_GEN_MODEL` select widget with two new widgets:**

```python
# 1. Image Generation Backend (new)
cl.input_widget.Select(
    id="IMAGE_GEN_BACKEND",
    label="Image Generation Backend",
    values=["comfyui", "gemini"],
    initial_index=["comfyui", "gemini"].index(
        get_dynamic_setting("IMAGE_GEN_BACKEND", "comfyui")
    ),
),

# 2. ComfyUI Workflow (new — only applies when backend=comfyui)
cl.input_widget.Select(
    id="COMFYUI_WORKFLOW_NAME",
    label="ComfyUI Workflow",
    values=["flux_dev", "flux_schnell"],
    initial_index=0,
),

# 3. Gemini Image Model (kept — used when backend=gemini)  
cl.input_widget.Select(
    id="IMAGE_GEN_MODEL",
    label="Gemini Image Model (fallback only)",
    values=["gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image"],
    initial_index=0,
),
```

Add a **ComfyUI status check** in `on_start()` that pings `127.0.0.1:8188` and shows a green/red status message.

### `.env.example` — [MODIFY]

Add the new variables:
```env
# ── ComfyUI (Local Image Generation — Default) ────────────────────────────
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_WORKFLOW=./comfyui/comfyui_flux_workflow.json
COMFYUI_TIMEOUT=300
IMAGE_GEN_BACKEND=comfyui
```

### `requirements.txt` — [MODIFY]

Add:
```
websocket-client>=1.8.0
aiohttp>=3.9.0
```

(`aiohttp` for async HTTP to ComfyUI; `websocket-client` for WebSocket monitoring)

---

## Phase 4: VRAM Offload Flow (Critical Implementation Detail)

The `generate_character_image` MCP tool is a **synchronous** function (decorated with `@mcp.tool()`). To run async offloading inside it, we'll use `asyncio.run()` or run via a thread executor. Here's the exact flow:

```
generate_character_image() called for character #1
│
├─ [Check] backend == "comfyui"?  YES
│
├─ [Check] already_offloaded flag in module-level state?  NO
│   ├─ Call offload_ollama_model(model_name, ollama_url)
│   │   └─ POST /api/generate {"model": "qwen2.5:...", "keep_alive": 0}
│   ├─ Sleep 3 seconds (VRAM drain)
│   └─ Set already_offloaded = True
│
├─ Generate image via ComfyUI WebSocket client
│   ├─ POST /prompt with workflow JSON + character prompt + random seed
│   ├─ WebSocket monitor until "execution_complete" event
│   ├─ GET /view?filename=<output_file>
│   └─ Save to outputs/image_assets/<character_name>.png
│
└─ Return file path

[For character #2, #3... the offload step is skipped (flag is True)]
```

> [!NOTE]
> The `already_offloaded` flag is a module-level variable in `mcp_server.py` that resets to `False` at server startup. This means for each new pipeline run, the model is offloaded once (before the first image), not once per character.

---

## Phase 5: ComfyUI Workflow JSON Injection Strategy

The workflow JSON has numbered node IDs. We need to find the right nodes dynamically. The `comfyui_client.py` will scan the workflow for node types instead of hardcoding IDs:

```python
def inject_prompt_into_workflow(workflow: dict, prompt: str, seed: int) -> dict:
    """Find CLIPTextEncode (positive) and KSampler nodes, inject prompt + seed."""
    w = copy.deepcopy(workflow)
    for node_id, node in w.items():
        class_type = node.get("class_type", "")
        
        # Inject prompt into positive CLIP text encoder
        if class_type in ("CLIPTextEncode", "CLIPTextEncodeFlux"):
            inputs = node.get("inputs", {})
            if "text" in inputs and isinstance(inputs.get("text"), str):
                # Heuristic: the positive prompt node contains the main description
                inputs["text"] = prompt
        
        # Inject seed into sampler
        if class_type in ("KSampler", "KSamplerAdvanced"):
            node["inputs"]["seed"] = seed
    
    return w
```

> [!IMPORTANT]
> After you export your workflow JSON, share it with me so I can verify that the node detection heuristic above works for your specific Flux workflow. Some custom Flux workflows have non-standard node structures.

---

## Phase 6: Chainlit UI Changes (Detailed)

### Session Start Status Banner

When a chat session starts, `on_start()` will:
1. Ping `http://127.0.0.1:8188/system_stats` with a 2-second timeout
2. Display inline status:
   - ✅ `ComfyUI Online — Flux model ready` (green)
   - ⚠️ `ComfyUI Offline — will fallback to Gemini` (yellow)

### Settings Panel (Updated Layout)

| Setting | Widget | Values | Default |
|---|---|---|---|
| Agent LLM Provider | Select | groq / google / ollama | ollama |
| Ollama Model | Select | qwen2.5:7b-instruct-q4_K_M / llama3.2 | qwen2.5... |
| **Image Backend** | **Select** | **comfyui / gemini** | **comfyui** |
| ComfyUI Workflow | Select | flux_dev / flux_schnell | flux_dev |
| Gemini Image Model | Select | gemini-2.0-flash... / gemini-2.5... | gemini-2.0... |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Should ComfyUI be auto-launched?**
> The Writer's Room could attempt to launch ComfyUI as a subprocess if it's not already running. This requires knowing the exact path to your ComfyUI install. Do you want this, or will you always start ComfyUI manually before running the Writer's Room?

> [!IMPORTANT]
> **Q2: Which exact Flux model did you install?**
> The workflow JSON and prompt format differs slightly between:
> - FLUX.1-schnell (fast, lower quality, ~4 steps)
> - FLUX.1-dev (slower, higher quality, ~20-28 steps)  
> - Custom quantized variants (e.g., FLUX.1-dev Q4_K_M GGUF)
> Please confirm so I can set appropriate sampler steps in the workflow.

> [!IMPORTANT]
> **Q3: Should the LLM be reloaded after image generation?**
> After Flux finishes, Ollama is still unloaded. If the user starts another pipeline immediately, Qwen will need to reload (adds ~20-30 seconds). Options:
> - A) Auto-reload after image gen completes (adds a small delay but keeps LLM warm)
> - B) Let Ollama load on-demand for the next request (zero extra delay now, but slow on next use)
>
> Which do you prefer?

---

## Verification Plan

### Automated Tests
1. **ComfyUI connectivity**: Ping `127.0.0.1:8188/system_stats` — expect 200 OK
2. **Workflow JSON validity**: Load and validate the exported JSON — check for `CLIPTextEncode` and `KSampler` nodes
3. **Ollama offload**: Call offload function and verify VRAM usage drops via `nvidia-smi`
4. **End-to-end Generate mode**: Run full pipeline with a short story — verify image saved at `outputs/image_assets/`
5. **Fallback test**: Set backend to `gemini` in settings — verify Gemini path activates

### Manual Verification
- Open Chainlit UI → verify `Image Backend: comfyui` is default in settings
- Submit a story → watch pipeline logs for `[ComfyUI]` log lines
- Confirm images appear in the deliverables section of the chat
- Switch backend to `gemini` in settings → submit another story → verify Gemini images generated

---

## Implementation Order

1. `comfyui/` directory + export workflow JSON (you do this in ComfyUI)
2. `config.py` updates (new constants)
3. `comfyui/vram_manager.py` (Ollama offload logic)
4. `comfyui/comfyui_client.py` (ComfyUI API client)
5. `mcp_server.py` — update `generate_character_image` tool
6. `app.py` — update settings panel + add ComfyUI status check
7. `.env.example` + `requirements.txt` updates
8. End-to-end test run
