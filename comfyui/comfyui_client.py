import os
import json
import uuid
import asyncio
import logging
from pathlib import Path
import aiohttp
import websocket

logger = logging.getLogger(__name__)

async def check_comfyui_reachable(base_url: str = "http://127.0.0.1:8188", timeout: float = 2.0) -> bool:
    """Check if the ComfyUI server is online."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/system_stats", timeout=timeout) as response:
                return response.status == 200
    except Exception:
        return False

def inject_prompt_into_workflow(workflow: dict, prompt: str, seed: int) -> dict:
    """Find prompt, seed, and resolution nodes, and inject our values dynamically."""
    import copy
    w = copy.deepcopy(workflow)

    for node_id, node in w.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        
        # Inject positive prompt
        if class_type == "PrimitiveStringMultiline" and "value" in inputs:
            # We assume the main PrimitiveStringMultiline is the prompt 
            # (especially if title is "Prompt")
            if node.get("_meta", {}).get("title") == "Prompt":
                inputs["value"] = prompt
            elif "A vintage motorcycle" in str(inputs["value"]): # Fallback check for the specific json default
                inputs["value"] = prompt
                
        # Inject seed
        if class_type == "RandomNoise" and "noise_seed" in inputs:
            inputs["noise_seed"] = seed
        if class_type in ("KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"):
            if "seed" in inputs:
                inputs["seed"] = seed
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed

    return w

async def generate_image_comfyui(
    character_name: str,
    prompt: str,
    workflow_path: str,
    base_url: str = "http://127.0.0.1:8188",
    output_dir: str = "./outputs/image_assets",
    timeout: int = 300,
) -> str:
    """Generate an image using the local ComfyUI instance."""
    
    # 1. Load Workflow JSON
    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        return f"ERROR: ComfyUI workflow JSON not found at {workflow_path}"
        
    try:
        workflow_data = json.loads(workflow_file.read_text(encoding="utf-8"))
    except Exception as e:
        return f"ERROR: Failed to parse workflow JSON: {e}"

    # 2. Inject Prompt & Seed
    import random
    seed = random.randint(1, 2**63 - 1)
    prompt_with_suffix = f"Cinematic character portrait, professional film still. Character: {character_name}. Description: {prompt}. Style: dramatic lighting, high detail, photorealistic, movie poster quality."
    
    modified_workflow = inject_prompt_into_workflow(workflow_data, prompt_with_suffix, seed)

    client_id = str(uuid.uuid4())
    payload = {"prompt": modified_workflow, "client_id": client_id}

    # 3. Queue Prompt via HTTP
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/prompt", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return f"ERROR: ComfyUI API returned {resp.status}: {text}"
                resp_json = await resp.json()
                prompt_id = resp_json.get("prompt_id")
    except Exception as e:
        return f"ERROR: Failed to queue ComfyUI workflow: {e}"

    # 4. Monitor via WebSocket
    ws = websocket.WebSocket()
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
    try:
        ws.connect(ws_url)
    except Exception as e:
        return f"ERROR: Failed to connect to ComfyUI websocket: {e}"

    logger.info(f"[ComfyUI] Generating image for {character_name} (Prompt ID: {prompt_id})...")
    
    import time
    start_time = time.time()
    
    # Simple loop to wait for execution_success or execution_cached
    completed = False
    output_filename = None
    
    while not completed:
        if time.time() - start_time > timeout:
            ws.close()
            return f"ERROR: ComfyUI generation timed out after {timeout} seconds."
            
        try:
            ws.settimeout(5.0)
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                mtype = msg.get("type", "")
                data = msg.get("data", {})
                
                if mtype == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                    completed = True
                
                if mtype == "execution_success" and data.get("prompt_id") == prompt_id:
                    completed = True
                
                if mtype == "execution_error" and data.get("prompt_id") == prompt_id:
                    ws.close()
                    err_msg = data.get('exception_message', 'Unknown Error')
                    return f"ERROR: ComfyUI execution failed: {err_msg}"
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            ws.close()
            return f"ERROR: WebSocket read error: {e}"
            
    ws.close()

    # 5. Fetch History to find output filename
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/history/{prompt_id}") as resp:
                history = await resp.json()
                
        # History contains node outputs
        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})
        
        # Find the image filename in outputs (from the SaveImage node)
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for image in node_output["images"]:
                    if image.get("type") == "output":
                        output_filename = image.get("filename")
                        subfolder = image.get("subfolder", "")
                        if subfolder:
                            output_filename = f"{subfolder}/{output_filename}"
                        break
    except Exception as e:
        return f"ERROR: Failed to fetch ComfyUI history: {e}"
        
    if not output_filename:
        history_status = prompt_history.get("status", {})
        err = history_status.get("error")
        if err:
            return f"ERROR: ComfyUI workflow failed: {err.get('exception_message', str(err))}"
        return f"ERROR: No image output found in ComfyUI history for prompt {prompt_id}. History: {json.dumps(prompt_history)[:200]}..."

    # 6. Download the final image
    safe_name = character_name.replace(" ", "_").replace("/", "_")
    out_path = Path(output_dir) / f"{safe_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/view?filename={output_filename}&type=output") as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    out_path.write_bytes(img_bytes)
                    logger.info(f"[ComfyUI] Successfully saved {out_path.resolve()}")
                    return str(out_path.resolve())
                else:
                    return f"ERROR: Failed to download image from ComfyUI ({resp.status})"
    except Exception as e:
        return f"ERROR: Failed to download image from ComfyUI: {e}"
