import logging
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


async def free_comfyui_vram(base_url: str = "http://127.0.0.1:8188") -> bool:
    """
    Tell ComfyUI to unload its Flux models from VRAM and RAM.
    Call this BEFORE loading Qwen via Ollama so they don't compete for memory.
    ComfyUI keeps its models hot between runs — this clears that cache.
    """
    logger.info("[VRAMManager] Requesting ComfyUI to free VRAM/RAM...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"unload_models": True, "free_memory": True}
            async with session.post(
                f"{base_url}/free",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logger.info("[VRAMManager] ComfyUI VRAM freed successfully.")
                    return True
                else:
                    body = await resp.text()
                    logger.warning("[VRAMManager] ComfyUI /free returned %s: %s", resp.status, body)
                    return False
    except Exception as e:
        logger.warning("[VRAMManager] Could not reach ComfyUI to free memory (may be offline): %s", e)
        return False


async def offload_ollama_model(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Unload an Ollama model from VRAM by setting keep_alive to 0.
    Call this BEFORE running ComfyUI image generation.
    """
    logger.info("[VRAMManager] Attempting to offload Ollama model: %s", model_name)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"model": model_name, "keep_alive": 0}
            async with session.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status == 200:
                    logger.info("[VRAMManager] Successfully offloaded %s.", model_name)
                    return True
                else:
                    logger.warning(
                        "[VRAMManager] Offload returned %s: %s",
                        response.status,
                        await response.text(),
                    )
                    return False
    except Exception as e:
        logger.error("[VRAMManager] Failed to offload Ollama model %s: %s", model_name, e)
        return False


async def wait_for_vram_clear(wait_seconds: float = 3.0) -> None:
    """
    Halt execution briefly to let the GPU driver finish freeing VRAM.
    """
    logger.info("[VRAMManager] Waiting %.1fs for VRAM to clear...", wait_seconds)
    await asyncio.sleep(wait_seconds)
    logger.info("[VRAMManager] Wait complete.")
