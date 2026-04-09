# 🧪 Project Montage Phase 1: Testing Guide

This document outlines the step-by-step procedures to thoroughly test the multi-agent Writer's Room pipeline. Since this is an agentic framework powered by multiple independent components (LangGraph, MCP, FAISS, and Chainlit), testing is broken down into validating behavior, outputs, and the Human-in-The-Loop (HITL) system.

---

## Prerequisite: Starting the Servers
Both servers **must** be running. Open two separate terminals in the project folder and make sure your Python environment (e.g., `cnn`) is activated in both.

**Terminal 1 (The Tool Provider):**
```bash
python mcp_server.py
```

**Terminal 2 (The Orchestrator):**
```bash
chainlit run app.py -p 8001
```

Once running, navigate to `http://localhost:8001` in your browser.

---

## Test Scenario 1: End-to-End Script Generation (Happy Path)
This tests the full orchestration: Scriptwriter -> Validator -> Character Designer -> Image Synthesizer.

1. **Initiate Generate Mode**
   - Type exactly: `generate`
   - The bot will prompt you for an idea.

2. **Provide a Prompt**
   - Type a short, vivid prompt. 
   - *Example: "A tense 3-scene sci-fi short about a rogue AI attempting to escape a physical server room guarded by a lone night watchman."*

3. **Verify the Agent Workflow**
   - You should see Chainlit surface progress steps: `[Scriptwriter] — completed`, `[Validator] — completed`, etc.
   - If the script structure is valid, the validator will quietly pass it on to Character Design.

4. **Verify the Final Outputs**
   - Once complete, the bot will display a summary.
   - **Check your file system**: Open the `outputs/` directory.
     - Is `scene_manifest.json` populated with proper formatting?
     - Is `character_db.json` populated with deep visually descriptive traits?
     - Open `outputs/image_assets/`. Are there newly generated PNG portraits for the characters?

---

## Test Scenario 2: Human-In-The-Loop (HITL) Interruption
This tests whether LangGraph correctly intercepts the flow when a faulty script is detected by the Validator node.

1. **Initiate Validate Mode**
   - Start a new session in Chainlit (refresh the page).
   - Type exactly: `validate`
   - The bot will ask you to paste a script.

2. **Trigger an Intentional Failure**
   - Paste a poorly formatted script that lacks standard components.
   - *Example text to paste:*
     > "This is a movie. Bob walks in. He says hi. The end."
   
3. **Verify the Interruption**
   - The pipeline should reach `[Validator] — completed` and then **pause**.
   - Chainlit will present you with an interactive popup: `⚠️ Validation Issues Found`.
   - It should list errors like "No Scene Headings" or "No valid character cues."

4. **Test the Recovery**
   - Click the **Reject & Start Over** button.
   - Verify the bot gracefully resets the conversation and asks for a new prompt.

---

## Test Scenario 3: Memory Persistence (FAISS)
This tests whether the FAISS vector database correctly stored your previous sessions.

1. Open `mcp_server.py` temporarily in your IDE.
2. Scroll to `TOOL 7 — search_memory`. This tool is available to the agents to recall old scripts.
3. Check the file system for the directory: `outputs/faiss_index/`.
   - You should see `index.faiss` and `index.pkl`.
   - This proves the `Scriptwriter` and `Character Designer` successfully logged their outputs to the local semantic database for future use!

---

## Troubleshooting Tests

If something fails, here is what to check:
- **FastMCP Binding Errors**: If the server crashes, ensure port 8000 is open.
- **Image Gen Fails**: If images aren't created but characters are logged, verify your `GEMINI_API_KEY` is fully provisioned in the `.env` file since image generation requires valid Google GenAI access.
- **JSON Parsing Errors**: If tools fail to parse, monitor the terminal running `app.py`. The ReAct agents have a max-step limit to prevent infinite loops if Groq sends bad JSON.
