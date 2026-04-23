# Agentic AI (CS-4015) | Course Project: Phase II
## **The Studio Floor: Video and Audio Synthesis Layer**

---

### ■ Introduction
[cite_start]Phase 2 transforms the structured outputs of Phase 1 into multimodal audiovisual content[cite: 1]. [cite_start]This phase implements a **parallel multi-agent execution architecture** where independent agents process audio and video streams concurrently, synchronize through temporal alignment, and maintain state continuity via shared memory[cite: 2].

### ■ System Objectives
[cite_start]The primary goal is to convert `scene_manifest.json` into tangible audiovisual scenes[cite: 3].
* [cite_start]**Generation Tasks**: Voice (TTS/cloning), video frames, face-mapped characters, and lip-synced outputs[cite: 3].
* [cite_start]**Primary Outputs**: `raw_scenes/scene_01.mp4`, audio tracks (`.wav`), and intermediate frame sequences[cite: 3].

---

### ■ Architectural Design Principles
1. **Parallel Processing Architecture**: Separate audio and video branches that converge at the Lip Sync Agent.
2. **Task Graph-based Execution**: Uses the `get_task_graph` MCP Tool to decompose scenes into independent, parallelizable units.
3. **Stateful Resumability**: Utilizes `commit_memory` to save intermediate outputs, allowing recovery from failures or interruptions.

---

### ■ Agent Definitions & MCP Tools

| Agent | Role & Responsibilities | MCP Tools |
| :--- | :--- | :--- |
| **Scene Parser** | Transforms `scene_manifest.json` into executable tasks; handles segmentation and parallel routing. | `get_task_graph`, `commit_memory` |
| **Voice Synthesis** | Generates emotion-aware speech and voice cloning aligned with character identities. | `voice_cloning_synthesizer` |
| **Video Generation** | Creates scene visuals from character references and environment descriptions. | `query_stock_footage` |
| **Face Swap** | Maps generated characters onto video frames; **must** validate identity before mapping. | `face_swapper`, `identity_validator` |
| **Lip Sync** | Synchronizes audio waveforms with facial movements for temporal consistency. | `lip_sync_aligner` |

---

### ■ Workflow & Synchronization
* [cite_start]**Multimodal Synchronization**: Solves the **Temporal Alignment Problem** by ensuring speech timing matches lip motion and maintaining scene continuity[cite: 7].
* [cite_start]**LangGraph Implementation**: Uses the `Send()` API for branching across five specific nodes: `Scene_parser_node`, `Voice_synth_node`, `Video_gen_node`, `Face_swap_node`, and `Lip_sync_node`[cite: 7].

---

### ■ Deliverables & Evaluation
**Deliverables Checklist:**
- [ ] `raw_scenes/*.mp4`
- [ ] Audio tracks (`.wav`)
- [ ] Task graph logs

**Evaluation Rubric (Total: 70 Marks):**
* **Audio Quality**: 20 Marks (Natural speech synthesis)
* **Video Quality**: 20 Marks (Visual coherence)
* **Parallel Architecture**: 10 Marks (Proper branching implementation)
* **Lip Sync Accuracy**: 10 Marks (Temporal alignment)
* **MCP Tool Usage**: 5 Marks (Correct integration)
* **Fault Tolerance**: 5 Marks (Resumability)