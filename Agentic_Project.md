## **THE WRITER’S ROOM: Autonomous Story and Image Generation Layer**

### **1. Introduction**
[cite_start]Phase 1 establishes the foundation of the **PROJECT MONTAGE** system, transforming raw human intent into structured, machine-interpretable narrative representations[cite: 1]. [cite_start]Unlike traditional pipelines, this module operates as a multi-agent creative system using **LangGraph** stateful workflows[cite: 2].

**Agent Characteristics:**
* [cite_start]Collaborate autonomously[cite: 2].
* [cite_start]Maintain shared memory[cite: 2].
* [cite_start]Operate under **MCP-based** (Model Context Protocol) tool discovery constraints[cite: 2].
* [cite_start]Orchestrated via stateful workflows[cite: 2].

---

### **2. System Objectives**
[cite_start]The primary goal is to produce three core outputs[cite: 3]:
* [cite_start]`scene_manifest.json`: A structured screenplay[cite: 3].
* [cite_start]`character_db.json`: A persistent character identity store[cite: 3].
* [cite_start]`image_assets/`: AI-generated character visuals[cite: 3].

**Additional Objectives:**
* [cite_start]Ensure compatibility with downstream video/audio agents[cite: 3].
* [cite_start]Support both manual script input and autonomous LLM generation[cite: 3].

---

### **3. Architectural Design Principles**

#### **3.1 Multi-Agent Collaboration Model**
The system follows a **Supervisor-Worker** hierarchical model:
* **Supervisor:** Implicitly handled via LangGraph routing.
* **Worker Agents:** Scriptwriter, Character Designer, Image Synthesis, and Validator.
* **Interaction:** Agents have clearly scoped responsibilities, interact via a shared state, and use MCP tools dynamically.

#### **3.2 MCP-based Tool Discovery Constraint**
[cite_start]A strict constraint is enforced: **All tools must be discovered dynamically via MCP** with no hardcoded APIs[cite: 4].
* [cite_start]Agents query the MCP registry at runtime[cite: 5].
* [cite_start]Tools are invoked via structured JSON schemas[cite: 5].

#### **3.3 Stateful Memory System**
All agents interact with a persistent memory layer (Vector DB like **ChromaDB** or **FAISS**) to store script history, character metadata, and image references. This supports continuity, failure recovery, and future personalization.

---

### **4. Script Intake Logic**
This phase implements dual-mode ingestion:

* **Mode 1: Direct Script Injection (Manual)**
    1.  User uploads script.
    2.  [cite_start]**Script Validator Agent** checks structure (Scene headings, Dialogue labels, Action descriptions)[cite: 6].
    3.  Conversion to standardized JSON.

* **Mode 2: Autonomous Script Generation**
    1.  User provides a prompt.
    2.  [cite_start]**Scriptwriter Agent** expands it into a multi-scene screenplay with dialogues and visual cues[cite: 7, 8].

---

### **5. Agent Definitions**

| Agent | Role | Key Responsibilities |
| :--- | :--- | :--- |
| **Scriptwriter** | Transform prompts into production scripts. | [cite_start]Scene segmentation, dialogue, and consistency[cite: 9]. |
| **Script Validator**| Ensure correctness of manual scripts. | [cite_start]Validation of headers, labels, and structure[cite: 10]. |
| **Human-in-the-loop**| Checkpoint control. | [cite_start]Prevents hallucinations and aligns with user intent[cite: 11]. |
| **Character Designer**| Formalize character identities. | [cite_start]Extracts traits, appearance, and maintains consistency[cite: 12]. |
| **Image Synthesizer** | Generate visual representations. | [cite_start]Implementation via Stable Diffusion/ComfyUI via MCP[cite: 13]. |

---

### **6. Deliverables & Evaluation**
The final deliverables include the **Scene Manifest**, **Character Database**, and the **LangGraph Workflow**.

**Evaluation Rubric (Top Criteria):**
* **Agent Definition (20 pts):** Clear roles and reasoning loops.
* **Script Quality (15 pts):** Structured and coherent scenes.
* **MCP Integration (15 pts):** Proper tool usage without hardcoding.
* **LangGraph Workflow (10 pts):** StateGraph correctness.
* **Human-in-the-Loop (10 pts):** Proper checkpoint design.