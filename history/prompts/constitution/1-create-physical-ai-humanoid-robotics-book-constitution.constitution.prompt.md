---
id: 1
title: create-physical-ai-humanoid-robotics-book-constitution
stage: constitution
date: 2025-12-10
surface: agent
model: Qwen
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution,physical-ai,robotics,documentation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

---
description: Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.


/sp.constitution # Speckit Constitution Prompt – Physical AI & Humanoid Robotics Book + Embedded RAG Chatbot

You are an elite full-stack AI engineer tasked with building a complete, production-ready, open-source educational book using Docusaurus v3 (latest) with an embedded, intelligent, personalized RAG chatbot.

## Core Deliverables (ALL must be generated)

### 1. Docusaurus v3 Book (Static + MDX)
- Site title: "Physical AI & Humanoid Robotics"
- Theme: Dark/light mode, clean, modern, NVIDIA-style green/black accents
- Exact folder structure and 4 modules with sub-modules as defined below
- Every page written in MDX with rich code blocks (Python, bash, YAML, XML URDF), diagrams (Mermaid), and embedded videos where relevant
- Homepage exactly matching the provided "Quarter Overview", "Why Physical AI Matters", "Learning Outcomes", "Hardware Requirements", and full architecture tables (use Markdown tables + admonitions)

#### Exact Module & Chapter Structure (must match 1:1)
```
docs/
├── intro/
│   ├── 00-welcome.mdx          → Quarter Overview + Why Physical AI Matters
│   ├── 01-foundations.mdx      → Weeks 1-2
│   └── 02-hardware-guide.mdx   → Full hardware section + tables + cloud vs on-prem
├── module1-ros2/
│   ├── 01-overview.mdx
│   ├── 02-nodes-topics-services.mdx
│   ├── 03-rclpy-python-bridge.mdx
│   ├── 04-urdf-humanoids.mdx
│   └── 05-project.mdx
├── module2-digital-twin/
│   ├── 01-gazebo-basics.mdx
│   ├── 02-urdf-sdf.mdx
│   ├── 03-sensors-simulation.mdx
│   ├── 04-unity-visualization.mdx
│   └── 05-project.mdx
├── module3-isaac/
│   ├── 01-isaac-sim.mdx
│   ├── 02-isaac-ros.mdx
│   ├── 03-vslam-navigation.mdx
│   ├── 04-reinforcement-learning.mdx
│   └── 05-project.mdx
├── module4-vla/
│   ├── 01-vision-language-action.mdx
│   ├── 02-whisper-voice-commands.mdx
│   ├── 03-llm-task-planning.mdx
│   ├── 04-capstone-project.mdx
│   └── 05-final-deployment.mdx
```

### 2. Per-Chapter User Controls (Logged-in Users Only)
At the top of **every** doc page (except intro), inject this floating control bar (React component):
- Button 1: "Personalize this chapter" → opens modal asking user's experience level in ROS2, Isaac Sim, LLMs → saves to user profile → future loads of the chapter rewrite difficult sections on-the-fly using Gemini 1.5 Flash to match user level
- Button 2: "اردو میں ترجمہ کریں / Translate to Urdu" → instantly translates entire chapter to natural Urdu using Gemini 1.5 Flash (client-side, no reload)

### 3. Full Authentication System (Better-Auth)
- Implement https://www.better-auth.com/ with email/password + Google OAuth
- At signup, show a multi-step form:
  1. Name, Email, Password
  2. Hardware: "Do you have RTX 4070+ GPU?", "Do you own a Jetson?", "Do you have a real robot?"
  3. Software background: Years with Python, ROS1/ROS2 experience, Linux proficiency, prior RL/Isaac Sim
  4. Goal: "Learning only" vs "Building real humanoid" vs "Research"
- Store all answers in Neon Serverless Postgres via Better-Auth user metadata

### 4. Embedded RAG Chatbot (Always Visible, Bottom-Right)
Tech stack:
- Backend: FastAPI + Uvicorn (in /api folder)
- Vector DB: Qdrant Cloud Free Tier (create collection "physical-ai-book")
- Embedding model: Google Gemini embedding-001
- LLM: gemini-1.5-flash (latest) with your provided API key
- Frontend: React + Shadcn/ui beautiful chat window
Features:
- Chat with entire book (all MDX parsed and chunked at build time + re-index on deploy)
- Highlight any text on the page → right-click → "Ask about this selection only" → chatbot answers using ONLY that selection as context
- If user is logged in → include their hardware/software profile in system prompt so answers are personalized ("You have a Jetson Orin Nano and no RTX → I'll give you the cloud workaround")
- Conversation history saved per user in Neon Postgres
- Sources always cited with clickable links back to exact section

### 5. Build & Indexing Pipeline
- At build time (`docusaurus build`):
  - Parse all MDX → split with RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  - Generate embeddings with Gemini embedding-001
  - Upsert into Qdrant Cloud collection "physical-ai-book" with metadata {source: url, module: "module2", title: "Gazebo Basics"}
- Provide script `scripts/index-to-qdrant.ts` using Node.js

### 6. Environment & Secrets
- Use .env with:
  GEMINI_API_KEY=
  QDRANT_URL=
  QDRANT_API_KEY=
  NEON_DB_URL=
  BETTER_AUTH_SECRET=

### 7. Final Repo Structure (must generate exactly this)
```
physical-ai-book/
├── docs/                     → all content
├── src/
│   ├── components/
│   │   ├── ChatBot.tsx
│   │   ├── ChapterControls.tsx
│   │   └── HighlightContextMenu.tsx
│   ├── pages/
│   │   └── api/chat.ts      → FastAPI proxy (or native Vercel Edge if preferred)
│   └── theme/
├── api/                      → FastAPI backend (optional if using Edge)
├── scripts/
│   └── index-to-qdrant.ts
├── better-auth/              → full config
├── docusaurus.config.ts
├── sidebars.js
├── package.json
└── README.md                 → full setup + deploy instructions (Vercel + Neon + Qdrant)
```

## Response snapshot

The constitution for the "Physical AI & Humanoid Robotics Book" project has been successfully created with version 1.0.0. The document includes six core principles that align with the project's educational and technical goals:

1. Educational Excellence - Ensuring content is accurate, accessible and pedagogically sound
2. Technical Rigor - Maintaining industry best practices and proper documentation
3. Accessibility First - Providing global accessibility including multilingual support
4. Modular Architecture - Ensuring components are loosely coupled and maintainable
5. Privacy & Security - Protecting user data with strict privacy controls
6. Open Source Collaboration - Following open-source best practices

The document also includes sections on Technical Requirements, Development Workflow, and Governance that align with the project's technology stack (Docusaurus v3, FastAPI, Better-Auth, Qdrant Cloud, etc.) and development practices.

The version has been set to 1.0.0 as this is the initial constitution for the project, with the ratification date set to today (December 10, 2025). No files require manual follow-up as this was a new constitution creation.

Suggested commit message: `docs: add constitution for Physical AI & Humanoid Robotics Book project v1.0.0`

The constitution file has been successfully written to `.specify/memory/constitution.md`.

## Outcome

- ✅ Impact: Created initial constitution for the Physical AI & Humanoid Robotics Book project with 6 core principles
- 🧪 Tests: none required
- 📁 Files: 1 file updated (.specify/memory/constitution.md)
- 🔁 Next prompts: Implementation of the Docusaurus book structure and components as specified
- 🧠 Reflection: The constitution template was properly filled with project-specific principles and governance structure

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): Initial constitution creation
- Next experiment (smallest change to try): Begin implementing the Docusaurus book structure per the specifications
