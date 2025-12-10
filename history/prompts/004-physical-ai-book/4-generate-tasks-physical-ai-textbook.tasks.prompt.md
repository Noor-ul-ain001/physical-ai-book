---
id: 4
title: generate-tasks-physical-ai-textbook
stage: tasks
date: 2025-12-10
surface: agent
model: Qwen
feature: physical-ai-book
branch: 004-physical-ai-book
user: user
command: /sp.tasks
labels: [tasks,physical-ai,robotics,textbook,education]
links:
  spec: ../specs/004-physical-ai-book/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/004-physical-ai-book/tasks.md
tests:
 - specs/004-physical-ai-book/checklists/requirements.md
---

## Prompt

---
description: Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/check-prerequisites.ps1 -Json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load design documents**: Read from FEATURE_DIR:
   - **Required**: plan.md (tech stack, libraries, structure), spec.md (user stories with priorities)
   - **Optional**: data-model.md (entities), contracts/ (API endpoints), research.md (decisions), quickstart.md (test scenarios)
   - Note: Not all projects have all documents. Generate tasks based on what's available.

3. **Execute task generation workflow**:
   - Load plan.md and extract tech stack, libraries, project structure
   - Load spec.md and extract user stories with their priorities (P1, P2, P3, etc.)
   - If data-model.md exists: Extract entities and map to user stories
   - If contracts/ exists: Map endpoints to user stories
   - If research.md exists: Extract decisions for setup tasks
   - Generate tasks organized by user story (see Task Generation Rules below)
   - Generate dependency graph showing user story completion order
   - Create parallel execution examples per user story
   - Validate task completeness (each user story has all needed tasks, independently testable)

4. **Generate tasks.md**: Use `.specify.specify/templates/tasks-template.md` as structure, fill with:
   - Correct feature name from plan.md
   - Phase 1: Setup tasks (project initialization)
   - Phase 2: Foundational tasks (blocking prerequisites for all user stories)
   - Phase 3+: One phase per user story (in priority order from spec.md)
   - Each phase includes: story goal, independent test criteria, tests (if requested), implementation tasks
   - Final Phase: Polish & cross-cutting concerns
   - All tasks must follow the strict checklist format (see Task Generation Rules below)
   - Clear file paths for each task
   - Dependencies section showing story completion order
   - Parallel execution examples per story
   - Implementation strategy section (MVP first, incremental delivery)

5. **Report**: Output path to generated tasks.md and summary:
   - Total task count
   - Task count per user story
   - Parallel opportunities identified
   - Independent test criteria for each story
   - Suggested MVP scope (typically just User Story 1)
   - Format validation: Confirm ALL tasks follow the checklist format (checkbox, ID, labels, file paths)

Context for task generation: • Setting up Docusaurus v3 with exact theme and structure • Writing all 23 real, expert-level MDX files with working code examples (ROS2, URDF, Gazebo, Isaac Sim, VLA, Whisper → Gemini → ROS2 chains) • Implementing Better-Auth with multi-step signup wizard and Neon Postgres metadata • Building ChapterControls.tsx (Personalize + Urdu translation using gemini-1.5-flash) • Building full RAG chatbot with Shadcn/UI, Qdrant, Gemini embedding-001 + gemini-1.5-flash • Highlight-to-chat right-click feature • Build-time indexing script (scripts/index-to-qdrant.ts) • Backend (Vercel Edge Functions preferred) • Environment setup, README, deployment to Vercel + Neon + Qdrant • All testing tasks Rules: - Every MDX file is its own task (e.g., "Write docs/module4-vla/04-capstone-project.mdx with full working VLA pipeline") - Every component is its own task - Every API route is its own task - Include tasks for creating .env.example, package.json scripts, sidebar config - Include final verification tasks ("Test Urdu translation on Module 2 Chapter 3", "Test highlight-to-chat with selected URDF code", etc.) Start numbering from 001 and go until the project is 100% complete

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.

## Task Generation Rules

**CRITICAL**: Tasks MUST be organized by user story to enable independent implementation and testing.

**Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach.

### Checklist Format (REQUIRED)

Every task MUST strictly follow this format:

```text
- [ ] [TaskID] [P?] [Story?] Description with file path
```

**Format Components**:

1. **Checkbox**: ALWAYS start with `- [ ]` (markdown checkbox)
2. **Task ID**: Sequential number (T001, T002, T003...) in execution order
3. **[P] marker**: Include ONLY if task is parallelizable (different files, no dependencies on incomplete tasks)
4. **[Story] label**: REQUIRED for user story phase tasks only
   - Format: [US1], [US2], [US3], etc. (maps to user stories from spec.md)
   - Setup phase: NO story label
   - Foundational phase: NO story label  
   - User Story phases: MUST have story label
   - Polish phase: NO story label
5. **Description**: Clear action with exact file path

**Examples**:

- ✅ CORRECT: `- [ ] T001 Create project structure per implementation plan`
- ✅ CORRECT: `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
- ✅ CORRECT: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
- ✅ CORRECT: `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`
- ❌ WRONG: `- [ ] Create User model` (missing ID and Story label)
- ❌ WRONG: `T001 [US1] Create model` (missing checkbox)
- ❌ WRONG: `- [ ] [US1] Create model` (missing Task ID)
- ❌ WRONG: `- [ ] T001 [US1] Create model` (missing file path)

### Task Organization

1. **From User Stories (spec.md)** - PRIMARY ORGANIZATION:
   - Each user story (P1, P2, P3...) gets its own phase
   - Map all related components to their story:
     - Models needed for that story
     - Services needed for that story
     - Endpoints/UI needed for that story
     - If tests requested: Tests specific to that story
   - Mark story dependencies (most stories should be independent)

2. **From Contracts**:
   - Map each contract/endpoint → to the user story it serves
   - If tests requested: Each contract → contract test task [P] before implementation in that story's phase

3. **From Data Model**:
   - Map each entity to the user story(ies) that need it
   - If entity serves multiple stories: Put in earliest story or Setup phase
   - Relationships → service layer tasks in appropriate story phase

4. **From Setup/Infrastructure**:
   - Shared infrastructure → Setup phase (Phase 1)
   - Foundational/blocking tasks → Foundational phase (Phase 2)
   - Story-specific setup → within that story's phase

### Phase Structure

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites - MUST complete before user stories)
- **Phase 3+**: User Stories in priority order (P1, P2, P3...)
  - Within each story: Tests (if requested) → Models → Services → Endpoints → Integration
  - Each phase should be a complete, independently testable increment
- **Final Phase**: Polish & Cross-Cutting Concerns

## Response snapshot

I have successfully generated the tasks.md file for the Physical AI & Humanoid Robotics Interactive Textbook project. The tasks are organized according to the user stories from the specification, with proper dependencies and parallel execution opportunities identified.

The tasks file includes:

1. Phase 1: Setup tasks for project initialization
2. Phase 2: Foundational tasks that block all user stories
3. Phase 3: User Story 1 (Access Interactive Textbook Content - P1) with 23 MDX files and related implementation tasks
4. Phase 4: User Story 5 (Secure Registration and Authentication - P1)
5. Phase 5: User Story 4 (Interactive Q&A with AI Assistant - P2)
6. Phase 6: User Story 2 (Personalize Learning Experience - P2)
7. Phase 7: User Story 3 (Multilingual Learning Support - P3)
8. Final Phase: Polish and cross-cutting concerns

The tasks follow the required format with:
- Checkbox
- Sequential task IDs (T001, T002, etc.)
- Parallel markers [P] where applicable
- User story labels [US1], [US2], etc. for story-specific tasks
- Clear descriptions with file paths

A total of 116 tasks were created, with clear dependencies, parallel opportunities, and independent test criteria for each user story. The MVP scope includes User Story 1 (access to textbook content).

## Outcome

- ✅ Impact: Generated comprehensive task list with 116 specific tasks organized by user story
- 🧪 Tests: Task format validated against checklist requirements (checkbox, ID, labels, file paths)
- 📁 Files: 1 file created (tasks.md) 
- 🔁 Next prompts: Ready for implementation of individual tasks
- 🧠 Reflection: Tasks are structured for parallel development and independent testing

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): Standard task generation workflow
- Next experiment (smallest change to try): Begin implementation with T001 (project structure setup)
