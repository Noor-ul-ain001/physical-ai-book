---

description: "Task list template for feature implementation"
---

# Tasks: Physical AI & Humanoid Robotics Interactive Textbook

**Input**: Design documents from `/specs/004-physical-ai-book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /sp.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan with all required directories
- [ ] T002 [P] Initialize package.json with required dependencies for Docusaurus v3, FastAPI, Better-Auth
- [ ] T003 [P] Create .env.example and .env with all required environment variables
- [ ] T004 [P] Set up TypeScript configuration (tsconfig.json)
- [ ] T005 [P] Create README.md with full setup and deployment instructions
- [ ] T006 Set up initial Docusaurus v3 project with classic preset
- [ ] T007 [P] Configure basic dark/light mode with NVIDIA-inspired theme (green/black accents)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T008 [P] Initialize Better-Auth with email/password and Google OAuth
- [ ] T009 [P] Set up Neon Serverless Postgres connection
- [ ] T010 [P] Create User and UserProfile table schemas in database
- [ ] T011 [P] Create backend API structure with FastAPI
- [ ] T012 [P] Set up Qdrant Cloud collection "physical-ai-book"
- [ ] T013 [P] Create Gemini API integration utilities
- [ ] T014 Create docusaurus.config.ts with custom NVIDIA-themed configuration
- [ ] T015 Create sidebar.js with the exact module and chapter structure
- [ ] T016 [P] Create indexing script at scripts/index-to-qdrant.ts
- [ ] T017 [P] Create api/chat endpoint in backend
- [ ] T018 [P] Create api/highlight-chat endpoint in backend
- [ ] T019 Create better-auth configuration with multi-step signup wizard

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access Interactive Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Enable users to access comprehensive educational content through the interactive textbook website

**Independent Test**: A new user can navigate the site, read chapters, and understand the fundamental concepts of physical AI and humanoid robotics without any additional features.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T020 [P] [US1] Contract test for homepage navigation in tests/contract/test_navigation.py
- [ ] T021 [P] [US1] Integration test for chapter access in tests/integration/test_content_access.py

### Implementation for User Story 1

- [ ] T022 [P] [US1] Write docs/intro/00-welcome.mdx with Quarter Overview, Why Physical AI Matters, Learning Outcomes
- [ ] T023 [P] [US1] Write docs/intro/01-foundations.mdx with Weeks 1-2 content (foundations, sensors, humanoid landscape)
- [ ] T024 [P] [US1] Write docs/intro/02-hardware-guide.mdx with complete hardware guide, tables, cloud vs on-prem, Jetson kits, Unitree options
- [ ] T025 [P] [US1] Write docs/module1-ros2/01-overview.mdx with ROS2 overview content
- [ ] T026 [P] [US1] Write docs/module1-ros2/02-nodes-topics-services.mdx with nodes, topics, services content
- [ ] T027 [P] [US1] Write docs/module1-ros2/03-rclpy-python-bridge.mdx with rclpy Python bridge content
- [ ] T028 [P] [US1] Write docs/module1-ros2/04-urdf-humanoids.mdx with URDF for humanoids content
- [ ] T029 [P] [US1] Write docs/module1-ros2/05-project.mdx with ROS2 project content
- [ ] T030 [P] [US1] Write docs/module2-digital-twin/01-gazebo-basics.mdx with Gazebo basics content
- [ ] T031 [P] [US1] Write docs/module2-digital-twin/02-urdf-sdf.mdx with URDF-SDF content
- [ ] T032 [P] [US1] Write docs/module2-digital-twin/03-sensors-simulation.mdx with sensors simulation content
- [ ] T033 [P] [US1] Write docs/module2-digital-twin/04-unity-visualization.mdx with Unity visualization content
- [ ] T034 [P] [US1] Write docs/module2-digital-twin/05-project.mdx with digital twin project content
- [ ] T035 [P] [US1] Write docs/module3-isaac/01-isaac-sim.mdx with Isaac Sim content
- [ ] T036 [P] [US1] Write docs/module3-isaac/02-isaac-ros.mdx with Isaac ROS content
- [ ] T037 [P] [US1] Write docs/module3-isaac/03-vslam-navigation.mdx with vSLAM navigation content
- [ ] T038 [P] [US1] Write docs/module3-isaac/04-reinforcement-learning.mdx with reinforcement learning content
- [ ] T039 [P] [US1] Write docs/module3-isaac/05-project.mdx with Isaac project content
- [ ] T040 [P] [US1] Write docs/module4-vla/01-vision-language-action.mdx with VLA content
- [ ] T041 [P] [US1] Write docs/module4-vla/02-whisper-voice-commands.mdx with Whisper voice commands content
- [ ] T042 [P] [US1] Write docs/module4-vla/03-llm-task-planning.mdx with LLM task planning content
- [ ] T043 [P] [US1] Write docs/module4-vla/04-capstone-project.mdx with capstone project content
- [ ] T044 [P] [US1] Write docs/module4-vla/05-final-deployment.mdx with final deployment content
- [ ] T045 [US1] Update sidebar.js to include all 23 MDX files with proper navigation
- [ ] T046 [US1] Add Mermaid diagrams to all MDX files where relevant
- [ ] T047 [US1] Add embedded YouTube/Vimeo links to MDX files where relevant
- [ ] T048 [US1] Add working code examples (ROS2 Python, URDF, launch files, Gazebo SDF, Isaac Sim scripts) to MDX files
- [ ] T049 [US1] Add styling for code blocks and formatting to docusaurus.config.ts
- [ ] T050 [US1] Test navigation and content access across all chapters

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 5 - Secure Registration and Authentication (Priority: P1)

**Goal**: Enable users to register and authenticate securely using email/password or Google OAuth

**Independent Test**: A new user can register using either email/password or Google OAuth and then log in to access the system.

### Tests for User Story 5 (OPTIONAL - only if tests requested) ⚠️

- [ ] T051 [P] [US5] Contract test for registration endpoint in tests/contract/test_auth.py
- [ ] T052 [P] [US5] Integration test for login flow in tests/integration/test_auth_flow.py

### Implementation for User Story 5

- [ ] T053 [US5] Complete Better-Auth multi-step signup wizard implementation
- [ ] T054 [US5] Create database migration for User table
- [ ] T055 [US5] Create database migration for UserProfile table
- [ ] T056 [US5] Implement hardware questions during signup (RTX 4070+, Jetson, real robot?)
- [ ] T057 [US5] Implement software background questions during signup (Python years, ROS experience, etc.)
- [ ] T058 [US5] Implement primary goal questions during signup (Learning | Building | Research)
- [ ] T059 [US5] Store user responses in Neon Postgres as metadata
- [ ] T060 [US5] Implement profile retrieval API endpoint
- [ ] T061 [US5] Test email/password registration and login
- [ ] T062 [US5] Test Google OAuth registration and login
- [ ] T063 [US5] Verify profile information is stored and retrievable

**Checkpoint**: At this point, User Stories 1 AND 5 should both work independently

---

## Phase 5: User Story 4 - Interactive Q&A with AI Assistant (Priority: P2)

**Goal**: Enable users to ask questions about the textbook content to an AI assistant that has full knowledge of the material

**Independent Test**: A user can ask questions about the textbook content and receive accurate, helpful responses with source citations.

### Tests for User Story 4 (OPTIONAL - only if tests requested) ⚠️

- [ ] T064 [P] [US4] Contract test for chat endpoint in tests/contract/test_chat.py
- [ ] T065 [P] [US4] Contract test for highlight-chat endpoint in tests/contract/test_highlight_chat.py
- [ ] T066 [P] [US4] Integration test for chat functionality in tests/integration/test_chat_integration.py

### Implementation for User Story 4

- [ ] T067 [US4] Create ChatBot.tsx component with Shadcn UI and Tailwind styling
- [ ] T068 [US4] Implement chat interface with message history display
- [ ] T069 [US4] Create backend endpoint for /api/chat with streaming response
- [ ] T070 [US4] Implement RAG pipeline to query Qdrant vector store
- [ ] T071 [US4] Create embeddings for all MDX content using Gemini embedding-001
- [ ] T072 [US4] Implement source citation functionality in chat responses
- [ ] T073 [US4] Create backend endpoint for /api/highlight-chat
- [ ] T074 [US4] Implement targeted query functionality using only selected text
- [ ] T075 [US4] Create HighlightContextMenu.tsx component for right-click context menu
- [ ] T076 [US4] Integrate highlight context menu with chat interface
- [ ] T077 [US4] Implement conversation history persistence in Neon Postgres
- [ ] T078 [US4] Create Message model and database operations
- [ ] T079 [US4] Create Conversation model and database operations
- [ ] T080 [US4] Create SourceReference model and database operations
- [ ] T081 [US4] Test basic chat functionality with textbook content
- [ ] T082 [US4] Test highlight-to-chat functionality
- [ ] T083 [US4] Test source citation accuracy
- [ ] T084 [US4] Test conversation history persistence

**Checkpoint**: At this point, User Stories 1, 5 AND 4 should all work independently

---

## Phase 6: User Story 2 - Personalize Learning Experience (Priority: P2)

**Goal**: Enable users to provide their background information and hardware capabilities to personalize the learning journey

**Independent Test**: A user can complete the profile setup during registration and see content adjustments based on their profile information during future visits.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T085 [P] [US2] Contract test for profile update endpoint in tests/contract/test_profile.py
- [ ] T086 [P] [US2] Integration test for profile-based personalization in tests/integration/test_personalization.py

### Implementation for User Story 2

- [ ] T087 [US2] Create ChapterControls.tsx component with personalize button
- [ ] T088 [US2] Create profile modal that asks user's experience in ROS2, Isaac Sim, LLMs
- [ ] T089 [US2] Implement profile saving to database
- [ ] T090 [US2] Update profile API endpoint to PUT /api/profile
- [ ] T091 [US2] Implement content adaptation logic using gemini-1.5-flash
- [ ] T092 [US2] Create function to simplify content based on user's skill level
- [ ] T093 [US2] Personalize chatbot responses based on user's hardware and skill level
- [ ] T094 [US2] Test content personalization based on user profile
- [ ] T095 [US2] Test personalized chatbot responses

**Checkpoint**: At this point, User Stories 1, 5, 4 AND 2 should all work independently

---

## Phase 7: User Story 3 - Multilingual Learning Support (Priority: P3)

**Goal**: Enable users to translate textbook content into Urdu to better understand complex concepts

**Independent Test**: A user can select the Urdu translation option and read an entire chapter in their preferred language.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T096 [P] [US3] Contract test for translation API endpoint in tests/contract/test_translation.py

### Implementation for User Story 3

- [ ] T097 [US3] Add Urdu translation button to ChapterControls.tsx component
- [ ] T098 [US3] Implement Urdu translation functionality using gemini-1.5-flash
- [ ] T099 [US3] Create client-side streaming translation
- [ ] T100 [US3] Test Urdu translation quality and accuracy
- [ ] T101 [US3] Test translation persistence across navigation
- [ ] T102 [US3] Verify translation fallback in case of API unavailability

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T103 [P] Documentation updates in docs/
- [ ] T104 [P] Create comprehensive README with setup, deploy to Vercel, Neon, Qdrant instructions
- [ ] T105 [P] Code cleanup and refactoring
- [ ] T106 [P] Performance optimization across all stories
- [ ] T107 [P] Additional unit tests (if requested) in tests/unit/
- [ ] T108 [P] Security hardening
- [ ] T109 [P] Run quickstart.md validation
- [ ] T110 [P] Performance testing for RAG responses
- [ ] T111 [P] Content accuracy verification process
- [ ] T112 [P] Edge case handling (AI translation unavailability, slow connections, etc.)
- [ ] T113 [P] Final verification: Test Urdu translation on Module 2 Chapter 3
- [ ] T114 [P] Final verification: Test highlight-to-chat with selected URDF code
- [ ] T115 [P] Final verification: Test personalization with different skill levels
- [ ] T116 [P] Final verification: Test concurrent users accessing the RAG chatbot

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 5 (P1)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 4 (P2)**: Depends on US1 (content exists for RAG) - May integrate with US1/US5 but should be independently testable
- **User Story 2 (P2)**: Depends on US5 (authentication) - May integrate with US1/US4 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- MDX files within a module can be written in parallel [P]
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# MDX files can be written in parallel:
Task: "Write docs/intro/00-welcome.mdx with Quarter Overview, Why Physical AI Matters, Learning Outcomes" [P]
Task: "Write docs/intro/01-foundations.mdx with Weeks 1-2 content (foundations, sensors, humanoid landscape)" [P]
Task: "Write docs/intro/02-hardware-guide.mdx with complete hardware guide, tables, cloud vs on-prem, Jetson kits, Unitree options" [P]

# Module files can be written in parallel:
Task: "Write docs/module1-ros2/01-overview.mdx with ROS2 overview content" [P]
Task: "Write docs/module1-ros2/02-nodes-topics-services.mdx with nodes, topics, services content" [P]
Task: "Write docs/module1-ros2/03-rclpy-python-bridge.mdx with rclpy Python bridge content" [P]
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 5 → Test independently → Deploy/Demo
4. Add User Story 4 → Test independently → Deploy/Demo
5. Add User Story 2 → Test independently → Deploy/Demo
6. Add User Story 3 → Test independently → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (content creation)
   - Developer B: User Story 5 (authentication)
   - Developer C: User Story 4 (chatbot)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence