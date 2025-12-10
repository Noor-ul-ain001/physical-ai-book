# Feature Specification: Physical AI & Humanoid Robotics Interactive Textbook

**Feature Branch**: `004-physical-ai-book`
**Created**: 2025-12-10
**Status**: Draft
**Input**: User description: "PROJECT TITLE: Physical AI & Humanoid Robotics – An Interactive Textbook with Personalised RAG Chatbot CORE REQUIREMENTS (ALL MUST BE IMPLEMENTED EXACTLY): 1. Docusaurus v3 Site - Title: \"Physical AI & Humanoid Robotics\" - Classic preset with custom NVIDIA-inspired theme (black/green/dark mode first) - Full docs-only mode (no blog, no versions) - Exact folder and file structure as specified (see below) 2. Exact Content Structure (1:1 match required) docs/ ├── intro/ │ ├── 00-welcome.mdx → Full Quarter Overview, Why Physical AI Matters, Learning Outcomes │ ├── 01-foundations.mdx → Weeks 1–2 content (foundations, sensors, humanoid landscape) │ └── 02-hardware-guide.mdx → Complete hardware guide with all tables, cloud vs on-prem, Jetson kits, Unitree options ├── module1-ros2/ │ ├── 01-overview.mdx │ ├── 02-nodes-topics-services.mdx │ ├── 03-rclpy-python-bridge.mdx │ ├── 04-urdf-humanoids.mdx │ └── 05-project.mdx ├── module2-digital-twin/ │ ├── 01-gazebo-basics.mdx │ ├── 02-urdf-sdf.mdx │ ├── 03-sensors-simulation.mdx │ ├── 04-unity-visualization.mdx │ └── 05-project.mdx ├── module3-isaac/ │ ├── 01-isaac-sim.mdx │ ├── 02-isaac-ros.mdx │ ├── 03-vslam-navigation.mdx │ ├── 04-reinforcement-learning.mdx │ └── 05-project.mdx ├── module4-vla/ │ ├── 01-vision-language-action.mdx │ ├── 02-whisper-voice-commands.mdx │ ├── 03-llm-task-planning.mdx │ ├── 04-capstone-project.mdx │ └── 05-final-deployment.mdx Every MDX file must contain real, expert-level educational content with working code examples (ROS2 Python, URDF, launch files, Gazebo SDF, Isaac Sim scripts, Gemini + Whisper VLA pipelines), Mermaid diagrams, and embedded YouTube/Vimeo links where relevant. 3. Authentication – Better-Auth (latest) - Full implementation of https://www.better-auth.com/ - Email/password + Google OAuth - Multi-step signup wizard that asks: • Hardware (RTX 4070+, Jetson, real robot?) • Software background (Python years, ROS experience, Linux level, Isaac/RL exposure) • Primary goal (Learning | Building real humanoid | Research) - All answers saved as user metadata in Neon Serverless Postgres 4. Per-Chapter Floating Controls (logged-in users only) Create reusable React component <ChapterControls /> injected at the top of every doc page except intro: - Button 1 → \"Personalize this chapter\" → modal → updates user profile → on every future visit, difficult sections are automatically rewritten/simplified in real-time using gemini-1.5-flash based on their stored skill level - Button 2 → \"اردو میں ترجمہ کریں / Translate to Urdu\" → instantly translates entire current chapter to natural Urdu via gemini-1.5-flash (client-side, streaming) 5. Embedded RAG Chatbot (bottom-right, always visible) - Beautiful Shadcn/UI + Tailwind chat interface - Powered by gemini-1.5-flash (latest) and Gemini embedding-001 - Vector store: Qdrant Cloud Free Tier (collection: physical-ai-book) - All MDX files parsed and indexed at build time (chunk_size=1000, overlap=200) - Features: • Normal chat with full book knowledge • Highlight any text → right-click → \"Ask about this selection only\" → answer uses ONLY that text as context • Logged-in users get personalized answers using their hardware/software profile from DB • Full conversation history persisted per user in Neon Postgres • Every answer cites sources with clickable links to exact section 6. Backend - FastAPI (/api route) OR Vercel Edge Functions (you choose the cleaner one) - Endpoints: /api/chat (streaming), /api/highlight-chat 7. Indexing Script - scripts/index-to-qdrant.ts (Node.js + TypeScript) - Runs on npm run index → parses all MDX → creates embeddings → upserts to Qdrant with metadata 8. Environment Variables (.env.example + .env) GEMINI_API_KEY= QDRANT_URL= QDRANT_API_KEY= NEON_DB_URL= BETTER_AUTH_SECRET= 9. Final Repository Structure (exact) physical-ai-book/ ├── docs/ → all 23 MDX files with full expert content ├── src/ │ ├── components/ │ │ ├── ChatBot.tsx │ │ ├── ChapterControls.tsx │ │ └── HighlightContextMenu.tsx │ └── theme/ ├── api/ → FastAPI or edge functions ├── scripts/ │ └── index-to-qdrant.ts ├── better-auth/ → complete config + signup wizard ├── static/ ├── docusaurus.config.ts ├── sidebars.js ├── package.json ├── tsconfig.json └── README.md → full setup, deploy to Vercel, Neon, Qdrant instructions"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Interactive Textbook Content (Priority: P1)

As a learner interested in physical AI and humanoid robotics, I want to access comprehensive educational content through an interactive textbook website so that I can learn about ROS2, digital twins, Isaac Sim, and Vision-Language-Action models in a structured way.

**Why this priority**: This is the core value proposition of the product - providing educational content in an accessible and interactive format.

**Independent Test**: A new user can navigate the site, read chapters, and understand the fundamental concepts of physical AI and humanoid robotics without any additional features.

**Acceptance Scenarios**:

1. **Given** I am on the homepage of the Physical AI & Humanoid Robotics textbook, **When** I navigate to different modules and chapters, **Then** I can read all educational content with proper formatting, code examples, diagrams, and embedded media.

2. **Given** I am viewing a chapter, **When** I encounter complex concepts, **Then** I can understand them through the provided explanations, code examples, and visual aids.

---

### User Story 2 - Personalize Learning Experience (Priority: P2)

As a learner with varying skill levels in robotics and AI, I want to personalize my learning journey by providing my background information and hardware capabilities, so that the content adapts to my knowledge level and available resources.

**Why this priority**: Personalization significantly improves learning outcomes by adapting content complexity to individual needs.

**Independent Test**: A user can complete the profile setup during registration and see content adjustments based on their profile information during future visits.

**Acceptance Scenarios**:

1. **Given** I am a registered user, **When** I visit a chapter for the first time, **Then** I am prompted to set up my profile with hardware capabilities and skill levels.

2. **Given** I have set up my profile, **When** I revisit chapters, **Then** complex sections are automatically simplified based on my skill level.

3. **Given** I have set up my profile, **When** I use the RAG chatbot, **Then** responses are tailored to my hardware and skill level.

---

### User Story 3 - Multilingual Learning Support (Priority: P3)

As a learner who is more comfortable in Urdu, I want to translate textbook content into Urdu so that I can better understand complex concepts and materials.

**Why this priority**: Providing content in multiple languages significantly expands the accessibility of the educational resource.

**Independent Test**: A user can select the Urdu translation option and read an entire chapter in their preferred language.

**Acceptance Scenarios**:

1. **Given** I am viewing a chapter, **When** I click the "Translate to Urdu" button, **Then** the entire chapter content is translated to natural Urdu.

2. **Given** I have switched to Urdu, **When** I navigate to a different chapter, **Then** I can continue reading in Urdu.

---

### User Story 4 - Interactive Q&A with AI Assistant (Priority: P2)

As a learner seeking clarification on complex topics, I want to ask questions about the textbook content to an AI assistant that has full knowledge of the material so that I can get immediate, accurate answers.

**Why this priority**: An AI assistant significantly enhances the learning experience by providing immediate feedback and personalized explanations.

**Independent Test**: A user can ask questions about the textbook content and receive accurate, helpful responses with source citations.

**Acceptance Scenarios**:

1. **Given** I am reading a chapter, **When** I activate the chatbot and ask a question about the textbook content, **Then** I receive a relevant answer based on the book's content with source citations.

2. **Given** I have highlighted text in a chapter, **When** I right-click and select "Ask about this selection only", **Then** the chatbot answers using only that specific text as context.

3. **Given** I am a logged-in user with profile information, **When** I ask questions to the chatbot, **Then** answers are personalized based on my hardware and skill level.

---

### User Story 5 - Secure Registration and Authentication (Priority: P1)

As a new learner, I want to register and authenticate securely using email/password or Google OAuth so that I can access personalized features while maintaining my privacy.

**Why this priority**: User authentication is a prerequisite for personalization and profile management features.

**Independent Test**: A new user can register using either email/password or Google OAuth and then log in to access the system.

**Acceptance Scenarios**:

1. **Given** I am a new user, **When** I register with email and password, **Then** I can successfully create an account and log in.

2. **Given** I have a Google account, **When** I choose Google OAuth for registration, **Then** I can successfully create an account and log in.

3. **Given** I am registering, **When** I provide profile information during the multi-step signup, **Then** this information is securely stored and accessible for personalization.

---

### Edge Cases

- What happens when the AI translation service is temporarily unavailable? (System should fall back to original language with a notice)
- How does the system handle users with slow internet connections when loading rich media content? (System should provide options to disable rich content)
- What if the Qdrant vector database is temporarily down? (System should gracefully degrade with a notice to users)
- How does the system handle concurrent users accessing the RAG chatbot? (System should scale to handle load)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a Docusaurus v3 website with the title "Physical AI & Humanoid Robotics"
- **FR-002**: System MUST implement a dark/light mode interface with NVIDIA-style green/black accents
- **FR-003**: System MUST provide 23 MDX content pages organized in 5 modules as specified in the content structure
- **FR-004**: System MUST implement Better-Auth authentication with email/password and Google OAuth
- **FR-005**: System MUST collect user profile information during registration including hardware, software background, and goals
- **FR-006**: System MUST store user profile information in a database (Neon Serverless Postgres)
- **FR-007**: System MUST provide a "Personalize this chapter" feature that adapts content based on user profile
- **FR-008**: System MUST provide a "Translate to Urdu" feature for all content chapters
- **FR-009**: System MUST implement an embedded RAG chatbot that can answer questions about the textbook content
- **FR-010**: System MUST allow users to ask questions about selected text only via right-click context menu
- **FR-011**: System MUST provide personalized chatbot responses based on user profile information
- **FR-012**: System MUST persist conversation history for logged-in users
- **FR-013**: System MUST cite sources with clickable links back to exact sections in responses
- **FR-014**: System MUST index all MDX content in a vector database (Qdrant Cloud) at build time
- **FR-015**: System MUST provide an indexing script that can parse MDX files and create embeddings
- **FR-016**: System MUST provide API endpoints for chat functionality (streaming and highlight-based)

### Key Entities

- **User**: Represents a registered user with authentication credentials and profile information
- **UserProfile**: Contains hardware capabilities, software background, learning goals, and skill levels
- **Chapter**: Represents a section of the textbook with educational content, code examples, and diagrams
- **Conversation**: Represents a user's chat history with the AI assistant
- **Message**: An individual message within a conversation between user and AI assistant
- **ContentSource**: Represents a specific section of textbook content that can be referenced as a source

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can register and authenticate within 2 minutes using either email/password or Google OAuth
- **SC-002**: 95% of users can successfully navigate to any chapter and read the content without technical issues
- **SC-003**: 90% of users complete the multi-step profile setup during registration
- **SC-004**: Translation to Urdu is available for 100% of chapter content within 5 seconds of clicking the button
- **SC-005**: The RAG chatbot provides accurate answers to 90% of questions about textbook content with proper source citations
- **SC-006**: Personalized content adaptation is applied to 100% of chapters based on user skill level
- **SC-007**: The system handles 1000 concurrent users without performance degradation
- **SC-008**: 95% of users find the personalized chatbot responses helpful for their specific hardware and skill level
