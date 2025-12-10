# Physical AI & Humanoid Robotics – An Interactive Textbook with Personalised RAG Chatbot

An interactive textbook focused on Physical AI and Humanoid Robotics with personalized learning experience, multilingual support, and an AI-powered RAG chatbot to answer questions about the content.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Development](#development)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

- **Interactive Textbook**: Comprehensive educational content on Physical AI and Humanoid Robotics
- **23 Expert-Level MDX Chapters**: Organized across 5 modules covering ROS2, Digital Twins, Isaac Sim, and Vision-Language-Action models
- **Personalized Learning**: Content adapts based on user's skill level and hardware capabilities
- **Multilingual Support**: Real-time translation to Urdu using Gemini 1.5 Flash
- **AI-Powered RAG Chatbot**: Ask questions about the textbook content and receive answers with source citations
- **Highlight-to-Chat**: Right-click on any text to ask specific questions about that content
- **Responsive Design**: Works on all device sizes with NVIDIA-inspired theme

## Tech Stack

- **Frontend**: Docusaurus v3 with MDX support
- **Backend**: FastAPI
- **Authentication**: Better-Auth with email/password and Google OAuth
- **Database**: Neon Serverless Postgres
- **Vector Database**: Qdrant Cloud
- **AI/ML**: Google Gemini 1.5 Flash and Gemini embedding-001
- **UI Components**: Shadcn/UI and Tailwind CSS
- **Deployment**: Vercel (frontend), with backend options

## Prerequisites

- Node.js 18+ with npm
- Python 3.11+
- Git
- Access to Google Cloud Platform (for Gemini APIs)
- Access to Qdrant Cloud (for vector storage)
- Neon Serverless Postgres account

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/physical-ai-book.git
   cd physical-ai-book
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies** (for backend)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install fastapi uvicorn python-multipart google-generativeai qdrant-client
   ```

4. **Configure environment variables**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your API keys and configuration to `.env`

5. **Initialize the project**
   ```bash
   npm run build
   ```

## Development

1. **Start the Docusaurus development server**
   ```bash
   npm run dev
   ```

2. **Start the backend API server** (in a separate terminal)
   ```bash
   cd api
   uvicorn main:app --reload
   ```

3. **Run the content indexer** (when adding new content)
   ```bash
   npm run index
   ```

## Deployment

### Frontend (Vercel)

1. Connect your repository to Vercel
2. Set the build command to `npm run build`
3. Set the output directory to `build`
4. Add your environment variables in the Vercel dashboard

### Backend

Deploy the FastAPI backend using your preferred platform (Vercel, Docker, etc.).

### Environment Variables

You'll need to set up the following environment variables:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `QDRANT_URL`: Your Qdrant Cloud cluster URL
- `QDRANT_API_KEY`: Your Qdrant Cloud API key
- `NEON_DB_URL`: Your Neon Serverless Postgres connection string
- `BETTER_AUTH_SECRET`: Secret for Better-Auth (generate a random string)

## Project Structure

```
physical-ai-book/
├── docs/                     # All content (23 MDX files with full expert content)
├── src/
│   ├── components/
│   │   ├── ChatBot.tsx       # AI chatbot component
│   │   ├── ChapterControls.tsx # Personalization/translation controls
│   │   └── HighlightContextMenu.tsx # Context menu for text selection
│   └── theme/                # Docusaurus custom theme components
├── api/                      # FastAPI backend services
├── scripts/
│   └── index-to-qdrant.ts    # Indexing script
├── better-auth/              # Complete Better-Auth configuration
├── static/                   # Static assets
├── docusaurus.config.ts      # Docusaurus configuration
├── sidebars.js               # Navigation structure
├── package.json              # Project dependencies
├── tsconfig.json             # TypeScript configuration
└── README.md                 # Setup and deployment instructions
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

---

## License

MIT License - See [LICENSE](LICENSE) for more information.