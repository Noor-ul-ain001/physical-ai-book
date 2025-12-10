# Quickstart Guide: Physical AI & Humanoid Robotics Interactive Textbook

**Feature**: Physical AI & Humanoid Robotics Interactive Textbook  
**Date**: 2025-12-10

## Overview

This guide provides instructions for setting up, running, and contributing to the Physical AI & Humanoid Robotics Interactive Textbook project. The project is a comprehensive educational platform with Docusaurus frontend, AI-powered personalization, and RAG-based chatbot.

## Prerequisites

- Node.js 18+ with npm
- Python 3.11+
- Git
- Access to Google Cloud Platform (for Gemini APIs)
- Access to Qdrant Cloud (for vector storage)
- Neon Serverless Postgres account

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/physical-ai-book.git
cd physical-ai-book
```

### 2. Install Dependencies

```bash
# Install frontend dependencies
npm install

# Setup Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn python-multipart google-generativeai qdrant-client
```

### 3. Environment Variables

Create a `.env` file in the project root with the following:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DB_URL=your_neon_database_connection_string
BETTER_AUTH_SECRET=your_auth_secret
NEXTAUTH_URL=http://localhost:3000
```

## Running the Application

### 1. Start the Backend API Server

```bash
cd api
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### 2. Start the Docusaurus Frontend (in a new terminal)

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Chat API
- `POST /api/chat` - Send message to the chatbot
- `POST /api/highlight-chat` - Send message with selected text as context

### Authentication
Better-Auth provides authentication endpoints automatically.

### Content Indexing
- `POST /api/index` - Run the indexing process (requires admin access)

## Development Workflow

### 1. Adding New Content

1. Create new MDX files in the `docs/` directory following the structure:
   ```
   docs/
   ├── intro/
   │   ├── 00-welcome.mdx
   │   ├── 01-foundations.mdx
   │   └── 02-hardware-guide.mdx
   ├── module1-ros2/
   │   ├── 01-overview.mdx
   │   ├── 02-nodes-topics-services.mdx
   │   └── ...
   ```

2. Update `sidebars.js` to include the new content in navigation

3. Run content indexing after adding new content:
   ```bash
   npm run index
   ```

### 2. Running the Indexing Process

```bash
npm run index
```

This command will:
1. Parse all MDX files in the `docs/` directory
2. Split content using RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
3. Generate embeddings using Gemini embedding-001
4. Upsert to Qdrant Cloud collection "physical-ai-book"

### 3. Running Tests

```bash
# Frontend tests
npm test

# Backend tests
cd api
python -m pytest
```

## Key Components

### 1. Docusaurus Frontend
- Located in `docs/` directory with 23 MDX files
- Custom theme components in `src/theme/`
- Interactive components in `src/components/`

### 2. Backend API
- FastAPI application in `api/` directory
- Handles chat requests, indexing, and other backend operations
- Connects to Qdrant Cloud for vector search and Neon Postgres for user data

### 3. Authentication System
- Better-Auth implementation in `better-auth/` directory
- Collects user profile information during multi-step signup
- Stores profile data in Neon Postgres

### 4. Content Personalization
- Implemented via `ChapterControls.tsx` component
- Adapts content complexity based on user profile using Gemini 1.5 Flash
- Provides Urdu translation functionality

## Configuration Files

- `docusaurus.config.ts` - Docusaurus configuration
- `sidebars.js` - Navigation structure
- `tsconfig.json` - TypeScript configuration
- `.env.example` - Example environment variables file

## Deployment

### 1. Frontend (Vercel)
The Docusaurus frontend is configured for deployment on Vercel. Use the Vercel CLI:

```bash
vercel
```

### 2. Backend API
The FastAPI backend can be deployed as serverless functions on Vercel or as a container on other platforms.

### 3. Database & Vector Storage
- Neon Serverless Postgres for user data
- Qdrant Cloud Free Tier for vector embeddings

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If experiencing rate limit errors, check your API usage in the GCP and Qdrant dashboards

2. **Indexing Failures**: Make sure your GEMINI_API_KEY and QDRANT credentials are correctly set

3. **Authentication Issues**: Check that BETTER_AUTH_SECRET matches between server and client environments

### Useful Commands

```bash
# Build the frontend for production
npm run build

# Serve the built frontend locally
npm run serve

# Run content indexing
npm run index

# Check for linting issues
npm run lint

# Format code
npm run format
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `npm test`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Submit a pull request