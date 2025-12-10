# Physical AI & Humanoid Robotics – An Interactive Textbook with Personalised RAG Chatbot

An interactive textbook focused on Physical AI and Humanoid Robotics with personalized learning experience, multilingual support, and an AI-powered RAG chatbot to answer questions about the content.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project creates a complete digital twin system for humanoid robotics, combining:
- Physics-accurate simulation in Isaac Sim
- ROS2 communication and control
- Visual SLAM for navigation and mapping
- Reinforcement learning for behavior learning
- Interactive textbook interface with personalized content
- Multilingual support (Urdu translation)
- AI-powered RAG chatbot for Q&A

## Features

- **Interactive Textbook**: 23 comprehensive chapters covering robotics, AI, and humanoid systems
- **20-DOF Humanoid Model**: Complete physics simulation with accurate dynamics
- **Real-time vSLAM**: Visual SLAM for mapping and localization
- **Personalized Learning**: Content adapts based on user's skill level and goals
- **Multilingual Support**: Instant Urdu translation of content
- **AI-Powered Q&A**: RAG-based chatbot that answers questions with source citations
- **Highlight-to-Chat**: Right-click on text to ask specific questions
- **Reinforcement Learning**: Train complex humanoid behaviors with RL
- **ROS2 Integration**: Full compatibility with ROS2 ecosystem

## Tech Stack

- **Frontend**: Docusaurus v3 with MDX support
- **Backend**: FastAPI
- **Simulation**: Isaac Sim
- **Robotics Framework**: ROS2 Humble Hawksbill
- **Authentication**: Better-Auth with Google OAuth
- **Database**: Neon Serverless Postgres
- **Vector Database**: Qdrant Cloud
- **AI/ML**: Google Gemini 1.5 Flash and embedding-001
- **UI Components**: Shadcn/UI and Tailwind CSS
- **Deployment**: Vercel (frontend), with backend options

## Prerequisites

- Node.js 18+ with npm
- Python 3.11+
- Git
- Access to Google Cloud Platform (for Gemini APIs)
- Access to Qdrant Cloud (for vector storage)
- Neon Serverless Postgres account
- NVIDIA GPU with CUDA support (for Isaac Sim physics)
- Isaac Sim installed and configured

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/physical-ai-book.git
   cd physical-ai-book
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies** (for backend and Isaac Sim integration)
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install backend dependencies
   pip install fastapi uvicorn python-multipart google-generativeai qdrant-client torch
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the project structure**
   ```bash
   npm run build
   ```

## Usage

### Local Development

1. **Start the Docusaurus development server**
   ```bash
   npm run dev
   ```

2. **Start the backend API server** (in separate terminal)
   ```bash
   cd api
   uvicorn main:app --reload
   ```

3. **Run the content indexer** (when adding new content)
   ```bash
   npm run index
   ```

### Building for Production

```bash
npm run build
```

### Running Tests

```bash
# Frontend tests
npm test

# Backend tests
cd api
python -m pytest
```

## Project Structure

```
physical-ai-book/
├── docs/                     # Content chapters (23 MDX files with expert content)
├── src/
│   ├── components/           # React components
│   │   ├── ChatBot.tsx       # AI chatbot interface
│   │   ├── ChapterControls.tsx # Personalization/translation controls
│   │   └── HighlightContextMenu.tsx # Context menu for text selection
│   └── theme/                # Custom Docusaurus theme components
├── api/                      # FastAPI backend services
├── scripts/
│   └── index-to-qdrant.ts    # Content indexing script
├── better-auth/              # Better-Auth configuration
├── static/                   # Static assets
├── docusaurus.config.ts      # Docusaurus configuration
├── sidebars.js               # Navigation structure
├── package.json              # Project dependencies
├── tsconfig.json             # TypeScript configuration
└── README.md                 # This file
```

## Development

### Adding New Content

1. Create new MDX files in the `docs/` directory
2. Update `sidebars.js` to include new content in navigation
3. Run content indexing: `npm run index`
4. Test locally with `npm run dev`

### Custom Components

Custom React components in `src/components/` can be used in MDX files:

```md
import CustomComponent from '@site/src/components/CustomComponent';

<CustomComponent />
```

### Environment Configuration

The system uses these environment variables (defined in `.env.example`):

- `GEMINI_API_KEY`: Google Gemini API key
- `QDRANT_URL`: Qdrant Cloud URL
- `QDRANT_API_KEY`: Qdrant Cloud API key
- `NEON_DB_URL`: Neon Postgres connection string
- `BETTER_AUTH_SECRET`: Better-Auth secret

## Deployment

### Frontend Deployment (Vercel)

1. Connect your repository to Vercel
2. Set build command to `npm run build`
3. Set output directory to `build`
4. Add environment variables in Vercel dashboard

### Backend Deployment

The FastAPI backend can be deployed to various platforms:

- **Vercel Functions**: For serverless deployment
- **Self-hosted**: Using Docker or bare metal
- **Other Cloud Platforms**: AWS, GCP, Azure

### Complete Deployment Setup

For full production deployment with all features:

1. Deploy frontend to Vercel
2. Deploy backend to cloud platform
3. Set up Neon Postgres database
4. Configure Qdrant Cloud collection
5. Add custom domain with SSL certificate

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Check your Google Cloud and Qdrant usage dashboards
2. **Slow Performance**: Verify GPU utilization and reduce content complexity if needed
3. **Authentication Issues**: Ensure all Better-Auth configuration is correct
4. **Indexing Failures**: Check API keys and network connectivity

### Performance Optimization

- Use CDN for static assets
- Implement caching for API responses
- Optimize vector database queries
- Reduce image sizes in content

### Debugging

- Enable verbose logging in backend settings
- Check browser console for frontend issues
- Validate all environment variables are set
- Verify Isaac Sim and ROS2 integration

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Guidelines

- Follow ESLint and Prettier formatting rules
- Write tests for new functionality
- Document new features in the appropriate MDX files
- Maintain performance and accessibility standards

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for Isaac Sim and Omniverse platform
- Google for Gemini APIs
- The open-source robotics community
- Contributors and beta testers

---

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

Visit `http://localhost:3000` to view the interactive textbook.

For more details on any specific component, see the individual documentation files in the `docs/` directory.