# Quick Start Guide

Get your Physical AI textbook chatbot up and running in minutes!

## Option 1: Simple Setup (No RAG)

Perfect for testing and development. Uses keyword-based responses.

```bash
# 1. Install dependencies
npm install
cd api && pip install fastapi uvicorn python-dotenv pydantic

# 2. Start backend (keyword-based)
cd api
python simple_main.py

# 3. Start frontend (in another terminal)
npm start

# 4. Open http://localhost:3000
# Click the 🤖 icon and start chatting!
```

## Option 2: Full RAG Setup (Recommended)

Get intelligent, context-aware responses powered by Gemini AI and your textbook content.

### Prerequisites
- Google Cloud account with Gemini API access
- Qdrant Cloud account (free tier available)

### Setup Steps

**1. Configure API Keys**

Run the interactive setup:
```bash
python scripts/setup_rag.py
```

Or manually edit `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key
```

**2. Install Python Dependencies**
```bash
cd api
pip install -r requirements.txt
```

**3. Index Your Content**
```bash
# From project root
python scripts/index_content.py
```

This processes all textbook content (~5-10 minutes):
- Splits content into chunks
- Generates embeddings with Gemini
- Uploads to Qdrant vector database

**4. Start RAG-Enabled Backend**
```bash
cd api
python rag_main.py
```

**5. Start Frontend**
```bash
# In another terminal
npm start
```

**6. Test It Out!**

Visit http://localhost:3000 and try:
- "What is ROS2 and why is it used?"
- "Explain how Isaac Sim works"
- "How does vSLAM enable robot navigation?"
- "What's the difference between digital twins and simulation?"

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3000
npx kill-port 3000

# Kill process on port 8000
npx kill-port 8000
```

### "RAG services not configured"
Check your `.env` file has valid API keys. Test them:
```bash
# Test health endpoint
curl http://localhost:8000/health
```

### Indexing Failed
Make sure:
1. API keys are correct in `.env`
2. You have internet connection
3. Qdrant cluster is active

### Chatbot Not Appearing
1. Clear browser cache
2. Check browser console for errors
3. Restart the dev server

## What's Next?

- **Customize**: Edit prompts in `api/rag_main.py`
- **Add Content**: Add new MDX files to `docs/` and re-index
- **Deploy**: Follow `RAG_SETUP.md` for production deployment
- **Extend**: Add user authentication, conversation history, etc.

## Architecture

```
User Browser (http://localhost:3000)
     ↓
Docusaurus Frontend + React Chatbot
     ↓ (API calls)
FastAPI Backend (http://localhost:8000)
     ↓
┌────────────────┬──────────────────┐
│                │                  │
│  Gemini API    │   Qdrant Cloud   │
│  (Embeddings   │   (Vector DB)    │
│  + Generation) │                  │
└────────────────┴──────────────────┘
```

## Files Overview

```
📁 Project Root
├── 📄 QUICKSTART.md          ← You are here!
├── 📄 RAG_SETUP.md           ← Detailed RAG setup
├── 📄 .env                   ← API keys (don't commit!)
├── 📁 scripts/
│   ├── setup_rag.py         ← Interactive setup helper
│   └── index_content.py     ← Content indexing script
├── 📁 api/
│   ├── simple_main.py       ← Simple backend (no RAG)
│   ├── rag_main.py          ← Full RAG backend
│   └── requirements.txt     ← Python dependencies
├── 📁 docs/                  ← Your textbook content
└── 📁 src/
    ├── components/
    │   └── Chatbot.tsx      ← Chatbot UI
    └── theme/
        └── Root.tsx         ← Global integration
```

## Support

Need help? Check:
- `RAG_SETUP.md` for detailed RAG setup
- Terminal logs for error messages
- Browser console for frontend issues
- [Gemini API Docs](https://ai.google.dev/docs)
- [Qdrant Docs](https://qdrant.tech/documentation/)

Happy learning! 🚀
